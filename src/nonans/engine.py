"""
nonans.engine
=============
Sentinel — the open monitoring layer.

Three gates, one rule
---------------------
  Gate 1 — Norm proxy (every step, O(n)):
    Cheap norm-deviation check. Near-zero cost. Decides if Gate 2 runs.

  Gate 2 — Structural scan (when proxy fires OR periodic, O(m·n·k)):
    Randomized SVD + entropy-based health scores + three-signal classify.
    Updates the FaultContext for the affected tensor.

  Gate 3 — Context query (when resolver activates, O(1)):
    Resolver calls Sentinel.get_context(tensor_id). Returns FaultContext.
    Resolution becomes informed by the full structural trajectory instead
    of a blind local approximation (clamp/replace/skip).

Thread safety
-------------
Context store protected by threading.RLock.
- Writes:  step_watch()    — training thread
- Reads:   get_context()   — resolver thread (potentially different)
Gates 1 and 2 use per-parameter state without locking (training thread only).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .primitives import (
    collapse_rate,
    norm_deviation,
    randomized_svd_topk,
    shape_score,
    signal_score,
)
from ._torch_bindings import (
    momentum_aligned_to_collapse,
    restoring_force,
)
from .decision import classify
from .protocol import FaultContext, Health, RingBuffer


# ─── Per-parameter internal state ────────────────────────────────────────────

@dataclass
class _ParamState:
    """Mutable state for one tracked parameter. Training-thread only."""
    norm_buf:        RingBuffer
    shape_buf:       RingBuffer
    last_healthy_sv: torch.Tensor
    scan_count:      int = 0

    @classmethod
    def create(cls, window: int) -> "_ParamState":
        return cls(
            norm_buf        = RingBuffer(window * 3),
            shape_buf       = RingBuffer(window),
            last_healthy_sv = torch.empty(0, dtype=torch.float32),
        )


# ─── Sentinel ────────────────────────────────────────────────────────────────

class Sentinel:
    """
    The open monitoring layer.

    Runs in parallel with training. Silent faults become visible and
    classified. The Sentinel feeds FaultContext to the resolver at moment
    of singularity, enabling informed resolution.

    Usage
    -----
    >>> sentinel = Sentinel(model, optimizer)
    >>> # after every optimizer.step():
    >>> sentinel.step_watch()
    >>> # the resolver (potentially on a different thread) calls:
    >>> ctx = sentinel.get_context("net.2.weight")
    >>> sentinel.display()    # optional visual health map

    Parameters
    ----------
    model           : the model being trained (any nn.Module)
    optimizer       : the optimizer (used for momentum-alignment check)
    window          : trajectory window in training steps
    k               : number of singular values to approximate
    proxy_threshold : norm-deviation fraction that triggers Gate 2
    scan_every      : periodic Gate 2 scan interval (steps)
    min_numel       : skip parameters with fewer elements than this
    """

    def __init__(
        self,
        model,
        optimizer,
        window:          int   = 12,
        k:               int   = 8,
        proxy_threshold: float = 0.12,
        scan_every:      int   = 25,
        min_numel:       int   = 16,
    ) -> None:
        self.model           = model
        self.optimizer       = optimizer
        self.window          = window
        self.k               = k
        self.proxy_threshold = proxy_threshold
        self.scan_every      = scan_every
        self.min_numel       = min_numel

        self._param_state: Dict[str, _ParamState]   = {}
        self._context:     Dict[str, FaultContext]  = {}
        self._lock        = threading.RLock()
        self._step: int    = 0

        self._stats = {
            "proxy_fires":    0,
            "svd_calls":      0,
            "real_faults":    0,
            "ghost_faults":   0,
            "context_hits":   0,
            "context_misses": 0,
        }

        self._init_params()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_params(self) -> None:
        for name, param in self.model.named_parameters():
            if self._should_track(param):
                self._param_state[name] = _ParamState.create(self.window)

    def _should_track(self, param: torch.Tensor) -> bool:
        return param.requires_grad and param.numel() >= self.min_numel

    # ── Gate 1: Norm proxy ────────────────────────────────────────────────────

    def _proxy(self, name: str, param: torch.Tensor) -> bool:
        try:
            norm = float(param.data.detach().to(torch.float32).norm())
        except Exception:
            return False

        # NaN-aware: a NaN/Inf norm itself trips the proxy
        if not (norm == norm) or norm == float("inf"):
            self._stats["proxy_fires"] += 1
            return True

        state = self._param_state[name]
        state.norm_buf.append(norm)
        hist = state.norm_buf.as_list()

        dev = norm_deviation(norm, hist, tail=3)
        if dev > self.proxy_threshold:
            self._stats["proxy_fires"] += 1
            return True
        return False

    # ── Gate 2: Structural scan ───────────────────────────────────────────────

    def _scan(self, name: str, param: torch.Tensor) -> Optional[FaultContext]:
        t0 = time.perf_counter()
        state = self._param_state[name]

        try:
            sv = randomized_svd_topk(param.data, self.k)
        except Exception:
            return None

        if sv.numel() == 0:
            return None

        self._stats["svd_calls"] += 1
        state.scan_count += 1

        ss  = shape_score(sv)
        sig = signal_score(sv)
        rf  = restoring_force(param)

        if ss >= 0.72:
            state.last_healthy_sv = sv.clone()

        state.shape_buf.append(ss)
        traj = state.shape_buf.as_tensor()

        rate    = collapse_rate(traj)
        mom_bad = momentum_aligned_to_collapse(param, self.optimizer)

        health, confidence = classify(
            shape=ss, signal=sig, force=rf,
            rate=rate, momentum_bad=mom_bad,
            history_len=len(traj),
        )

        if health == Health.REAL:
            self._stats["real_faults"] += 1
        elif health == Health.GHOST:
            self._stats["ghost_faults"] += 1

        compute_ms = (time.perf_counter() - t0) * 1000.0

        return FaultContext(
            tensor_id                    = name,
            tensor_shape                 = tuple(param.shape),
            step                         = self._step,
            shape_score                  = ss,
            signal_score                 = sig,
            restoring_force              = rf,
            collapse_rate                = rate,
            momentum_aligned_to_collapse = mom_bad,
            health                       = health,
            confidence                   = confidence,
            singular_values              = sv.cpu(),
            last_healthy_sv              = state.last_healthy_sv.cpu().clone(),
            trajectory                   = traj.cpu(),
            compute_time_ms              = compute_ms,
        )

    # ── Main step ─────────────────────────────────────────────────────────────

    def step_watch(self) -> Dict[str, FaultContext]:
        """Call once after optimizer.step(). Runs Gates 1 and 2."""
        self._step += 1
        periodic = (self._step % self.scan_every == 0)

        new_contexts: Dict[str, FaultContext] = {}

        for name, param in self.model.named_parameters():
            if name not in self._param_state or not param.requires_grad:
                continue

            fired = self._proxy(name, param)
            if fired or periodic:
                ctx = self._scan(name, param)
                if ctx is not None:
                    new_contexts[name] = ctx

        if new_contexts:
            with self._lock:
                self._context.update(new_contexts)

        return dict(self._context)

    # ── Gate 3: Context query (resolver-side) ─────────────────────────────────

    def get_context(self, tensor_id: str) -> Optional[FaultContext]:
        """O(1) thread-safe lookup. Called by the resolver at singularity."""
        with self._lock:
            ctx = self._context.get(tensor_id)
        if ctx is not None:
            self._stats["context_hits"] += 1
        else:
            self._stats["context_misses"] += 1
        return ctx

    def get_all_contexts(self) -> Dict[str, FaultContext]:
        with self._lock:
            return dict(self._context)

    # ── Stats and display ─────────────────────────────────────────────────────

    @property
    def step(self) -> int:
        return self._step

    def stats(self) -> dict:
        return {"step": self._step, **self._stats}

    def display(self) -> None:
        """Print health map. Sorted by severity: REAL first, HEALTHY last."""
        W = 76
        print(f"\n{'═' * W}")
        print(
            f"  NONANS SENTINEL  │  step {self._step:>5}  │  "
            f"real {self._stats['real_faults']:>3}  "
            f"ghost {self._stats['ghost_faults']:>3}  "
            f"svd_calls {self._stats['svd_calls']:>4}"
        )
        print(f"{'─' * W}")
        print(f"  {'LAYER':<38}  {'SHAPE':>5}  {'SIG':>5}  {'GRAD':>8}  {'STATE':<10}  {'CONF':>4}")
        print(f"  {'─' * 71}")

        with self._lock:
            contexts = list(self._context.items())

        if not contexts:
            print("  (no scans completed yet)")
            print(f"{'═' * W}\n")
            return

        for name, ctx in sorted(contexts, key=lambda x: x[1].health.severity):
            short   = ("…" + name[-36:]) if len(name) > 37 else name
            n_full  = round(ctx.shape_score  * 4)
            n_sig   = round(ctx.signal_score * 4)
            sh_bar  = "▮" * n_full + "░" * (4 - n_full)
            sig_bar = "▮" * n_sig  + "░" * (4 - n_sig)
            print(
                f"  {short:<38}  {sh_bar:>5}  {sig_bar:>5}  "
                f"{ctx.restoring_force:>8.4f}  "
                f"{ctx.health.icon} {ctx.health.name:<8}  "
                f"{ctx.confidence:>4.2f}"
            )
        print(f"{'═' * W}\n")

    def summary(self) -> str:
        s = self._stats
        return "\n".join([
            f"NoNans Sentinel @ step {self._step}",
            f"  SVD calls      : {s['svd_calls']}",
            f"  Proxy fires    : {s['proxy_fires']}",
            f"  Real faults    : {s['real_faults']}",
            f"  Ghost faults   : {s['ghost_faults']}",
            f"  Context hits   : {s['context_hits']}",
            f"  Context misses : {s['context_misses']}",
        ])
