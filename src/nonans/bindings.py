"""
nonans.bindings
===============
Framework integration surface.

Two public components
---------------------
1. Registry
   Process-global singleton mapping tensor_id → FaultContext.
   Resolvers query this at moment of singularity without needing a
   direct reference to a Sentinel instance.

2. wrap(model, optimizer, **kwargs)
   The single integration surface. Creates a Sentinel, registers it,
   installs the optimizer hook. Returns a WrappedModel that trains
   identically to the original.

Design rule
-----------
The user's training code changes by exactly one line.
Everything else is invisible.

    Before:  model = MyModel()
    After:   model = nonans.wrap(model, optimizer)
"""
from __future__ import annotations

import threading
import weakref
from typing import Dict, Optional

import torch

from .engine import Sentinel
from .protocol import FaultContext


# ─── Registry ─────────────────────────────────────────────────────────────────

class _Registry:
    """
    Process-global mapping: tensor_id → FaultContext via active Sentinels.

    Multiple Sentinels may be registered (multi-model training). Resolvers
    call Registry.get(tensor_id) without holding direct Sentinel references.
    Thread-safe. Designed for high-frequency reads, low-frequency writes.
    """

    def __init__(self) -> None:
        self._sentinels: Dict[int, weakref.ref] = {}
        self._lock      = threading.RLock()

    def register(self, sentinel: Sentinel) -> None:
        wid = id(sentinel)
        with self._lock:
            self._sentinels[wid] = weakref.ref(
                sentinel, lambda _: self._deregister(wid)
            )

    def _deregister(self, wid: int) -> None:
        with self._lock:
            self._sentinels.pop(wid, None)

    def get(self, tensor_id: str) -> Optional[FaultContext]:
        """
        Returns the most-confident context for tensor_id across all sentinels.
        None if no sentinel has context for this tensor.
        """
        best: Optional[FaultContext] = None
        with self._lock:
            sentinels = list(self._sentinels.values())
        for ref in sentinels:
            s = ref()
            if s is None:
                continue
            ctx = s.get_context(tensor_id)
            if ctx is None:
                continue
            if best is None or ctx.confidence > best.confidence:
                best = ctx
        return best

    def clear(self) -> None:
        with self._lock:
            self._sentinels.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._sentinels)


Registry = _Registry()


# ─── Wrapped model ────────────────────────────────────────────────────────────

class WrappedModel:
    """
    Transparent wrapper. Identical interface to the inner model, plus the
    sentinel methods (step_watch, display, get_context).
    """

    def __init__(self, model, sentinel: Sentinel) -> None:
        # Direct __dict__ access avoids the __getattr__ recursion trap.
        self.__dict__["_model"]    = model
        self.__dict__["_sentinel"] = sentinel

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate all unknown attribute access to the wrapped model."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_model"], name)

    def parameters(self, recurse: bool = True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, **kwargs):
        return self._model.state_dict(**kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._model.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self

    def to(self, *args, **kwargs):
        self._model.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        self._model.cuda(device)
        return self

    def cpu(self):
        self._model.cpu()
        return self

    # ── Sentinel interface ────────────────────────────────────────────────────

    def step_watch(self):
        """Call after optimizer.step(). The only new call in the loop (auto-hooked by default)."""
        return self._sentinel.step_watch()

    def display(self) -> None:
        self._sentinel.display()

    def get_context(self, tensor_id: str) -> Optional[FaultContext]:
        return self._sentinel.get_context(tensor_id)

    @property
    def sentinel(self) -> Sentinel:
        return self._sentinel

    @property
    def model(self):
        return self._model

    def summary(self) -> str:
        return self._sentinel.summary()


# ─── Optimizer auto-hook ──────────────────────────────────────────────────────

class _OptimizerHook:
    """Patches optimizer.step() to automatically call sentinel.step_watch()."""

    def __init__(self, optimizer, sentinel: Sentinel) -> None:
        self._optimizer = optimizer
        self._sentinel  = sentinel
        self._original_step = optimizer.step

    def install(self) -> None:
        optimizer = self._optimizer
        sentinel  = self._sentinel
        original  = self._original_step

        def patched_step(closure=None):
            result = original(closure)
            try:
                sentinel.step_watch()
            except Exception:
                # Sentinel errors must NEVER crash training. This is a hard rule.
                pass
            return result

        optimizer.step = patched_step

    def uninstall(self) -> None:
        self._optimizer.step = self._original_step


# ─── wrap() — the entire API ──────────────────────────────────────────────────

def wrap(
    model,
    optimizer,
    auto_hook:       bool  = True,
    window:          int   = 12,
    k:               int   = 8,
    proxy_threshold: float = 0.12,
    scan_every:      int   = 25,
    min_numel:       int   = 16,
) -> WrappedModel:
    """
    Single integration surface for the open monitoring layer.

    Creates Sentinel, registers it, optionally installs the optimizer hook.

    Example
    -------
    >>> import torch, nonans
    >>> model     = MyModel()
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> model     = nonans.wrap(model, optimizer)   # the only change
    >>>
    >>> for x, y in loader:
    ...     optimizer.zero_grad()
    ...     loss = criterion(model(x), y)
    ...     loss.backward()
    ...     optimizer.step()   # step_watch called automatically
    """
    sentinel = Sentinel(
        model           = model,
        optimizer       = optimizer,
        window          = window,
        k               = k,
        proxy_threshold = proxy_threshold,
        scan_every      = scan_every,
        min_numel       = min_numel,
    )

    Registry.register(sentinel)

    hook: Optional[_OptimizerHook] = None
    if auto_hook:
        hook = _OptimizerHook(optimizer, sentinel)
        hook.install()

    wrapped = WrappedModel(model, sentinel)
    wrapped.__dict__["_optimizer_hook"] = hook    # prevent GC
    return wrapped
