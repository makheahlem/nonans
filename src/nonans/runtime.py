"""
nonans.runtime
==============
Inference-time structural health monitor for transformer attention.

The training-time analog (engine.py) applies H = -Σ p log p to the
singular-value distribution of weight tensors. This module applies the
same formula to the attention-weight distribution at each forward pass.

Same principle. Same formula. Same near-zero overhead on already-resident
data. Different distribution. Different timing — signal available BEFORE
the output token is produced.

Three fault conditions
----------------------
  COLLAPSE  H → 0          single-token fixation (degenerate attention)
  MAXIMUM   H → log(n)     uniform attention, no structured extraction
  DEVIATION |H - H̄| > κσ   outside calibrated nominal range

The first two are absolute and require no calibration. The third
requires a short calibration pass over in-distribution data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .primitives import attention_entropy, attention_entropy_batched
from .protocol import RuntimeHealth


# ─── Per-head calibration ─────────────────────────────────────────────────────

@dataclass
class HeadCalibration:
    """
    Nominal entropy distribution for one (layer, head) pair.
    Populated during a calibration pass over representative in-distribution data.
    """
    head:      int
    layer:     int
    mean:      float
    std:       float
    p5:        float
    p95:       float
    n_samples: int = 0

    @property
    def collapse_threshold(self) -> float:
        return self.p5

    @property
    def maximum_threshold(self) -> float:
        return self.p95

    @property
    def deviation_threshold(self) -> float:
        return 3.0 * self.std


# ─── Runtime monitor ──────────────────────────────────────────────────────────

class RuntimeMonitor:
    """
    Inference-time structural health monitor for transformer attention.

    Mode 1 (uncalibrated) — detects absolute COLLAPSE and MAXIMUM.
    Mode 2 (calibrated)   — also detects DEVIATION from in-distribution.

    Parameters
    ----------
    collapse_threshold_abs : absolute H threshold for COLLAPSE fault
    sequence_length        : expected sequence length (for max-entropy bound)
    deviation_sigma        : σ multiplier for calibrated DEVIATION detection
    """

    def __init__(
        self,
        collapse_threshold_abs: float = 0.1,
        sequence_length:        int   = 512,
        deviation_sigma:        float = 3.0,
    ) -> None:
        self.collapse_threshold_abs = float(collapse_threshold_abs)
        self.sequence_length        = int(sequence_length)
        self.deviation_sigma        = float(deviation_sigma)
        self.maximum_threshold_abs  = float(torch.log(torch.tensor(float(sequence_length)))) * 0.95

        self._calibration: Dict[Tuple[int, int], HeadCalibration]  = {}
        self._cal_buffer:  Dict[Tuple[int, int], List[float]]      = {}

    # ── Entropy primitives (1-D / batched) ────────────────────────────────────

    @staticmethod
    def entropy(weights: torch.Tensor) -> float:
        return attention_entropy(weights)

    @staticmethod
    def entropy_normalized(weights: torch.Tensor) -> float:
        """Normalized entropy ∈ [0, 1]. 0 = collapsed, 1 = uniform."""
        n = weights.numel()
        if n == 0:
            return 0.0
        H = attention_entropy(weights)
        H_max = float(torch.log(torch.tensor(float(n)))) if n > 1 else 1.0
        return float(max(0.0, min(1.0, H / (H_max + 1e-8))))

    # ── Calibration ───────────────────────────────────────────────────────────

    def update_calibration(self, weights: torch.Tensor, layer: int, head: int) -> None:
        """Accumulate one calibration observation for (layer, head)."""
        key = (layer, head)
        if key not in self._cal_buffer:
            self._cal_buffer[key] = []
        self._cal_buffer[key].append(self.entropy(weights))

    def finalize_calibration(self) -> Dict[Tuple[int, int], HeadCalibration]:
        """Compute nominal distributions from accumulated observations."""
        for (layer, head), samples in self._cal_buffer.items():
            t = torch.tensor(samples, dtype=torch.float64)
            self._calibration[(layer, head)] = HeadCalibration(
                head      = head,
                layer     = layer,
                mean      = float(t.mean()),
                std       = float(t.std() + 1e-8),
                p5        = float(torch.quantile(t, 0.05)),
                p95       = float(torch.quantile(t, 0.95)),
                n_samples = int(t.numel()),
            )
        self._cal_buffer.clear()
        return dict(self._calibration)

    # ── Classification ────────────────────────────────────────────────────────

    def classify_head(
        self,
        weights: torch.Tensor,
        layer:   int,
        head:    int,
    ) -> Tuple[RuntimeHealth, float, float]:
        """
        Classify one head's structural state.
        Returns (state, confidence ∈ [0, 1], entropy_value).
        """
        H = self.entropy(weights)
        n = weights.numel()
        H_max = float(torch.log(torch.tensor(float(n)))) if n > 1 else 1.0

        # Absolute collapse
        if H < self.collapse_threshold_abs:
            conf = 1.0 - (H / max(self.collapse_threshold_abs, 1e-10))
            return RuntimeHealth.COLLAPSE, float(max(0.0, min(1.0, conf))), H

        # Absolute maximum
        if H > self.maximum_threshold_abs:
            conf = (H - self.maximum_threshold_abs) / (H_max - self.maximum_threshold_abs + 1e-8)
            return RuntimeHealth.MAXIMUM, float(max(0.0, min(1.0, conf))), H

        # Calibrated deviation
        cal = self._calibration.get((layer, head))
        if cal is not None:
            if H < cal.collapse_threshold:
                conf = (cal.collapse_threshold - H) / (cal.collapse_threshold + 1e-8)
                return RuntimeHealth.COLLAPSE, float(max(0.0, min(1.0, conf))), H
            if H > cal.maximum_threshold:
                conf = (H - cal.maximum_threshold) / (H_max - cal.maximum_threshold + 1e-8)
                return RuntimeHealth.MAXIMUM, float(max(0.0, min(1.0, conf))), H
            z = abs(H - cal.mean) / cal.std
            if z > self.deviation_sigma:
                conf = min(1.0, (z - self.deviation_sigma) / self.deviation_sigma)
                return RuntimeHealth.DEVIATION, float(conf), H
        elif len(self._calibration) > 0:
            # Some heads calibrated, this one isn't — be honest
            return RuntimeHealth.UNKNOWN, 0.0, H

        return RuntimeHealth.HEALTHY, min(1.0, H / H_max), H

    # ── Batched check (forward-pass hot path) ─────────────────────────────────

    def check_layer(self, attention_per_head: torch.Tensor, layer: int
                    ) -> List[Tuple[int, RuntimeHealth, float, float]]:
        """
        Check all heads in one layer.

        Parameters
        ----------
        attention_per_head : tensor of shape (n_heads, seq_len), softmax outputs
        layer              : layer index

        Returns
        -------
        list of (head_idx, state, confidence, entropy)
        """
        if attention_per_head.ndim == 1:
            attention_per_head = attention_per_head.unsqueeze(0)

        # Single batched entropy call — the hot-path optimization
        H_per_head = attention_entropy_batched(attention_per_head)

        results: List[Tuple[int, RuntimeHealth, float, float]] = []
        for h_idx in range(attention_per_head.shape[0]):
            state, conf, H = self.classify_head(
                attention_per_head[h_idx], layer, h_idx
            )
            results.append((h_idx, state, conf, H))
        return results

    def display_layer(self, layer: int, results: list) -> None:
        """Print per-head health for one layer, severity-sorted."""
        print(f"\n  Layer {layer:>3}  {'HEAD':>4}  {'ENTROPY':>8}  {'STATE':<10}  {'CONF':>4}")
        print(f"  {'─' * 50}")
        for head_idx, state, conf, H in sorted(results, key=lambda x: x[1].severity):
            print(
                f"  {'':>9}  {head_idx:>4}  {H:>8.4f}  "
                f"{state.icon} {state.name:<8}  {conf:>4.2f}"
            )
