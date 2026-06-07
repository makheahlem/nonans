"""
nonans.decision
===============

Ghost-vs-real classifier (training-time) and runtime attention
classifier (inference-time).

The ghost/real classifier consumes five structural signals produced by
:mod:`nonans.primitives` and returns a discrete :class:`Health` state.
Thresholds are sourced from :mod:`nonans.protocol` and are locked
against the B2/B4/B5 benchmark results.

The runtime classifier maps a normalized attention entropy to a
:class:`RuntimeHealth` state. Two modes:

  * uncalibrated  — absolute COLLAPSE/MAXIMUM thresholds, ~100% FA rate
                    on in-distribution data, but zero setup cost.
  * calibrated    — per-head nominal distribution (mean/std/p5/p95)
                    learned from a calibration pass; reduces FA rate to
                    0% on in-distribution input (verified, B6).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .protocol import (
    EVIDENCE_REAL,
    FORCE_THRESHOLD,
    Health,
    MIN_HISTORY,
    MOMENTUM_THRESHOLD,
    RATE_THRESHOLD,
    RuntimeHealth,
    SHAPE_CRITICAL,
    SHAPE_HEALTHY,
)


# ─── Training-time classifier ────────────────────────────────────────────────

def classify(
    shape: float,
    signal: float,
    force: float,
    rate: float,
    momentum_align: float,
    history_len: int,
) -> Tuple[Health, float]:
    """Three-signal majority classifier.

    Returns (Health, confidence). The decision logic:

      * UNKNOWN if trajectory history is too short
      * HEALTHY if shape ≥ SHAPE_HEALTHY (always)
      * REAL    if shape < SHAPE_CRITICAL (always)
      * Otherwise count fault signals:
            rate < RATE_THRESHOLD            → +1
            force < FORCE_THRESHOLD          → +1
            momentum_align < MOMENTUM_THRESH → +1
        REAL if evidence ≥ 2, else GHOST.
    """
    if history_len < MIN_HISTORY:
        return Health.UNKNOWN, 0.0
    if shape >= SHAPE_HEALTHY:
        return Health.HEALTHY, min(1.0, shape)
    if shape < SHAPE_CRITICAL:
        # Confidence rises as shape approaches zero
        confidence = 0.85 + 0.15 * (1.0 - shape / SHAPE_CRITICAL)
        return Health.REAL, min(confidence, 0.99)
    evidence = 0
    if rate < RATE_THRESHOLD:
        evidence += 1
    if force < FORCE_THRESHOLD:
        evidence += 1
    if momentum_align < MOMENTUM_THRESHOLD:
        evidence += 1
    confidence = evidence / 3.0
    if evidence >= EVIDENCE_REAL:
        return Health.REAL, confidence
    return Health.GHOST, 1.0 - confidence


# ─── Inference-time classifier ───────────────────────────────────────────────

def classify_attention(
    H: float,
    n: int,
    collapse_thr_abs: float = 0.1,
    max_thr_abs: Optional[float] = None,
    calib: Optional[Dict[str, float]] = None,
    deviation_sigma: float = 3.0,
) -> Tuple[RuntimeHealth, float]:
    """Classify an attention head's entropy into a RuntimeHealth state.

    Parameters
    ----------
    H : float
        Raw Shannon entropy of the attention distribution.
    n : int
        Number of attention positions (sequence length).
    collapse_thr_abs : float
        Absolute entropy below which the state is COLLAPSE without
        requiring calibration.
    max_thr_abs : float, optional
        Absolute entropy above which the state is MAXIMUM. Defaults to
        0.95 * log(n).
    calib : dict, optional
        Per-head nominal distribution with keys ``mean``, ``std``,
        ``p5``, ``p95``. If provided, enables DEVIATION detection.
    deviation_sigma : float
        Z-score threshold for DEVIATION (vs calibration).

    Returns
    -------
    (RuntimeHealth, confidence)
    """
    H_max = float(np.log(n)) if n > 1 else 1.0
    if max_thr_abs is None:
        max_thr_abs = H_max * 0.95

    if H < collapse_thr_abs:
        conf = 1.0 - (H / max(collapse_thr_abs, 1e-10))
        return RuntimeHealth.COLLAPSE, float(np.clip(conf, 0.0, 1.0))
    if H > max_thr_abs:
        conf = (H - max_thr_abs) / (H_max - max_thr_abs + 1e-8)
        return RuntimeHealth.MAXIMUM, float(np.clip(conf, 0.0, 1.0))

    if calib is not None:
        if H < calib["p5"]:
            conf = (calib["p5"] - H) / (calib["p5"] + 1e-8)
            return RuntimeHealth.COLLAPSE, float(np.clip(conf, 0.0, 1.0))
        if H > calib["p95"]:
            conf = (H - calib["p95"]) / (H_max - calib["p95"] + 1e-8)
            return RuntimeHealth.MAXIMUM, float(np.clip(conf, 0.0, 1.0))
        z = abs(H - calib["mean"]) / max(calib["std"], 1e-8)
        if z > deviation_sigma:
            return RuntimeHealth.DEVIATION, float(min(1.0, (z - deviation_sigma) / deviation_sigma))

    return RuntimeHealth.HEALTHY, min(1.0, H / H_max)
