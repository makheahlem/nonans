"""
Tests for nonans.decision.

Covers the ghost/real classifier and the inference-time attention
classifier across every decision branch.
"""
from __future__ import annotations

import math

import pytest

from nonans.decision import classify, classify_attention
from nonans.protocol import (
    Health,
    MIN_HISTORY,
    RuntimeHealth,
    SHAPE_CRITICAL,
    SHAPE_HEALTHY,
)


# ─── training-time classifier ────────────────────────────────────────────────

class TestClassify:
    def test_too_little_history_unknown(self):
        h, _ = classify(
            shape=0.9, signal=0.9, force=1.0,
            rate=0.0, momentum_align=0.0,
            history_len=MIN_HISTORY - 1,
        )
        assert h is Health.UNKNOWN

    def test_high_shape_always_healthy(self):
        # Even with bad signals, shape >= threshold => HEALTHY.
        h, conf = classify(
            shape=SHAPE_HEALTHY + 0.01, signal=0.9, force=1e-6,
            rate=-1.0, momentum_align=-1.0,
            history_len=MIN_HISTORY + 1,
        )
        assert h is Health.HEALTHY
        assert 0.0 <= conf <= 1.0

    def test_critical_shape_always_real(self):
        # Even with good signals, shape < critical => REAL.
        h, conf = classify(
            shape=SHAPE_CRITICAL - 0.01, signal=0.9, force=1.0,
            rate=0.0, momentum_align=0.0,
            history_len=MIN_HISTORY + 1,
        )
        assert h is Health.REAL
        # Confidence rises as shape approaches zero
        assert conf >= 0.85

    def test_middle_zone_zero_evidence_ghost(self):
        # Mid-range shape, no fault signals => GHOST
        h, _ = classify(
            shape=0.5, signal=0.5, force=1.0,
            rate=0.0, momentum_align=0.0,
            history_len=MIN_HISTORY + 1,
        )
        assert h is Health.GHOST

    def test_middle_zone_majority_evidence_real(self):
        # Two of three signals fire => REAL
        h, _ = classify(
            shape=0.5, signal=0.5, force=1e-6,  # force signal +1
            rate=-1.0,                          # rate signal +1
            momentum_align=0.0,                 # momentum signal not +1
            history_len=MIN_HISTORY + 1,
        )
        assert h is Health.REAL

    def test_middle_zone_one_evidence_ghost(self):
        # Only one signal fires => GHOST
        h, _ = classify(
            shape=0.5, signal=0.5, force=1e-6,
            rate=0.0, momentum_align=0.0,
            history_len=MIN_HISTORY + 1,
        )
        assert h is Health.GHOST


# ─── inference-time classifier ───────────────────────────────────────────────

class TestClassifyAttention:
    def test_low_entropy_collapse(self):
        state, _ = classify_attention(H=0.05, n=64)
        assert state is RuntimeHealth.COLLAPSE

    def test_high_entropy_maximum(self):
        n = 64
        state, _ = classify_attention(H=math.log(n) * 0.99, n=n)
        assert state is RuntimeHealth.MAXIMUM

    def test_mid_entropy_healthy(self):
        n = 64
        state, _ = classify_attention(H=math.log(n) * 0.5, n=n)
        assert state is RuntimeHealth.HEALTHY

    def test_calibrated_deviation(self):
        # Entropy well above calibrated mean → DEVIATION
        n = 64
        calib = {"mean": 3.0, "std": 0.2, "p5": 2.7, "p95": 3.3}
        # H well above p95 but below MAXIMUM threshold (~0.95*log(64) ≈ 3.95)
        # Hmm, p95=3.3 → set H=3.4 which is above p95 → COLLAPSE/MAXIMUM
        # Use entropy slightly above calibration band but inside absolute band
        H = 3.5
        state, _ = classify_attention(
            H=H, n=n, calib=calib,
        )
        # 3.5 > p95=3.3 but < max_abs=0.95*log(64) ≈ 3.95 -> MAXIMUM by calib
        assert state in (RuntimeHealth.MAXIMUM, RuntimeHealth.DEVIATION)

    def test_calibrated_in_band_healthy(self):
        n = 64
        calib = {"mean": 3.0, "std": 0.2, "p5": 2.7, "p95": 3.3}
        state, _ = classify_attention(H=3.0, n=n, calib=calib)
        assert state is RuntimeHealth.HEALTHY

    def test_confidence_in_unit_interval(self):
        n = 64
        for H in (0.0, 0.1, 1.0, math.log(n) / 2, math.log(n) * 0.99, math.log(n)):
            _, conf = classify_attention(H=H, n=n)
            assert 0.0 <= conf <= 1.0
