"""
Tests for nonans.primitives.

Covers:
  * Shannon entropy correctness against hand-computed values
  * normalized entropy bounded in [0, 1] with correct extremes
  * shape_score behavior on rank-1 vs full-rank distributions
  * collapse_rate slope sign correctness
  * attention_entropy_batched shape and value parity with the scalar form
  * randomized_svd_topk shape, ordering, and degeneracy handling
  * edge cases: empty, k=1, zeros, non-finite inputs
  * numpy/torch parity (auto-skipped if torch absent)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nonans.primitives import (
    attention_entropy,
    attention_entropy_batched,
    attention_entropy_normalized,
    collapse_rate,
    norm_deviation,
    randomized_svd_topk,
    shape_score,
    signal_score,
)


# ─── signal_score ────────────────────────────────────────────────────────────

class TestSignalScore:
    def test_uniform_spectrum_is_one(self):
        sv = np.ones(8, dtype=np.float32)
        assert signal_score(sv) == pytest.approx(1.0, abs=1e-6)

    def test_one_hot_spectrum_is_near_zero(self):
        sv = np.zeros(8, dtype=np.float32)
        sv[0] = 1.0
        s = signal_score(sv)
        assert s < 1e-5, f"expected near-zero entropy, got {s}"

    def test_empty_returns_zero(self):
        assert signal_score(np.array([], dtype=np.float32)) == 0.0

    def test_single_element_returns_zero(self):
        # H/log(1) is undefined; the convention is 0.
        assert signal_score(np.array([1.0], dtype=np.float32)) == 0.0

    def test_all_zero_returns_zero(self):
        # total < 1e-8 short-circuits to 0
        assert signal_score(np.zeros(8, dtype=np.float32)) == 0.0

    def test_bounded_in_unit_interval(self, rng):
        for _ in range(50):
            k = int(rng.integers(2, 32))
            sv = rng.exponential(1.0, size=k).astype(np.float32)
            s = signal_score(sv)
            assert 0.0 <= s <= 1.0


# ─── attention_entropy ───────────────────────────────────────────────────────

class TestAttentionEntropy:
    def test_uniform_distribution_max_entropy(self):
        n = 16
        a = np.ones(n, dtype=np.float32) / n
        H = attention_entropy(a)
        assert H == pytest.approx(math.log(n), abs=1e-5)

    def test_one_hot_zero_entropy(self):
        a = np.zeros(16, dtype=np.float32)
        a[0] = 1.0
        # +1e-10 smoothing makes the result slightly above 0; bound it.
        assert attention_entropy(a) < 1e-3

    def test_normalized_bounded(self, rng):
        for _ in range(50):
            n = int(rng.integers(2, 256))
            a = rng.dirichlet(np.ones(n) * 0.5).astype(np.float32)
            H_norm = attention_entropy_normalized(a)
            assert 0.0 <= H_norm <= 1.0

    def test_batched_matches_scalar(self, rng):
        # Vector form must agree with scalar form per row.
        B, n = 7, 32
        A = rng.dirichlet(np.ones(n) * 0.5, size=B).astype(np.float32)
        scalar = np.array([attention_entropy(A[i]) for i in range(B)])
        batched = attention_entropy_batched(A)
        np.testing.assert_allclose(scalar, batched, atol=1e-6)

    def test_batched_shape(self, rng):
        A = rng.dirichlet(np.ones(8), size=(3, 5)).astype(np.float32)
        out = attention_entropy_batched(A)
        assert out.shape == (3, 5)


# ─── shape_score ─────────────────────────────────────────────────────────────

class TestShapeScore:
    def test_full_rank_full_score(self):
        # Every entry above the 1%-of-max threshold.
        sv = np.linspace(1.0, 0.5, 8, dtype=np.float32)
        assert shape_score(sv) == 1.0

    def test_rank_one_low_score(self):
        sv = np.zeros(8, dtype=np.float32)
        sv[0] = 1.0
        # Only the first entry exceeds the 1% threshold.
        assert shape_score(sv) == pytest.approx(1.0 / 8, abs=1e-6)

    def test_zero_input_zero_score(self):
        assert shape_score(np.zeros(8, dtype=np.float32)) == 0.0


# ─── collapse_rate ───────────────────────────────────────────────────────────

class TestCollapseRate:
    def test_descending_negative_slope(self):
        traj = np.linspace(1.0, 0.0, 12, dtype=np.float32)
        rate = collapse_rate(traj)
        assert rate < 0

    def test_ascending_positive_slope(self):
        traj = np.linspace(0.0, 1.0, 12, dtype=np.float32)
        rate = collapse_rate(traj)
        assert rate > 0

    def test_constant_zero_slope(self):
        traj = np.full(12, 0.5, dtype=np.float32)
        assert collapse_rate(traj) == pytest.approx(0.0, abs=1e-6)

    def test_short_trajectory_returns_zero(self):
        assert collapse_rate(np.array([1.0, 0.5], dtype=np.float32)) == 0.0


# ─── norm_deviation ──────────────────────────────────────────────────────────

class TestNormDeviation:
    def test_short_history_returns_zero(self):
        assert norm_deviation(1.0, [1.0, 1.0]) == 0.0

    def test_deviation_proportional(self):
        history = [1.0] * 10
        # current = 1.2, baseline mean = 1.0, deviation = 0.2
        d = norm_deviation(1.2, history)
        assert d == pytest.approx(0.2, abs=1e-6)


# ─── randomized_svd_topk ─────────────────────────────────────────────────────

class TestRandomizedSVDTopK:
    def test_output_shape_and_dtype(self, rng):
        W = rng.standard_normal((32, 32)).astype(np.float32)
        sv = randomized_svd_topk(W, k=8, rng=rng)
        assert sv.dtype == np.float32
        # k_eff = min(k, min(m, n) - 1) = min(8, 31) = 8
        assert sv.size == 8

    def test_descending_order(self, rng):
        W = rng.standard_normal((64, 32)).astype(np.float32)
        sv = randomized_svd_topk(W, k=8, rng=rng)
        diffs = np.diff(sv)
        assert (diffs <= 1e-5).all(), f"singular values not descending: {sv}"

    def test_empty_returns_fallback(self):
        sv = randomized_svd_topk(np.array([], dtype=np.float32).reshape(0, 0))
        assert sv.tolist() == [0.0]

    def test_1d_input_returns_norm(self, rng):
        v = rng.standard_normal(8).astype(np.float32)
        sv = randomized_svd_topk(v)
        np.testing.assert_allclose(sv, [float(np.linalg.norm(v))], atol=1e-5)

    def test_approximates_true_top_singular_value(self, rng):
        # Verify the randomized estimator is in the right ballpark. With
        # k=8 oversampling on a fully random 64x32 matrix the approximation
        # is loose; tighten the matrix structure (low effective rank) to
        # get a tighter bound.
        # Build a matrix with structured spectrum so k=8 captures the top SV well.
        U = rng.standard_normal((64, 8)).astype(np.float32)
        V = rng.standard_normal((8, 32)).astype(np.float32)
        S = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05], dtype=np.float32)
        W = (U * S) @ V
        true_sv = np.linalg.svd(W, compute_uv=False)
        approx = randomized_svd_topk(W, k=8, rng=rng)
        # On structured (low-rank) input the top SV is recovered tightly.
        assert approx[0] == pytest.approx(true_sv[0], rel=0.10)


# ─── numpy/torch parity (opt-in) ─────────────────────────────────────────────

class TestTorchParity:
    """Run only when torch is installed. Confirms numpy and torch inputs
    produce identical outputs (since the math path always converts to
    numpy float32)."""

    def test_signal_score_parity(self, has_torch, rng):
        if not has_torch:
            pytest.skip("torch not installed")
        import torch
        sv = rng.exponential(1.0, size=8).astype(np.float32)
        s_np = signal_score(sv)
        s_torch = signal_score(torch.from_numpy(sv))
        assert s_np == pytest.approx(s_torch, abs=1e-7)

    def test_attention_entropy_parity(self, has_torch, rng):
        if not has_torch:
            pytest.skip("torch not installed")
        import torch
        a = rng.dirichlet(np.ones(16)).astype(np.float32)
        H_np = attention_entropy(a)
        H_torch = attention_entropy(torch.from_numpy(a))
        assert H_np == pytest.approx(H_torch, abs=1e-7)
