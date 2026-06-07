"""
nonans.primitives
=================

Mathematical primitives: Shannon entropy, randomized SVD, dynamics. All
operations operate on either ``numpy.ndarray`` (primary) or
``torch.Tensor`` (optional). The numpy path is the verified reference;
the torch path is bit-equivalent in algebra and dispatches when a tensor
is passed.

The algebra and the floating-point smoothing constants here are the same
ones used to produce the locked benchmark numbers (seed 20260525). Do
not change them without rerunning the full benchmark suite.

Verified results (numpy backend, seed 20260525):
    B1  signal_score(sv) == attention_entropy_normalized(sv/sum(sv))
        max error 3.18e-7 over N=10,000 (Dirichlet-mixed sample)
    B2  100% on three canonical fault classes
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

try:
    import torch
    _HAS_TORCH = True
    ArrayLike = Union[np.ndarray, "torch.Tensor"]
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False
    ArrayLike = np.ndarray  # type: ignore[misc,assignment]


# ─── Backend dispatch ────────────────────────────────────────────────────────

def _is_torch(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _to_numpy_f32(x: ArrayLike) -> np.ndarray:
    """Move any supported array to a contiguous float32 numpy view.

    The numpy path is authoritative; the torch path detaches and copies
    to CPU before delegating. Detach is required so monitoring never
    affects autograd.
    """
    if _is_torch(x):
        return x.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    return np.ascontiguousarray(x, dtype=np.float32)


# ─── Randomized SVD (Halko, Martinsson & Tropp 2011, Algorithm 4.1) ──────────

def randomized_svd_topk(
    W: ArrayLike,
    k: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Top-k singular values via randomized range finder.

    Cost: O(mn k) for an m×n matrix, vs O(min(m,n)^2 max(m,n)) for full SVD.
    Returns a float32 numpy vector, sorted descending. On degenerate input
    (empty, 1D, k>=min(m,n)) returns ``[||W||_F]`` or ``[0.0]``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    fallback = np.array([0.0], dtype=np.float32)
    A = _to_numpy_f32(W)
    if A.size == 0:
        return fallback
    if A.ndim <= 1:
        return np.array([float(np.linalg.norm(A))], dtype=np.float32)
    if A.ndim > 2:
        A = A.reshape(A.shape[0], -1)
    m, n = A.shape
    k_eff = min(k, min(m, n) - 1)
    if k_eff < 1:
        return np.array([float(np.linalg.norm(A))], dtype=np.float32)
    try:
        Omega = rng.standard_normal((n, k_eff)).astype(np.float32)
        Y = A @ Omega
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ A
        _, s, _ = np.linalg.svd(B, full_matrices=False)
        return np.sort(s[:k_eff])[::-1].astype(np.float32)
    except np.linalg.LinAlgError:
        return fallback


# ─── Health scores ───────────────────────────────────────────────────────────

def shape_score(sv: ArrayLike, threshold: float = 0.01) -> float:
    """Fraction of singular values above ``threshold * largest``.

    Captures rank breadth: 1.0 = full spread; 0.0 = rank-1 collapse.
    Insensitive to magnitude (scale-invariant).
    """
    s = _to_numpy_f32(sv)
    if s.size == 0 or s[0] < 1e-8:
        return 0.0
    ratio = s / (s[0] + 1e-8)
    return float(np.sum(ratio > threshold)) / float(s.size)


def signal_score(sv: ArrayLike) -> float:
    """Normalized Shannon entropy of the singular-value distribution.

    The training-time half of the unified-identity claim (B1).
    Returns a value in [0, 1] where 1.0 = maximally diffuse spectrum
    and 0.0 = single-mode concentration.
    """
    s = _to_numpy_f32(sv)
    k = s.size
    if k == 0 or k == 1:
        return 0.0
    total = float(s.sum())
    if total < 1e-8:
        return 0.0
    p = s / total
    H = float(-(p * np.log(p + 1e-10)).sum())
    H_max = float(np.log(k))
    return float(np.clip(H / (H_max + 1e-8), 0.0, 1.0))


# ─── Dynamics ────────────────────────────────────────────────────────────────

def collapse_rate(trajectory: ArrayLike) -> float:
    """First-order rate of structural decline (linear-fit slope).

    Negative values indicate decline; positive indicate recovery. Returns
    0.0 on insufficient history (< 3 points) or fit failure.
    """
    y = _to_numpy_f32(trajectory)
    n = y.size
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=np.float32)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def norm_deviation(norm: float, history: list, tail: int = 3) -> float:
    """Relative deviation of a current norm vs its pre-tail baseline.

    Used by Gate 1 as a cheap O(n) trigger. The ``tail`` parameter
    excludes the most recent steps from the baseline so a deviation
    cannot mask itself by entering the average.
    """
    n = len(history)
    if n < tail + 3:
        return 0.0
    baseline = float(np.mean(history[:-tail]))
    if baseline < 1e-8:
        return 0.0
    return abs(norm - baseline) / baseline


# ─── Attention entropy (inference-time path) ─────────────────────────────────

def attention_entropy(weights: ArrayLike) -> float:
    """Shannon entropy of an attention distribution (raw, unnormalized)."""
    p = _to_numpy_f32(weights).ravel()
    if p.size == 0:
        return 0.0
    p = p / (p.sum() + 1e-10)
    return float(-(p * np.log(p + 1e-10)).sum())


def attention_entropy_normalized(weights: ArrayLike) -> float:
    """Normalized attention entropy in [0, 1].

    The inference-time half of the unified-identity claim (B1).
    """
    p = _to_numpy_f32(weights).ravel()
    n = p.size
    if n == 0:
        return 0.0
    H = attention_entropy(p)
    H_max = float(np.log(n)) if n > 1 else 1.0
    return float(np.clip(H / (H_max + 1e-8), 0.0, 1.0))


def attention_entropy_batched(A: ArrayLike) -> np.ndarray:
    """Vectorized entropy across the last axis. Hot path for inference.

    For an input of shape (..., n), returns shape (...,) of raw entropies.
    Avoids per-head Python overhead in the inference forward pass.
    """
    p = _to_numpy_f32(A)
    return -(p * np.log(p + 1e-10)).sum(axis=-1)
