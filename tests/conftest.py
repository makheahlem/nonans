"""
Shared pytest fixtures for the nonans test suite.

Establishes:
  * the locked benchmark seed (20260525)
  * deterministic numpy RNG fixture
  * torch availability detection with auto-skip
  * cpu/gpu device fixtures parameterized over what is actually available
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure the src layout is on sys.path even when running pytest from the
# repository root without `pip install -e .` first. Real users will have
# installed the package; this is for the in-repo CI path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


SEED = 20260525


try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded NumPy generator. Identical across the suite."""
    return np.random.default_rng(SEED)


@pytest.fixture
def has_torch() -> bool:
    return _HAS_TORCH


@pytest.fixture
def has_cuda() -> bool:
    if not _HAS_TORCH:
        return False
    import torch
    return torch.cuda.is_available()


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> str:
    """Parameterized device fixture; auto-skips cuda when unavailable.

    Tests that opt into device parametrization run twice when CUDA is
    available, once otherwise. Single-device tests should not use this.
    """
    if request.param == "cuda":
        if not _HAS_TORCH:
            pytest.skip("torch not installed")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    return request.param


# ─── Markers ─────────────────────────────────────────────────────────────────

def pytest_collection_modifyitems(config, items):
    """Honor @pytest.mark.gpu by skipping when CUDA is unavailable."""
    skip_gpu = pytest.mark.skip(reason="requires CUDA-capable PyTorch")
    for item in items:
        if "gpu" in item.keywords:
            if not _HAS_TORCH:
                item.add_marker(skip_gpu)
                continue
            import torch
            if not torch.cuda.is_available():
                item.add_marker(skip_gpu)
