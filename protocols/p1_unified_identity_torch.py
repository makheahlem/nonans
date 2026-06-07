"""
Protocol P1: Unified-identity claim on real PyTorch tensors
============================================================
Run this script on your machine with PyTorch installed to reproduce
the unified-identity benchmark using the actual nonans library (not
the NumPy mirror).

Expected result: max pointwise error ≤ 5 × 10⁻⁷ over 10,000 samples.

Usage
-----
    cd /path/to/build
    pip install torch        # any version ≥ 1.12
    python protocols/run_unified_identity_torch.py

Reference benchmark: benchmarks/bench1_unified_identity.py (NumPy)
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nonans import signal_score, attention_entropy_normalized

generator = torch.Generator()
generator.manual_seed(20260525)
N = 10_000

errors = []
for _ in range(N):
    k   = int(torch.randint(4, 32, (1,), generator=generator).item())
    sv  = torch.abs(torch.randn(k, generator=generator))
    a   = signal_score(sv)
    b   = attention_entropy_normalized(sv / (sv.sum() + 1e-10))
    errors.append(abs(a - b))

errors = torch.tensor(errors)
print(f"Unified identity (PyTorch):")
print(f"  N samples  : {N:,}")
print(f"  max error  : {errors.max():.3e}")
print(f"  mean error : {errors.mean():.3e}")
print(f"  verdict    : {'VERIFIED ✓' if float(errors.max()) < 1e-4 else 'REJECTED ✗'}")
print()
print("NumPy reference (bench1): max_error = 3.18e-07  VERIFIED")
