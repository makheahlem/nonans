"""
Benchmark 1: Unified-identity claim
====================================
Numerical verification that signal_score(σ) ≡ attention_entropy_normalized(p)
when applied to the normalized version of the same distribution.

This is the central mathematical claim of the unified framework: training-
time weight health and inference-time attention health are the *same*
Shannon-entropy functional applied to different already-computed
distributions.

Outputs
-------
JSON summary written to figures/bench1_unified_identity.json
Histogram of pointwise errors → figures/bench1_unified_identity.png
"""
from __future__ import annotations

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nonans.primitives import signal_score, attention_entropy_normalized

from nonans.bench._harness import output_dir as _output_dir
OUT_DIR = _output_dir()

rng = np.random.default_rng(20260525)
N = 10_000   # large enough that any systematic bias would be visible
errors = []
for _ in range(N):
    k = int(rng.integers(4, 32))
    sv = rng.exponential(1.0, size=k).astype(np.float32)
    a  = signal_score(sv)
    b  = attention_entropy_normalized((sv / sv.sum()).astype(np.float32))
    errors.append(abs(a - b))

errors = np.array(errors)
result = {
    "n_samples":    int(N),
    "max_error":    float(errors.max()),
    "mean_error":   float(errors.mean()),
    "median_error": float(np.median(errors)),
    "p99_error":    float(np.percentile(errors, 99)),
    "claim":        "signal_score(sv) ≡ attention_entropy_normalized(sv/Σsv)",
    "verdict":      "VERIFIED" if errors.max() < 1e-4 else "REJECTED",
}

with open(os.path.join(OUT_DIR, "bench1_unified_identity.json"), "w") as f:
    json.dump(result, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.hist(np.log10(errors + 1e-12), bins=40, color="#3a6fa0", edgecolor="white", linewidth=0.4)
ax.set_xlabel(r"$\log_{10}\,|s(\sigma) - \tilde H(\sigma/\Sigma\sigma)|$")
ax.set_ylabel("count")
ax.set_title(f"Unified identity: pointwise error over N={N:,} samples\n"
             f"max = {result['max_error']:.2e}, mean = {result['mean_error']:.2e}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bench1_unified_identity.png"), dpi=160)
plt.close()

print("─" * 64)
print("Benchmark 1 — Unified identity claim")
print("─" * 64)
for k, v in result.items():
    print(f"  {k:<14} {v}")
print(f"\n  ✓ Wrote {os.path.join(OUT_DIR, 'bench1_unified_identity.json')}")
print(f"  ✓ Wrote {os.path.join(OUT_DIR, 'bench1_unified_identity.png')}")
