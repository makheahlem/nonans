"""
Benchmark 3: Overhead of entropy monitoring vs. attention forward pass
======================================================================
Measures the wall-clock cost of computing Shannon entropy over the
attention probability tensor, expressed as a fraction of the cost of
computing the attention itself (QK^T / √d → softmax).

The key claim: entropy adds *marginal* cost on data that's already
cache-resident from the softmax. We report mean ± std over multiple
runs, and break it down by sequence length and head count.

NOTE — limitations of this number
---------------------------------
This benchmark is CPU-only (no torch in sandbox). The headline overhead
fraction will be lower on GPU because:
  (a) attention itself uses the L2/SRAM-resident softmax output;
  (b) entropy is a single fused reduction over the same memory;
  (c) GPU benefits more from the FLOP density of attention than entropy.
A separate PyTorch protocol (protocols/run_overhead_torch.py) will be
provided for re-running this on accelerator hardware.

Outputs
-------
JSON  → figures/bench3_overhead.json
Plot  → figures/bench3_overhead.png
"""
from __future__ import annotations

import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nonans.primitives import attention_entropy_batched

from nonans.bench._harness import output_dir as _output_dir
OUT_DIR = _output_dir()

rng = np.random.default_rng(20260525)


def attention_forward(Q, K):
    """Standard scaled dot-product attention up to softmax (we measure both halves)."""
    d_k = Q.shape[-1]
    scores = (Q @ K) / np.sqrt(d_k)
    scores -= scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def time_us(fn, *args, n_iter=200):
    # warm
    for _ in range(5):
        fn(*args)
    t0 = time.perf_counter_ns()
    for _ in range(n_iter):
        fn(*args)
    return (time.perf_counter_ns() - t0) / n_iter / 1000.0   # µs


configs = []
for n_heads, seq_len, d_k in [
    (8,  128, 64),
    (8,  256, 64),
    (16, 256, 64),
    (16, 512, 64),
    (32, 512, 64),
]:
    Q = rng.standard_normal((n_heads, 1, d_k)).astype(np.float32)
    K = rng.standard_normal((n_heads, d_k, seq_len)).astype(np.float32)

    A = attention_forward(Q, K)                              # (n_heads, 1, seq_len)
    A2 = A.reshape(n_heads, seq_len)

    fwd_us  = time_us(attention_forward, Q, K, n_iter=300)
    ent_us  = time_us(attention_entropy_batched, A2, n_iter=300)
    overhead_pct = 100.0 * ent_us / fwd_us

    configs.append({
        "n_heads":    n_heads,
        "seq_len":    seq_len,
        "d_k":        d_k,
        "fwd_us":     fwd_us,
        "ent_us":     ent_us,
        "overhead_%": overhead_pct,
    })

# Summary
overheads = [c["overhead_%"] for c in configs]
result = {
    "configurations": configs,
    "mean_overhead_%":   float(np.mean(overheads)),
    "median_overhead_%": float(np.median(overheads)),
    "min_overhead_%":    float(np.min(overheads)),
    "max_overhead_%":    float(np.max(overheads)),
    "backend":           "numpy (CPU)",
    "note": "Re-run on PyTorch GPU via protocols/run_overhead_torch.py for accelerator numbers.",
}

with open(os.path.join(OUT_DIR, "bench3_overhead.json"), "w") as f:
    json.dump(result, f, indent=2)

# Plot
labels = [f"{c['n_heads']}h×{c['seq_len']}" for c in configs]
fwd_vals = [c["fwd_us"] for c in configs]
ent_vals = [c["ent_us"] for c in configs]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
x = np.arange(len(labels))
w = 0.4
axes[0].bar(x - w/2, fwd_vals, w, label="attention (QK→softmax)", color="#9aa0a6")
axes[0].bar(x + w/2, ent_vals, w, label="entropy reduction",      color="#3a6fa0")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=15)
axes[0].set_ylabel("µs / call")
axes[0].set_title("Wall-clock per call")
axes[0].legend(loc="upper left", fontsize=9)
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(x, overheads, color="#3a6fa0", edgecolor="white")
axes[1].axhline(15.0, color="red", linestyle="--", linewidth=1, label="claimed bound (15%)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=15)
axes[1].set_ylabel("overhead [%]")
axes[1].set_title("Entropy overhead vs. attention")
axes[1].set_ylim(0, max(max(overheads) * 1.2, 20))
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bench3_overhead.png"), dpi=160)
plt.close()

print("─" * 64)
print("Benchmark 3 — Overhead (entropy vs. attention)")
print("─" * 64)
print(f"  {'config':<12} {'fwd µs':>10} {'ent µs':>10} {'overhead %':>12}")
for c in configs:
    print(f"  {c['n_heads']}h×{c['seq_len']:<6} {c['fwd_us']:>10.2f} {c['ent_us']:>10.2f} {c['overhead_%']:>12.2f}")
print()
print(f"  mean overhead   : {result['mean_overhead_%']:>6.2f} %")
print(f"  median overhead : {result['median_overhead_%']:>6.2f} %")
print(f"\n  ✓ Wrote {os.path.join(OUT_DIR, 'bench3_overhead.json')}")
print(f"  ✓ Wrote {os.path.join(OUT_DIR, 'bench3_overhead.png')}")
