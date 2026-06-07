"""
Benchmark 2: Detection accuracy on three injected attention-fault classes
=========================================================================
Injects controlled attention pathologies and measures classifier accuracy.

  COLLAPSE  : one-hot-ish attention (mass on single position)
  MAXIMUM   : near-uniform attention (no structured extraction)
  HEALTHY   : moderate concentration (typical of well-formed attention)

This is *not* claiming "100% on real models". It is claiming the
classifier correctly recovers the fault class on the canonical injection
patterns that define each class. Real-model evaluation is benchmark 5.

Outputs
-------
JSON summary  → figures/bench2_detection.json
Confusion matrix figure → figures/bench2_detection.png
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

from nonans.primitives import attention_entropy
from nonans.decision import classify_attention
from nonans.protocol import RuntimeHealth

from nonans.bench._harness import output_dir as _output_dir
OUT_DIR = _output_dir()

rng = np.random.default_rng(20260525)
N = 1000
SEQ_LEN = 64


def concentrated(n, seq_len, temp=1.5):
    x = rng.standard_normal((n, seq_len)) * temp
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def diffuse(n, seq_len, temp=0.05):
    x = rng.standard_normal((n, seq_len)) * temp
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def onehot(n, seq_len):
    """One-hot-ish: 0.98 on one position, 0.02 spread on the rest."""
    out = np.zeros((n, seq_len), dtype=np.float32)
    pos = rng.integers(0, seq_len, size=n)
    for i, p in enumerate(pos):
        out[i, p] = 0.98
        rem = np.abs(rng.standard_normal(seq_len - 1))
        idx = [j for j in range(seq_len) if j != p]
        out[i, idx] = (rem / rem.sum() * 0.02).astype(np.float32)
    return out


datasets = {
    "HEALTHY":  (concentrated(N, SEQ_LEN, 1.5), RuntimeHealth.HEALTHY),
    "COLLAPSE": (onehot(N, SEQ_LEN),            RuntimeHealth.COLLAPSE),
    "MAXIMUM":  (diffuse(N, SEQ_LEN, 0.05),     RuntimeHealth.MAXIMUM),
}

# Confusion matrix
classes_order = ["HEALTHY", "COLLAPSE", "MAXIMUM", "DEVIATION", "UNKNOWN"]
M = np.zeros((3, len(classes_order)), dtype=np.int32)

per_class_acc = {}
mean_H = {}
for i_row, (label, (data, expected)) in enumerate(datasets.items()):
    correct = 0
    Hs = []
    for sample in data:
        H = attention_entropy(sample)
        state, _ = classify_attention(H, n=SEQ_LEN,
                                      collapse_thr_abs=0.2,
                                      max_thr_abs=float(np.log(SEQ_LEN)) * 0.95)
        Hs.append(H)
        M[i_row, classes_order.index(state.name)] += 1
        if state == expected:
            correct += 1
    acc = correct / float(N)
    per_class_acc[label] = acc
    mean_H[label] = float(np.mean(Hs))

result = {
    "n_per_class":           int(N),
    "seq_len":               int(SEQ_LEN),
    "accuracy_per_class":    per_class_acc,
    "mean_entropy_per_class": mean_H,
    "collapse_threshold_abs": 0.2,
    "maximum_threshold_abs":  float(np.log(SEQ_LEN)) * 0.95,
    "confusion_matrix":       M.tolist(),
    "classes_order":          classes_order,
}

with open(os.path.join(OUT_DIR, "bench2_detection.json"), "w") as f:
    json.dump(result, f, indent=2)

fig, ax = plt.subplots(figsize=(5.6, 3.5))
im = ax.imshow(M / N, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(classes_order)))
ax.set_xticklabels(classes_order, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels(["HEALTHY (true)", "COLLAPSE (true)", "MAXIMUM (true)"], fontsize=9)
for i in range(3):
    for j in range(len(classes_order)):
        if M[i, j] > 0:
            ax.text(j, i, f"{M[i, j]}", ha="center", va="center",
                    color=("white" if M[i, j] > N // 2 else "black"), fontsize=9)
ax.set_title(f"Detection — confusion matrix (N={N}/class, seq_len={SEQ_LEN})")
fig.colorbar(im, ax=ax, label="proportion")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bench2_detection.png"), dpi=160)
plt.close()

print("─" * 64)
print("Benchmark 2 — Detection accuracy on injected faults")
print("─" * 64)
print(f"  N per class    : {N}, seq_len: {SEQ_LEN}")
print(f"  collapse_thr   : {result['collapse_threshold_abs']}")
print(f"  maximum_thr    : {result['maximum_threshold_abs']:.4f}")
print()
print(f"  {'class':<10} {'accuracy':>10} {'mean H':>10}")
for k in ["HEALTHY", "COLLAPSE", "MAXIMUM"]:
    print(f"  {k:<10} {per_class_acc[k]:>10.4f} {mean_H[k]:>10.4f}")
print(f"\n  ✓ Wrote {os.path.join(OUT_DIR, 'bench2_detection.json')}")
print(f"  ✓ Wrote {os.path.join(OUT_DIR, 'bench2_detection.png')}")
