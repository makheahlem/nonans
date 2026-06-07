"""
Benchmark 6: Calibration reduces false-alarm rate
=================================================
Benchmark 5 revealed that uncalibrated attention-entropy detection
has 100% false-alarm rate on benign runs because the absolute band
[0.2, 0.95] does not match the nominal entropy of well-formed
attention distributions.

This benchmark shows that a short calibration pass on in-distribution
data — collecting per-head (mean, std, p5, p95) — recovers
discriminability between in-distribution attention and structurally
degenerate attention.

Protocol
--------
  1. Sample N_cal = 500 in-distribution attention vectors
     (moderately concentrated, T = 1.5).
  2. Fit nominal (mean, std, p5, p95).
  3. Test on:
     - N_id = 200 in-distribution → expected: HEALTHY
     - N_ood-collapse = 200 one-hot-ish → expected: COLLAPSE
     - N_ood-max     = 200 near-uniform → expected: MAXIMUM
  4. Report classification accuracy with and without calibration.

Output
------
JSON  → figures/bench6_calibration.json
Plot  → figures/bench6_calibration.png
"""
from __future__ import annotations

import json, os, sys
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
SEQ_LEN = 64


def concentrated(n, temp=1.5):
    x = rng.standard_normal((n, SEQ_LEN)) * temp
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x); return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

def diffuse(n, temp=0.05):
    x = rng.standard_normal((n, SEQ_LEN)) * temp
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x); return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

def onehot(n):
    out = np.zeros((n, SEQ_LEN), dtype=np.float32)
    pos = rng.integers(0, SEQ_LEN, size=n)
    for i, p in enumerate(pos):
        out[i, p] = 0.98
        rem = np.abs(rng.standard_normal(SEQ_LEN - 1))
        idx = [j for j in range(SEQ_LEN) if j != p]
        out[i, idx] = (rem / rem.sum() * 0.02).astype(np.float32)
    return out


# ─── 1. Calibration ────
cal = concentrated(500, temp=1.5)
cal_H = np.array([attention_entropy(s) for s in cal])
calib = {
    "mean": float(cal_H.mean()),
    "std":  float(cal_H.std() + 1e-8),
    "p5":   float(np.percentile(cal_H, 5)),
    "p95":  float(np.percentile(cal_H, 95)),
}

# ─── 2. Test sets ────
id_test    = concentrated(200, temp=1.5)
ood_coll   = onehot(200)
ood_max    = diffuse(200, temp=0.05)


def evaluate(data, expected, calib_used):
    correct = 0
    detail = {"HEALTHY": 0, "COLLAPSE": 0, "MAXIMUM": 0, "DEVIATION": 0, "UNKNOWN": 0}
    for s in data:
        H = attention_entropy(s)
        state, _ = classify_attention(
            H, n=SEQ_LEN, collapse_thr_abs=0.2,
            max_thr_abs=float(np.log(SEQ_LEN)) * 0.95,
            calib=calib_used, deviation_sigma=3.0,
        )
        detail[state.name] += 1
        if state == expected:
            correct += 1
    return correct / float(len(data)), detail


sets = [("in-distribution", id_test, RuntimeHealth.HEALTHY),
        ("ood-collapse",    ood_coll, RuntimeHealth.COLLAPSE),
        ("ood-max",         ood_max,  RuntimeHealth.MAXIMUM)]

result = {"calibration": calib, "results": {}}
for mode, calib_used in [("uncalibrated", None), ("calibrated", calib)]:
    result["results"][mode] = {}
    for label, data, exp in sets:
        acc, det = evaluate(data, exp, calib_used)
        result["results"][mode][label] = {"accuracy": acc, "detail": det}

with open(os.path.join(OUT_DIR, "bench6_calibration.json"), "w") as f:
    json.dump(result, f, indent=2)

# ─── 3. Plot ────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: entropy distribution per class with thresholds
ax = axes[0]
ax.hist([attention_entropy(s) for s in cal],         bins=30, alpha=0.6, label="calibration", color="#3a6fa0")
ax.hist([attention_entropy(s) for s in ood_coll],    bins=30, alpha=0.6, label="collapse OOD", color="#9a3a3a")
ax.hist([attention_entropy(s) for s in ood_max],     bins=30, alpha=0.6, label="maximum OOD", color="#9a9a3a")
ax.axvline(0.2,  color="black", linestyle=":", label="abs collapse 0.2")
ax.axvline(float(np.log(SEQ_LEN)) * 0.95, color="black", linestyle="--", label="abs max 0.95 log(n)")
ax.axvline(calib["p5"],  color="red", linestyle=":", label="calib p5")
ax.axvline(calib["p95"], color="red", linestyle="--", label="calib p95")
ax.set_xlabel("attention entropy H")
ax.set_ylabel("count")
ax.set_title("Entropy distributions and thresholds")
ax.legend(fontsize=7, loc="upper left")
ax.grid(True, alpha=0.3)

# Right: accuracy bars
ax = axes[1]
labels = [s[0] for s in sets]
unc = [result["results"]["uncalibrated"][l]["accuracy"] for l in labels]
cal_acc = [result["results"]["calibrated"][l]["accuracy"]  for l in labels]
x = np.arange(len(labels)); w = 0.4
ax.bar(x - w/2, unc,     w, label="uncalibrated", color="#9aa0a6", edgecolor="white")
ax.bar(x + w/2, cal_acc, w, label="calibrated",   color="#3a6fa0", edgecolor="white")
for i, v in enumerate(unc):    ax.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
for i, v in enumerate(cal_acc): ax.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.15); ax.set_ylabel("accuracy")
ax.set_title("Detection accuracy w/ vs w/o calibration")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bench6_calibration.png"), dpi=160)
plt.close()

print("─" * 64)
print("Benchmark 6 — Calibration effect")
print("─" * 64)
print(f"  Calibration sample size : 500")
print(f"  Calibration nominal     : H ~ N({calib['mean']:.3f}, {calib['std']:.3f})")
print(f"  p5/p95 band             : [{calib['p5']:.3f}, {calib['p95']:.3f}]")
print()
print(f"  {'set':<18} {'uncal acc':>10} {'cal acc':>10}")
for label, *_ in sets:
    u = result["results"]["uncalibrated"][label]["accuracy"]
    c = result["results"]["calibrated"][label]["accuracy"]
    print(f"  {label:<18} {u:>10.3f} {c:>10.3f}")
print(f"\n  ✓ Wrote {os.path.join(OUT_DIR, 'bench6_calibration.json')}")
print(f"  ✓ Wrote {os.path.join(OUT_DIR, 'bench6_calibration.png')}")
