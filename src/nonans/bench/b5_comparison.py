"""
Benchmark 5: Comparison vs state-of-the-art instability detectors
==================================================================
We compare the entropy-based signal_score detector against the three
current state-of-the-art instability detectors used in production
training infrastructure:

  (a) Gradient-norm monitor  — alarms when ||g||_2 > 5× running median
  (b) Loss-spike monitor     — alarms when loss > 3× running median
  (c) Norm-deviation monitor — alarms when |‖W‖ − mean| > 0.5·mean
  (d) signal_score (ours)    — alarms when signal_score < 0.40

For each, we measure on the same controlled-instability runs:
  * detection rate          : fraction of diverged runs that fired before NaN
  * mean lead time          : T_NaN − T_alarm averaged over fired runs
  * false-alarm rate        : fraction of converged runs that fired

The fair-comparison principle: same set of runs, same NaN times,
different alarms.

Outputs
-------
JSON   → figures/bench5_comparison.json
Table  → figures/bench5_comparison.png
"""
from __future__ import annotations

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nonans.primitives import (
    randomized_svd_topk, shape_score, signal_score,
    attention_entropy_normalized,
)

from nonans.bench._harness import output_dir as _output_dir
OUT_DIR = _output_dir()


def run_one(seed, d=32, seq_len=16, max_steps=200,
            rank_drain_start=10, rank_drain_rate=0.08,
            amp_start=80, amp_rate=0.03, induce_nan=True):
    """
    induce_nan=True  → controlled instability (will diverge)
    induce_nan=False → benign perturbations (should converge / stay finite)
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s    = np.linspace(1.0, 0.7, d).astype(np.float32)
    W    = ((U * s) @ V.T).astype(np.float32)

    X = rng.standard_normal((seq_len, d)).astype(np.float32)
    td = rng.standard_normal((d,)).astype(np.float32); td /= np.linalg.norm(td)

    norms, sig_h, ah, grad_norms, losses = [], [], [], [], []

    T_nan = None
    for t in range(max_steps):
        if induce_nan:
            if t >= rank_drain_start:
                W = (W + rank_drain_rate * (np.outer(td, td) @ W)).astype(np.float32)
            if t >= amp_start:
                W = (W * (1.0 + amp_rate * (t - amp_start))).astype(np.float32)
        else:
            # Benign random walk
            W = (W + 0.005 * rng.standard_normal(W.shape).astype(np.float32))

        n = float(np.linalg.norm(W))
        norms.append(n)
        if not np.isfinite(n):
            T_nan = t
            break

        # "gradient" surrogate = drain + amp contribution magnitude
        if induce_nan and t >= rank_drain_start:
            g = rank_drain_rate * np.linalg.norm(np.outer(td, td) @ W)
            if t >= amp_start:
                g += amp_rate * (t - amp_start) * n
        else:
            g = 0.005 * np.sqrt(W.size)
        grad_norms.append(float(g))

        # "loss" surrogate ∝ excess norm (proxy for divergence)
        loss = float(np.log1p(max(0.0, n - 1.0)))
        losses.append(loss)

        sv = randomized_svd_topk(W, k=8)
        sig_h.append(signal_score(sv))

        scores = (X @ W) / np.sqrt(d)
        scores -= scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        A = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
        ah.append(float(np.mean([attention_entropy_normalized(A[i]) for i in range(seq_len)])))

    return dict(seed=seed, T_nan=T_nan, induce_nan=induce_nan,
                norms=norms, signal_score_hist=sig_h, attn_entropy_hist=ah,
                grad_norm_hist=grad_norms, loss_hist=losses)


# ─── Detectors ────────────────────────────────────────────────────────────────

def detect_grad_norm(run, factor=5.0, window=10):
    g = run["grad_norm_hist"]
    for t in range(window, len(g)):
        med = float(np.median(g[t-window:t]))
        if med > 1e-8 and g[t] > factor * med:
            return t
    return None

def detect_loss_spike(run, factor=3.0, window=10):
    L = run["loss_hist"]
    for t in range(window, len(L)):
        med = float(np.median(L[t-window:t]))
        if med > 1e-8 and L[t] > factor * med:
            return t
    return None

def detect_norm_dev(run, factor=0.5, window=15):
    N = run["norms"]
    for t in range(window, len(N)):
        base = float(np.mean(N[t-window:t-3])) if t-window > 0 else float(np.mean(N[:t-3]))
        if base > 1e-8 and abs(N[t] - base) / base > factor:
            return t
    return None

def detect_signal_score(run, threshold=0.40):
    for t, s in enumerate(run["signal_score_hist"]):
        if s < threshold:
            return t
    return None

def detect_attn_entropy(run, low=0.20, high=0.95):
    for t, h in enumerate(run["attn_entropy_hist"]):
        if h < low or h > high:
            return t
    return None


DETECTORS = {
    "Gradient-norm spike (5×med)":  detect_grad_norm,
    "Loss spike (3×med)":            detect_loss_spike,
    "Weight-norm deviation (>50%)":  detect_norm_dev,
    "signal_score < 0.40 (ours)":    detect_signal_score,
    "attn entropy ∉ [0.2, 0.95] (ours)": detect_attn_entropy,
}


# ─── Sweep ────────────────────────────────────────────────────────────────────

UNSTABLE_SEEDS = list(range(20))
BENIGN_SEEDS   = list(range(100, 120))   # disjoint seeds

unstable_runs = [run_one(s, induce_nan=True)  for s in UNSTABLE_SEEDS]
benign_runs   = [run_one(s, induce_nan=False) for s in BENIGN_SEEDS]

table = {}
for name, fn in DETECTORS.items():
    # detection rate on unstable
    lead_times = []
    n_detected = 0
    for r in unstable_runs:
        T_alarm = fn(r)
        if T_alarm is not None and r["T_nan"] is not None and T_alarm < r["T_nan"]:
            n_detected += 1
            lead_times.append(r["T_nan"] - T_alarm)
    # false alarm on benign
    n_false = sum(1 for r in benign_runs if fn(r) is not None)

    table[name] = {
        "detection_rate":   n_detected / float(len(unstable_runs)),
        "mean_lead_steps":  (float(np.mean(lead_times)) if lead_times else None),
        "median_lead_steps":(float(np.median(lead_times)) if lead_times else None),
        "false_alarm_rate": n_false / float(len(benign_runs)),
        "n_unstable":       len(unstable_runs),
        "n_benign":         len(benign_runs),
    }

with open(os.path.join(OUT_DIR, "bench5_comparison.json"), "w") as f:
    json.dump(table, f, indent=2)


# ─── Visualization ────────────────────────────────────────────────────────────

names = list(table.keys())
det = [table[n]["detection_rate"]   for n in names]
lead = [table[n]["mean_lead_steps"] or 0 for n in names]
fa  = [table[n]["false_alarm_rate"] for n in names]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

colors = ["#9aa0a6", "#9aa0a6", "#9aa0a6", "#3a6fa0", "#9a3a3a"]

ax = axes[0]
ax.barh(range(len(names)), det, color=colors, edgecolor="white")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.set_xlim(0, 1.05); ax.set_xlabel("detection rate")
ax.set_title("Detection rate on diverged runs (n=20)"); ax.grid(True, alpha=0.3, axis="x")

ax = axes[1]
ax.barh(range(len(names)), lead, color=colors, edgecolor="white")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("mean lead time [steps before NaN]")
ax.set_title("Lead time (higher = better)"); ax.grid(True, alpha=0.3, axis="x")

ax = axes[2]
ax.barh(range(len(names)), fa, color=colors, edgecolor="white")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.set_xlim(0, 1.05); ax.set_xlabel("false-alarm rate (benign runs)")
ax.set_title("False-alarm rate (lower = better)"); ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bench5_comparison.png"), dpi=160)
plt.close()

print("─" * 80)
print("Benchmark 5 — Comparison vs SOTA detectors")
print("─" * 80)
print(f"  {'detector':<40} {'det':>5} {'lead':>7} {'FA':>6}")
print(f"  {'─' * 70}")
for n in names:
    t = table[n]
    lead_str = f"{t['mean_lead_steps']:.1f}" if t["mean_lead_steps"] is not None else "—"
    print(f"  {n:<40} {t['detection_rate']:>5.2f} {lead_str:>7} {t['false_alarm_rate']:>6.2f}")
print(f"\n  ✓ Wrote {os.path.join(OUT_DIR, 'bench5_comparison.json')}")
print(f"  ✓ Wrote {os.path.join(OUT_DIR, 'bench5_comparison.png')}")
