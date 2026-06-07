"""
Benchmark 4: Early-warning lead time (controlled-instability protocol)
======================================================================
Uses a CONTROLLED-instability protocol that reliably produces a NaN
trajectory so the precursor-vs-NaN gap is measurable. The realistic
training-run version is in protocols/run_early_warning_torch.py.

Protocol
--------
A weight matrix W ∈ R^{d×d} is updated by:

    W ← W − lr·g(t) + ε(t)

where g(t) is a low-rank perturbation that drains rank progressively
(simulating gradient-induced rank collapse — a documented failure
mode in transformers, e.g. Dong, Cordonnier, Loukas, "Attention is
not all you need", ICML 2021), and ε(t) is a multiplicative scale
that grows toward infinity in finite steps (simulating optimizer-
momentum amplification — the Adam-momentum-locks-on-noise pathway).

Metric
------
  lead_time = T_NaN − T_alarm
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
            amp_start=80, amp_rate=0.03):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s    = np.linspace(1.0, 0.7, d).astype(np.float32)
    W    = ((U * s) @ V.T).astype(np.float32)

    X = rng.standard_normal((seq_len, d)).astype(np.float32)
    td = rng.standard_normal((d,)).astype(np.float32)
    td /= np.linalg.norm(td)

    norms, ss_h, sig_h, ah = [], [], [], []
    T_alarm = None
    T_nan   = None

    for t in range(max_steps):
        if t >= rank_drain_start:
            outer = np.outer(td, td)
            W = (W + rank_drain_rate * (outer @ W)).astype(np.float32)
        if t >= amp_start:
            W = (W * (1.0 + amp_rate * (t - amp_start))).astype(np.float32)

        n = float(np.linalg.norm(W))
        norms.append(n)
        if not np.isfinite(n):
            T_nan = t
            break

        sv  = randomized_svd_topk(W, k=8)
        ss  = shape_score(sv)
        sig = signal_score(sv)
        ss_h.append(ss); sig_h.append(sig)

        scores = (X @ W) / np.sqrt(d)
        scores -= scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        A = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
        H = float(np.mean([attention_entropy_normalized(A[i]) for i in range(seq_len)]))
        ah.append(H)

        if T_alarm is None and sig < 0.40:
            T_alarm = t

    return dict(seed=seed, T_alarm=T_alarm, T_nan=T_nan,
                norms=norms, shape_score_hist=ss_h,
                signal_score_hist=sig_h, attn_entropy_hist=ah)


PROFILES = [("mild",     0.05, 0.04, 100),
            ("moderate", 0.08, 0.05, 80),
            ("severe",   0.12, 0.08, 60)]
SEEDS = list(range(10))

all_runs = []
for label, dr, ar, asn in PROFILES:
    for s in SEEDS:
        r = run_one(s, rank_drain_rate=dr, amp_rate=ar, amp_start=asn)
        r["profile"] = label
        all_runs.append(r)


def lead_time(r):
    if r["T_nan"] is None or r["T_alarm"] is None: return None
    return r["T_nan"] - r["T_alarm"]


diverged = [r for r in all_runs if r["T_nan"] is not None]
with_alarm = [r for r in diverged if r["T_alarm"] is not None]
lts = [lead_time(r) for r in with_alarm]

per_profile = {}
for label, *_ in PROFILES:
    rs = [r for r in all_runs if r["profile"] == label]
    pls = [lead_time(r) for r in rs if r["T_nan"] is not None and r["T_alarm"] is not None]
    per_profile[label] = {
        "n_runs": len(rs),
        "n_diverged": sum(1 for r in rs if r["T_nan"] is not None),
        "n_with_alarm": sum(1 for r in rs if r["T_alarm"] is not None and r["T_nan"] is not None),
        "mean_lead_steps":   float(np.mean(pls))   if pls else None,
        "median_lead_steps": float(np.median(pls)) if pls else None,
        "min_lead_steps":    int(np.min(pls))      if pls else None,
        "max_lead_steps":    int(np.max(pls))      if pls else None,
    }

result = {
    "protocol": "controlled-instability",
    "n_runs_total": len(all_runs),
    "n_diverged":   len(diverged),
    "n_with_alarm": len(with_alarm),
    "alarm_threshold": "signal_score < 0.40",
    "lead_time_overall": {
        "mean":   float(np.mean(lts)) if lts else None,
        "median": float(np.median(lts)) if lts else None,
        "min":    int(np.min(lts)) if lts else None,
        "max":    int(np.max(lts)) if lts else None,
        "p25":    float(np.percentile(lts, 25)) if lts else None,
        "p75":    float(np.percentile(lts, 75)) if lts else None,
    },
    "per_profile": per_profile,
}

with open(os.path.join(OUT_DIR, "bench4_early_warning.json"), "w") as f:
    json.dump(result, f, indent=2)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
rep = next(r for r in all_runs if r["profile"] == "severe" and r["T_nan"] is not None)

ax = axes[0, 0]
ax.plot(rep["norms"], color="#222")
if rep["T_alarm"] is not None:
    ax.axvline(rep["T_alarm"], color="orange", linestyle="--", label=f"alarm @ {rep['T_alarm']}")
if rep["T_nan"] is not None:
    ax.axvline(rep["T_nan"], color="red", linestyle="--", label=f"NaN @ {rep['T_nan']}")
ax.set_yscale("log")
ax.set_xlabel("step"); ax.set_ylabel("‖W‖₂  (log)")
ax.set_title(f"Weight norm — severe profile, seed {rep['seed']}")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(rep["signal_score_hist"], label="signal_score(σ)", color="#3a6fa0")
ax.plot(rep["shape_score_hist"],  label="shape_score(σ)",  color="#3a9a3a")
ax.axhline(0.40, color="orange", linestyle=":", label="alarm threshold")
if rep["T_alarm"] is not None: ax.axvline(rep["T_alarm"], color="orange", linestyle="--")
if rep["T_nan"]   is not None: ax.axvline(rep["T_nan"],   color="red",    linestyle="--")
ax.set_xlabel("step"); ax.set_ylabel("score ∈ [0,1]"); ax.set_ylim(0, 1.05)
ax.set_title("Health scores (same run)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
if lts:
    ax.hist(lts, bins=max(8, len(lts)//3), color="#3a6fa0", edgecolor="white")
    ax.axvline(float(np.mean(lts)),   color="red",   linestyle="--", label=f"mean = {np.mean(lts):.1f}")
    ax.axvline(float(np.median(lts)), color="green", linestyle="--", label=f"median = {np.median(lts):.1f}")
    ax.legend()
ax.set_xlabel("lead time [steps] = T_NaN − T_alarm")
ax.set_ylabel("count")
ax.set_title(f"Lead-time distribution (n={len(lts)})")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 1]
ax.plot(rep["signal_score_hist"], label="signal_score (W)",       color="#3a6fa0")
ax.plot(rep["attn_entropy_hist"], label="attn entropy norm (Wx)", color="#9a3a3a")
if rep["T_alarm"] is not None: ax.axvline(rep["T_alarm"], color="orange", linestyle="--", linewidth=0.8)
if rep["T_nan"]   is not None: ax.axvline(rep["T_nan"],   color="red",    linestyle="--", linewidth=0.8)
ax.set_xlabel("step"); ax.set_ylabel("score ∈ [0,1]"); ax.set_ylim(0, 1.05)
ax.set_title("Both signals fall together (unified identity)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bench4_early_warning.png"), dpi=160)
plt.close()

print("─" * 64)
print("Benchmark 4 — Early warning lead time (controlled-instability)")
print("─" * 64)
print(f"  total runs            : {len(all_runs)}")
print(f"  diverged (reached NaN): {len(diverged)}")
print(f"  fired alarm           : {len(with_alarm)}")
print(f"  alarm threshold       : signal_score < 0.40")
print()
if lts:
    print(f"  Lead time:  mean={np.mean(lts):.2f}  median={np.median(lts):.2f}  "
          f"min={np.min(lts)}  max={np.max(lts)}  [p25,p75]=[{np.percentile(lts,25):.1f},{np.percentile(lts,75):.1f}]")
    print()
    print(f"  Per profile:")
    for label, st in per_profile.items():
        print(f"    {label:<10} n={st['n_runs']:<3} div={st['n_diverged']:<3} "
              f"alarm={st['n_with_alarm']:<3} mean_lead={st['mean_lead_steps']}")
print(f"\n  ✓ Wrote {os.path.join(OUT_DIR, 'bench4_early_warning.json')}")
print(f"  ✓ Wrote {os.path.join(OUT_DIR, 'bench4_early_warning.png')}")
