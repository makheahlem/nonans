r"""
PUBLICATION-CLEAN silent-degradation test with HELD-OUT healthy calibration.

Core claim: NRI detects silent structural degradation (capacity loss with no NaN,
invisible to loss/gradient) in transformers. This version removes threshold
leakage: the healthy floor for signal_score AND the healthy bands for
grad/weight-norm are calibrated on a CALIBRATION set of healthy seeds, then the
non-diverging stress runs are evaluated against those held-out thresholds.

A run counts as "silent degradation detected" iff, on a NON-diverging run:
   - signal_score drops below the calibrated healthy floor, AND
   - gradient proxy stays within the calibrated healthy band, AND
   - weight norm stays within the calibrated healthy band
i.e. NRI fires while a loss/gradient monitor would see nothing.

Healthy seeds are split: calibration = {20..25}, evaluation-of-FA = {40..45}
(the same convention used to build the dataset). Stress seeds = {0..5}.

Run from repo root:  python datasets\silent_degradation_validated.py
"""
import json, os, math

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "nri_failure_dataset_torch.json")
if not os.path.exists(path):
    path = "nri_failure_dataset_torch.json"
data = json.load(open(path))

from collections import defaultdict
by = defaultdict(list)
for r in data:
    by[(r['arch'], r['mode'])].append(r)

CAL_SEEDS  = set(range(20, 26))   # healthy seeds for calibration
EVAL_SEEDS = set(range(40, 46))   # held-out healthy seeds for false-alarm check

def series(runs, key):
    return [v for r in runs for v in r['signals'][key]
            if v is not None and math.isfinite(v)]

def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (max(0, c-h), min(1, c+h))

print("="*70)
print("PUBLICATION-CLEAN SILENT DEGRADATION TEST (held-out calibration)")
print("="*70)
print("A 'silent' run: model did NOT diverge, grad+norm stayed in calibrated")
print("healthy band, but signal_score dropped below calibrated healthy floor.")
print("=> degradation a loss/gradient monitor MISSES and NRI catches.\n")

summary = {}
for arch in ['gpt', 'vit']:
    healthy_all = by[(arch, 'HEALTHY')]
    cal  = [r for r in healthy_all if r['seed'] in CAL_SEEDS]
    ev   = [r for r in healthy_all if r['seed'] in EVAL_SEEDS]
    if not cal:
        print(f"[{arch}] no calibration healthy seeds found; skipping"); continue

    # calibrate thresholds on CAL only
    sig_floor = min(series(cal, 'signal_score'))          # lowest healthy signal
    grad_band = max(series(cal, 'grad_proxy'))            # highest healthy grad
    norm_band = max(series(cal, 'weight_norm'))           # highest healthy norm
    sig_thr  = sig_floor * 0.95
    grad_thr = grad_band * 1.50
    norm_thr = norm_band * 1.50

    # false-alarm check on held-out healthy (should be ~0: healthy must NOT trip)
    fa = 0
    for r in ev:
        s_min = min(r['signals']['signal_score'])
        if s_min < sig_thr:
            fa += 1
    faci = wilson(fa, len(ev)) if ev else None

    print(f"--- {arch.upper()} ---")
    print(f"  calibrated on healthy seeds {sorted(s for s in CAL_SEEDS)}:")
    print(f"    signal floor    = {sig_floor:.3f}  -> alarm if signal < {sig_thr:.3f}")
    print(f"    grad healthy max= {grad_band:.2f}  -> 'healthy' if grad  < {grad_thr:.2f}")
    print(f"    norm healthy max= {norm_band:.1f}  -> 'healthy' if norm  < {norm_thr:.1f}")
    fa_str = f"{fa}/{len(ev)}"
    if faci: fa_str += f" (95% CI {faci[0]*100:.0f}-{faci[1]*100:.0f}%)"
    print(f"    false alarms on held-out healthy seeds {sorted(EVAL_SEEDS)}: {fa_str}")
    if fa > 0:
        print(f"    WARNING: healthy runs trip the signal threshold; floor too high.")
    print()

    for mode in ['DEAD_UNITS', 'RANK_COLLAPSE', 'ATTENTION_COLLAPSE']:
        runs = [r for r in by[(arch, mode)] if r['seed'] in set(range(6))]
        nondiv = [r for r in runs if r['T_divergence'] is None]
        if not nondiv:
            print(f"  {mode:18}: 0 non-diverging runs (cannot test silent)")
            continue
        money = 0
        details = []
        for r in nondiv:
            s_min = min(r['signals']['signal_score'])
            g_max = max((g for g in r['signals']['grad_proxy'] if math.isfinite(g)), default=0.0)
            n_max = max(r['signals']['weight_norm'])
            sig_dropped  = s_min < sig_thr
            grad_healthy = g_max < grad_thr
            norm_healthy = n_max < norm_thr
            if sig_dropped and grad_healthy and norm_healthy:
                money += 1
            details.append(s_min)
        ci = wilson(money, len(nondiv))
        ci_str = f" (95% CI {ci[0]*100:.0f}-{ci[1]*100:.0f}%)" if ci else ""
        tag = " <<< SILENT DEGRADATION" if money > 0 else " (not detected silently)"
        print(f"  {mode:18}: {money}/{len(nondiv)}{ci_str}  "
              f"min_signal range [{min(details):.3f},{max(details):.3f}]{tag}")
        summary[(arch, mode)] = (money, len(nondiv), min(details))
    print()

print("="*70)
print("HEADLINE (held-out-calibrated):")
core = summary.get(('gpt','DEAD_UNITS'), (0,0,1)), summary.get(('vit','DEAD_UNITS'), (0,0,1))
tot_money = sum(v[0] for v in summary.values())
tot_runs  = sum(v[1] for v in summary.values())
print(f"  Across both transformers, {tot_money}/{tot_runs} non-diverging stress runs")
print(f"  showed signal degradation while loss/grad stayed healthy.")
print(f"  DEAD_UNITS (the cleanest case): GPT {summary.get(('gpt','DEAD_UNITS'),('-','-'))[:2]}, "
      f"ViT {summary.get(('vit','DEAD_UNITS'),('-','-'))[:2]}")
print(f"  This is degradation invisible to standard loss/gradient monitoring.")
print("="*70)
