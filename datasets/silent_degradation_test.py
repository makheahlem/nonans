r"""
THE money-plot test: on NON-DIVERGING stress runs, does loss/gradient stay
HEALTHY while the NRI signal_score DROPS? If so, NRI detects silent structural
degradation that loss-based monitoring structurally cannot see.

This is the core claim: NRI's value is NOT (only) divergence prediction (crowded
space, architecture buffers most of it) but detection of degradation the
architecture masks -- invisible to loss/gradient, never a NaN (raw string to avoid escape warning).

Run: python datasets\silent_degradation_test.py
"""
import json, os, math

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "nri_failure_dataset_torch.json")
if not os.path.exists(path): path = "nri_failure_dataset_torch.json"
data = json.load(open(path))

from collections import defaultdict
g = defaultdict(list)
for r in data: g[(r['arch'], r['mode'])].append(r)

def frange(xs):
    xs=[x for x in xs if x is not None and math.isfinite(x)]
    return (min(xs),max(xs)) if xs else (None,None)

print("=== SILENT DEGRADATION TEST ===")
print("Claim: NRI signal drops while loss/grad stay healthy, on NON-diverging runs.\n")

for arch in ['gpt','vit']:
    # healthy reference for ALL three signals
    H = g[(arch,'HEALTHY')]
    h_sig = frange([v for r in H for v in r['signals']['signal_score']])
    h_grad= frange([v for r in H for v in r['signals']['grad_proxy']])
    h_norm= frange([v for r in H for v in r['signals']['weight_norm']])
    print(f"--- {arch.upper()} | healthy: signal[{h_sig[0]:.3f},{h_sig[1]:.3f}] "
          f"grad[{h_grad[0]:.2f},{h_grad[1]:.2f}] norm[{h_norm[0]:.1f},{h_norm[1]:.1f}] ---")

    for mode in ['DEAD_UNITS','RANK_COLLAPSE','ATTENTION_COLLAPSE']:
        runs = g[(arch,mode)]
        nondiv = [r for r in runs if r['T_divergence'] is None]
        if not nondiv: 
            print(f"  {mode:18}: all runs diverged (n/a for silent test)")
            continue
        # on non-diverging runs: did signal drop below healthy floor while grad+norm stayed in band?
        money_count = 0
        for r in nondiv:
            sig_min = min(r['signals']['signal_score']) if r['signals']['signal_score'] else 1.0
            grad_max= max(g_ for g_ in r['signals']['grad_proxy'] if math.isfinite(g_)) if r['signals']['grad_proxy'] else 0.0
            norm_max= max(r['signals']['weight_norm']) if r['signals']['weight_norm'] else 0.0
            signal_dropped = sig_min < h_sig[0]*0.95
            grad_healthy   = grad_max <= h_grad[1]*1.5   # grad stayed in healthy band
            norm_healthy   = norm_max <= h_norm[1]*1.5
            if signal_dropped and grad_healthy and norm_healthy:
                money_count += 1
        verdict = "<<< SILENT DEGRADATION DETECTED" if money_count>0 else "(no silent detection)"
        print(f"  {mode:18}: {money_count}/{len(nondiv)} runs = signal DROPPED while "
              f"grad+norm stayed HEALTHY  {verdict}")
    print()

print("Reading: a 'money' run = the model did NOT diverge, its loss-proxy (grad)")
print("and weight norm stayed in healthy range, BUT signal_score dropped below the")
print("healthy floor. Those runs are degradation a loss/grad monitor would MISS")
print("and NRI catches. This is the core, competition-free contribution.")
