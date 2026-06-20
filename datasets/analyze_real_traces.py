"""
Analyze the REAL-transformer traces (nri_failure_dataset_torch.json): does the
NRI signal RESPOND to injected stress even when the model does NOT diverge?
If signal_score / attn_entropy drift during RANK/ATTENTION/DEAD injection (vs
HEALTHY baseline), NRI detects structural stress the architecture absorbs --
still-useful monitoring, reported honestly.

Run: python datasets\analyze_real_traces.py
"""
import json, os, math

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "nri_failure_dataset_torch.json")
if not os.path.exists(path):
    path = "nri_failure_dataset_torch.json"
data = json.load(open(path))

from collections import defaultdict
g = defaultdict(list)
for r in data:
    g[(r['arch'], r['mode'])].append(r)

def stats(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs: return (None, None, None)
    return (min(xs), sum(xs)/len(xs), max(xs))

# Baseline: healthy signal range per arch (what "no stress" looks like)
print("=== Does the NRI signal respond to injected stress on REAL transformers? ===\n")
for arch in ['gpt','vit']:
    healthy = g[(arch,'HEALTHY')]
    h_sig = [v for r in healthy for v in r['signals']['signal_score']]
    h_ent = [v for r in healthy for v in r['signals']['attn_entropy']]
    hs = stats(h_sig); he = stats(h_ent)
    print(f"--- {arch.upper()} ---")
    print(f"  HEALTHY baseline:  signal_score [{hs[0]:.3f},{hs[2]:.3f}]  attn_entropy [{he[0]:.3f},{he[2]:.3f}]")
    for mode in ['RANK_COLLAPSE','ATTENTION_COLLAPSE','DEAD_UNITS','EXPLODING_NORM']:
        runs = g[(arch,mode)]
        if not runs: continue
        # min signal reached during injection (lower = more stress detected)
        sig_mins = [min(r['signals']['signal_score']) for r in runs if r['signals']['signal_score']]
        ent_mins = [min(r['signals']['attn_entropy']) for r in runs if r['signals']['attn_entropy']]
        sm = stats(sig_mins); em = stats(ent_mins)
        diverged = sum(1 for r in runs if r['T_divergence'] is not None)
        # does signal drop below healthy floor? (= detected stress)
        sig_resp = "YES" if (sm[1] is not None and hs[0] is not None and sm[1] < hs[0]*0.95) else "no"
        ent_resp = "YES" if (em[1] is not None and he[0] is not None and em[1] < he[0]*0.95) else "no"
        print(f"  {mode:18} div={diverged}/{len(runs)}  "
              f"min_signal~{sm[1]:.3f}({sig_resp})  min_entropy~{em[1]:.3f}({ent_resp})")
    print()
print("Reading: 'YES' = signal dropped below the healthy floor => NRI detected the")
print("injected stress even if the model did not diverge. 'no' = architecture")
print("absorbed it AND the signal stayed in healthy range (truly invisible).")
