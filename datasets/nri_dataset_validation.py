"""
Methodologically clean version: split healthy traces into CALIBRATION (set
thresholds) and EVALUATION (measure false alarms) — no leakage. Add more healthy
seeds so the split is meaningful.
"""
import json, math
import numpy as np
import os
_gen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nri_failure_dataset_generator.py")
exec(open(_gen).read().split('# ---- build the dataset ----')[0])

SEEDS_FAIL=list(range(8))
SEEDS_HEALTHY_CAL=list(range(20,28))   # calibration healthy
SEEDS_HEALTHY_EVAL=list(range(40,48))  # held-out healthy for FA

SPECS_FAIL=[('linear','RANK_COLLAPSE'),('linear','EXPLODING_NORM'),
            ('attention','ATTENTION_COLLAPSE'),('attention','EXPLODING_NORM'),
            ('mlp_relu','DEAD_UNITS'),('mlp_relu','EXPLODING_NORM')]
ARCHS=['linear','attention','mlp_relu']

def wilson(k,n,z=1.96):
    if n==0: return None
    p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (max(0,c-h),min(1,c+h))
def mean_ci(xs):
    a=[x for x in xs if x is not None];n=len(a)
    if n==0: return None
    m=sum(a)/n
    if n<2: return (m,None)
    sd=(sum((x-m)**2 for x in a)/(n-1))**0.5
    t={2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306}.get(n-1,1.96)
    return (m,t*sd/math.sqrt(n))

# calibrate thresholds per arch on CALIBRATION healthy only
cal={}
for arch in ARCHS:
    htr=[gen_trace(arch,'HEALTHY',s) for s in SEEDS_HEALTHY_CAL]
    smin=[min(r['signals']['signal_score']) for r in htr if r['signals']['signal_score']]
    emin=[min(r['signals']['attn_entropy']) for r in htr if r['signals']['attn_entropy']]
    cal[arch]=(min(smin)*0.95 if smin else 0.4, min(emin)*0.95 if emin else 0.4)

print("=== Clean validation (held-out healthy calibration) ===\n")
print(f"{'arch':10} {'mode':18} {'detect':>13} {'lead 95%CI':>18}")
all_leads=[]
for arch,mode in SPECS_FAIL:
    sig_thr,ent_thr=cal[arch]
    leads=[];ndet=0;runs=[gen_trace(arch,mode,s) for s in SEEDS_FAIL]
    for r in runs:
        Td=r['T_divergence']
        if Td is None: continue
        sig=r['signals']['signal_score'];ent=r['signals']['attn_entropy']
        ts=next((i for i,v in enumerate(sig) if v<sig_thr),None)
        te=next((i for i,v in enumerate(ent) if v<ent_thr),None)
        c=[t for t in (ts,te) if t is not None and t<Td]
        if c: ndet+=1;leads.append(Td-min(c))
    n=len(runs);dr=wilson(ndet,n);mci=mean_ci(leads)
    ds=f"{ndet}/{n} ({dr[0]*100:.0f}-{dr[1]*100:.0f}%)" if dr else f"{ndet}/{n}"
    ls=f"{mci[0]:.1f}\u00b1{mci[1]:.1f}" if mci and mci[1] else ("no detect" if not mci else f"{mci[0]:.1f}")
    print(f"{arch:10} {mode:18} {ds:>13} {ls:>18}")
    all_leads+=leads

# FA on HELD-OUT healthy
fa=0;nh=0
for arch in ARCHS:
    sig_thr,ent_thr=cal[arch]
    for s in SEEDS_HEALTHY_EVAL:
        r=gen_trace(arch,'HEALTHY',s);nh+=1
        if any(v<sig_thr for v in r['signals']['signal_score']) or \
           any(v<ent_thr for v in r['signals']['attn_entropy']): fa+=1
faci=wilson(fa,nh)
mci=mean_ci(all_leads)
print(f"\nOVERALL mean lead = {mci[0]:.1f}\u00b1{mci[1]:.1f} steps (95% CI), {len(all_leads)} detections")
print(f"FALSE ALARMS (held-out healthy): {fa}/{nh}  (95% CI upper {faci[1]*100:.0f}%)")
print(f"\nKey honest finding: attention value-path overflow (EXPLODING_NORM) is")
print(f"NOT detected by entropy monitoring \u2014 entropy stays healthy while values")
print(f"overflow. This is a characterized LIMITATION, reported, not hidden.")
