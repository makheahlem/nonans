"""
NRI Failure-Trace Dataset Generator (CPU, numpy) — Tier-2 core deliverable.

Produces a LABELED dataset of real numerical-instability traces across multiple
tiny architectures and distinct failure modes. Each trace records, per step, the
NRI signals + ground-truth labels (failure mode, divergence step). This is the
citable resource: not one model, but a taxonomy of failures with reproducible
traces.

Architectures (tiny, CPU-fast):
  - linear stack       (deep linear net: rank/conditioning failures)
  - attention block    (softmax attention: head collapse / concentration)
  - mlp-relu           (ReLU MLP: dead units / scaling blowup)

Failure modes (the taxonomy):
  - RANK_COLLAPSE      progressive rank drain (low-norm, late-blowup)
  - ATTENTION_COLLAPSE attention fixates on few keys (entropy -> 0)
  - EXPLODING_NORM     multiplicative weight amplification -> overflow
  - DEAD_UNITS         ReLU units progressively zero out
  - HEALTHY            stable training (negative class, for FA calibration)

Each row of the dataset = one trace = {arch, mode, seed, T_divergence, per-step
signals: signal_score, attention_entropy, weight_norm, grad_proxy}.
"""
import numpy as np, json
from numpy.linalg import svd, qr

# ---- NRI signals (faithful to nonans.primitives) ----
def signal_score(sv):
    sv=np.abs(np.asarray(sv,dtype=np.float64)); k=sv.size
    if k<=1: return 0.0
    tot=sv.sum()
    if tot<1e-12: return 0.0
    p=sv/tot; return float(-(p*np.log(p+1e-10)).sum()/np.log(k))
def attn_entropy_norm(A):
    n=A.shape[-1]; return float((-np.sum(A*np.log(A+1e-10),axis=-1)/np.log(n)).mean())
def softmax_rows(X):
    X=X-X.max(axis=-1,keepdims=True); e=np.exp(X); return e/e.sum(axis=-1,keepdims=True)
def topk_sv(W,k=8):
    return svd(W.astype(np.float64),compute_uv=False)[:k]

# ---- trace generators per (arch, mode) ----
def gen_trace(arch, mode, seed, T=200, d=32):
    rng=np.random.default_rng(seed)
    rec={'signal_score':[], 'attn_entropy':[], 'weight_norm':[], 'grad_proxy':[]}
    T_div=None

    if arch=='linear':
        U,_=qr(rng.standard_normal((d,d))); V,_=qr(rng.standard_normal((d,d)))
        W=((U*np.linspace(1,0.7,d))@V.T).astype(np.float32)
        td=rng.standard_normal(d).astype(np.float32); td/=np.linalg.norm(td)
        Wp=W.copy()
        for t in range(T):
            Wp=W.copy()
            if mode=='RANK_COLLAPSE' and t>=10:
                W=(W+0.08*(np.outer(td,td)@W)).astype(np.float32)
                if t>=80: W=(W*(1+0.03*(t-80))).astype(np.float32)
            elif mode=='EXPLODING_NORM' and t>=20:
                W=(W*(1+0.05*(t-20)/10)).astype(np.float32)
            elif mode=='HEALTHY':
                W=(W+0.005*rng.standard_normal(W.shape).astype(np.float32))
            nrm=float(np.linalg.norm(W))
            if not np.isfinite(nrm): T_div=t; break
            rec['signal_score'].append(signal_score(topk_sv(W)))
            rec['weight_norm'].append(nrm)
            rec['grad_proxy'].append(float(np.linalg.norm(W-Wp)))
            # attention readout from W (proxy)
            X=rng.standard_normal((8,d)).astype(np.float32)
            rec['attn_entropy'].append(attn_entropy_norm(softmax_rows((X@W)/np.sqrt(d))))

    elif arch=='attention':
        scores0=rng.standard_normal((d,d)).astype(np.float32)*0.5
        fix=rng.integers(0,d)
        V=rng.standard_normal((d,d)).astype(np.float32)
        for t in range(T):
            sc=scores0.copy()
            if mode=='ATTENTION_COLLAPSE' and t>=30:
                sc[:,fix]+=0.5*(t-30)
            elif mode=='EXPLODING_NORM':
                # softmax is bounded; real overflow is in the VALUE path A@V.
                # grow V's norm multiplicatively until A@V overflows fp32.
                sc=scores0
            elif mode=='HEALTHY':
                scores0=scores0+0.1*rng.standard_normal((d,d)).astype(np.float32); sc=scores0
            A=softmax_rows(sc)
            # value projection (where attention-path overflow actually happens)
            if mode=='EXPLODING_NORM':
                if t==0: V=rng.standard_normal((d,d)).astype(np.float32)
                if t>=20: V=(V*(1+0.08*(t-20))).astype(np.float32)
                out=A@V
                if not np.all(np.isfinite(out)): T_div=t; break
                rec['weight_norm'].append(float(np.linalg.norm(out)))
            else:
                rec['weight_norm'].append(float(np.linalg.norm(sc)))
            if not np.all(np.isfinite(A)): T_div=t; break
            rec['attn_entropy'].append(attn_entropy_norm(A))
            rec['signal_score'].append(signal_score(topk_sv(A)))
            rec['grad_proxy'].append(float(np.linalg.norm(sc-scores0)) if mode!='HEALTHY' else 0.1)
        # attention "divergence" = entropy collapse to ~0 (functional failure, not NaN)
        if T_div is None and mode=='ATTENTION_COLLAPSE':
            below=[i for i,h in enumerate(rec['attn_entropy']) if h<0.05]
            T_div=below[0] if below else None

    elif arch=='mlp_relu':
        W1=rng.standard_normal((d,d)).astype(np.float32)*0.3
        for t in range(T):
            W1p=W1.copy()
            if mode=='DEAD_UNITS' and t>=15:
                # progressively zero rows (dead units)
                ndead=min(d-1,int((t-15)*0.5))
                W1[:ndead,:]=0.0
            elif mode=='EXPLODING_NORM' and t>=20:
                W1=(W1*(1+0.06*(t-20))).astype(np.float32)
            elif mode=='HEALTHY':
                W1=(W1+0.005*rng.standard_normal(W1.shape).astype(np.float32))
            nrm=float(np.linalg.norm(W1))
            if not np.isfinite(nrm): T_div=t; break
            rec['signal_score'].append(signal_score(topk_sv(W1)))
            rec['weight_norm'].append(nrm)
            rec['grad_proxy'].append(float(np.linalg.norm(W1-W1p)))
            X=rng.standard_normal((8,d)).astype(np.float32)
            h=np.maximum(0,X@W1)
            rec['attn_entropy'].append(attn_entropy_norm(softmax_rows(h/np.sqrt(d))) if h.sum()>0 else 0.0)
        if T_div is None and mode=='DEAD_UNITS':
            # functional death = signal_score collapse
            below=[i for i,s in enumerate(rec['signal_score']) if s<0.2]
            T_div=below[0] if below else None

    return dict(arch=arch, mode=mode, seed=seed, T_divergence=T_div,
                length=len(rec['signal_score']), signals=rec)

# ---- build the dataset ----
SPECS = [
    ('linear','RANK_COLLAPSE'), ('linear','EXPLODING_NORM'), ('linear','HEALTHY'),
    ('attention','ATTENTION_COLLAPSE'), ('attention','EXPLODING_NORM'), ('attention','HEALTHY'),
    ('mlp_relu','DEAD_UNITS'), ('mlp_relu','EXPLODING_NORM'), ('mlp_relu','HEALTHY'),
]
SEEDS = list(range(8))

dataset=[]
for arch,mode in SPECS:
    for s in SEEDS:
        dataset.append(gen_trace(arch,mode,s))

# ---- summary: did each failure mode actually produce failures? ----
print("=== NRI Failure Dataset — generation summary ===")
print(f"{'arch':12} {'mode':18} {'n':>3} {'diverged':>9} {'mean_T_div':>11}")
from collections import defaultdict
agg=defaultdict(list)
for r in dataset:
    agg[(r['arch'],r['mode'])].append(r)
for (arch,mode),rs in agg.items():
    divs=[r['T_divergence'] for r in rs if r['T_divergence'] is not None]
    mt=f"{np.mean(divs):.1f}" if divs else "—"
    print(f"{arch:12} {mode:18} {len(rs):>3} {len(divs):>9} {mt:>11}")

# Save
with open('nri_failure_dataset.json','w') as f:
    json.dump(dataset,f)
print(f"\nTotal traces: {len(dataset)}  saved to nri_failure_dataset.json")
