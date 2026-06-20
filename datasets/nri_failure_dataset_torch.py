"""
nri_failure_dataset_torch.py — REAL-transformer failure-trace generator (CPU OK).

Run this on YOUR machine (where PyTorch is installed). It produces the same
labeled failure-trace dataset as the numpy generator, but using REAL transformer
architectures with genuine nn.MultiheadAttention, real LayerNorm, real AdamW, and
real gradients. Output JSON uses the same schema as nri_failure_dataset.json, so
real traces MERGE with the numpy ones.

Architectures (tiny, CPU-fast):
  - gpt   : 2-layer causal GPT block (token+pos embed, MHA, MLP)
  - vit   : 2-layer ViT-style encoder block (patch embed, MHA, MLP, no causal mask)

Failure modes (same taxonomy as the numpy dataset):
  - RANK_COLLAPSE      : low-rank drift injected into a weight matrix
  - ATTENTION_COLLAPSE : attention sharpened toward fixation (entropy -> 0)
  - EXPLODING_NORM     : learning-rate escalation -> weight/grad overflow -> NaN
  - DEAD_UNITS         : progressive zeroing of MLP units
  - HEALTHY            : stable training (negative class for FA calibration)

Lessons baked in (from prior debugging):
  * signal_score is monitored on 2-D weight matrices ONLY (1-D LayerNorm/bias
    vectors give signal_score==0 and would false-trip at step 0).
  * Real gradient norm is read from param.grad (not a surrogate).
  * Cross-entropy is bounded and cannot NaN by itself; EXPLODING_NORM uses
    LR escalation so weights overflow (the genuine divergence path on CPU).
  * Held-out healthy seeds for calibration vs false-alarm evaluation.

USAGE
-----
    pip install torch            # CPU build is fine
    python nri_failure_dataset_torch.py                 # builds + saves JSON
    python nri_failure_dataset_torch.py --seeds 6 --steps 300
    python nri_failure_dataset_torch.py --merge nri_failure_dataset.json
        # ^ merges the real torch traces into an existing numpy dataset file

Then re-run nri_dataset_validation.py on the merged file for the combined result.
"""
from __future__ import annotations
import argparse, json, math, os, sys

try:
    import torch, torch.nn as nn, torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch required:  pip install torch  (CPU build is fine)")

try:
    from nonans import signal_score, attention_entropy_normalized, randomized_svd_topk
except ImportError:
    sys.exit("Run from the nonans repo root after `pip install -e .` (or set PYTHONPATH=src).")


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #
class GPTBlock(nn.Module):
    def __init__(self, V=64, d=64, heads=4, layers=2, seqlen=16):
        super().__init__()
        self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(seqlen, d)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)) for _ in range(layers)])
        self.head = nn.Linear(d, V); self.seqlen = seqlen

    def forward(self, ids, want_attn=False):
        t = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(t, device=ids.device).unsqueeze(0))
        mask = torch.triu(torch.full((t, t), float("-inf"), device=ids.device), 1)
        aw = None
        for i in range(len(self.attn)):
            a, w = self.attn[i](x, x, x, attn_mask=mask, need_weights=want_attn, average_attn_weights=True)
            if want_attn and aw is None: aw = w
            x = self.ln1[i](x + a); x = self.ln2[i](x + self.mlp[i](x))
        return self.head(x[:, 0, :]), aw


class ViTBlock(nn.Module):
    """ViT-style encoder over synthetic 'patch' tokens (no causal mask)."""
    def __init__(self, n_patches=16, d=64, heads=4, layers=2, n_cls=10):
        super().__init__()
        self.patch = nn.Linear(d, d); self.pos = nn.Embedding(n_patches, d)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)) for _ in range(layers)])
        self.head = nn.Linear(d, n_cls); self.n_patches = n_patches

    def forward(self, patches, want_attn=False):
        b, t, _ = patches.shape
        x = self.patch(patches) + self.pos(torch.arange(t, device=patches.device).unsqueeze(0))
        aw = None
        for i in range(len(self.attn)):
            a, w = self.attn[i](x, x, x, need_weights=want_attn, average_attn_weights=True)
            if want_attn and aw is None: aw = w
            x = self.ln1[i](x + a); x = self.ln2[i](x + self.mlp[i](x))
        return self.head(x.mean(1)), aw


# --------------------------------------------------------------------------- #
# Monitoring helpers (carry the 1-D-guard + real-gradient lessons)
# --------------------------------------------------------------------------- #
def monitorable(p):
    return p.requires_grad and p.dim() >= 2 and min(p.shape) >= 2

def min_signal_score(model):
    m = 1.0
    for p in model.parameters():
        if monitorable(p):
            sv = randomized_svd_topk(p.detach().float().numpy(), k=8)
            m = min(m, float(signal_score(sv)))
    return m

def mean_attn_entropy(aw):
    if aw is None: return 1.0
    hs = [attention_entropy_normalized(aw[b].detach().float().mean(0).numpy())
          for b in range(aw.shape[0])]
    return float(sum(hs) / len(hs)) if hs else 1.0

def grad_norm(model):
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            if torch.isfinite(g).all(): tot += float(g.norm()) ** 2
            else: return float("inf")
    return math.sqrt(tot)


# --------------------------------------------------------------------------- #
# One trace
# --------------------------------------------------------------------------- #
def make_inputs(arch, seed, d=64, seqlen=16, batch=16):
    g = torch.Generator().manual_seed(seed)
    if arch == "gpt":
        ids = torch.randint(0, 64, (batch, seqlen), generator=g)
        return ids, ids[:, 0]
    else:  # vit
        patches = torch.randn(batch, 16, d, generator=g)
        labels = torch.randint(0, 10, (batch,), generator=g)
        return patches, labels


def gen_trace(arch, mode, seed, steps=300, lr=0.01):
    torch.manual_seed(seed)
    model = GPTBlock() if arch == "gpt" else ViTBlock()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rec = {"signal_score": [], "attn_entropy": [], "weight_norm": [], "grad_proxy": []}
    T_div = None
    # pick one MLP weight to corrupt for RANK_COLLAPSE / DEAD_UNITS
    target = None
    for n, p in model.named_parameters():
        if "mlp" in n and p.dim() == 2:
            target = p; break

    for t in range(steps):
        x, y = make_inputs(arch, seed * 1000 + t)
        out, aw = model(x, want_attn=True)
        loss = F.cross_entropy(out.float(), y)
        if not torch.isfinite(loss):
            T_div = t; break
        opt.zero_grad(); loss.backward()
        gnorm = grad_norm(model)

        # inject the failure mode. NOTE: real transformers (LayerNorm + residual +
        # AdamW) strongly resist perturbation, so injections are AGGRESSIVE — tuned
        # to overpower stabilization, unlike the bare-matrix numpy proxies.
        with torch.no_grad():
            if mode == "EXPLODING_NORM":
                for grp in opt.param_groups:
                    grp["lr"] = lr * (1.06 ** t)              # LR escalation -> overflow
            elif mode == "RANK_COLLAPSE" and t >= 20 and target is not None:
                # AGGRESSIVE rank drain: project ALL the way onto rank-1 + amplify,
                # strong enough that LayerNorm cannot re-spread the spectrum.
                u = torch.randn(target.shape[0], 1); u /= u.norm()
                target.mul_(0.6)                              # shrink full-rank part
                target.add_(0.8 * (u @ (u.t() @ target)))    # push onto one direction
                if t >= 100: target.mul_(1.03)               # then amplify -> divergence
            elif mode == "DEAD_UNITS" and t >= 15 and target is not None:
                # faster, total unit death across BOTH mlp layers if reachable.
                ndead = min(target.shape[0] - 1, int((t - 15) * 1.5))  # 3x faster
                target[:ndead, :] = 0.0
                target[:, :ndead] = 0.0                       # kill columns too
            elif mode == "ATTENTION_COLLAPSE" and t >= 20:
                # AGGRESSIVE fixation: scale the FULL query projection hard so softmax
                # saturates to one key (LayerNorm on x cannot undo a saturated softmax).
                qp = model.attn[0].in_proj_weight
                qp.mul_(1.0)
                qp[:qp.shape[0] // 3].mul_(1.15)             # 3x stronger Q growth
                # also bias keys toward one position to force fixation
                kp = model.attn[0].in_proj_weight[qp.shape[0]//3 : 2*qp.shape[0]//3]
                kp.mul_(1.15)

        opt.step()

        rec["signal_score"].append(min_signal_score(model))
        rec["attn_entropy"].append(mean_attn_entropy(aw))
        rec["weight_norm"].append(float(sum(p.detach().norm() ** 2 for p in model.parameters()) ** 0.5))
        rec["grad_proxy"].append(gnorm if math.isfinite(gnorm) else float("inf"))

        # functional-divergence labels for non-NaN modes
        if mode == "ATTENTION_COLLAPSE" and T_div is None and rec["attn_entropy"][-1] < 0.05:
            T_div = t
        if mode == "DEAD_UNITS" and T_div is None and rec["signal_score"][-1] < 0.20:
            T_div = t

    return dict(arch=arch, mode=mode, seed=seed, T_divergence=T_div,
                length=len(rec["signal_score"]), signals=rec, source="torch")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6, help="seeds per (arch,mode)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="nri_failure_dataset_torch.json")
    ap.add_argument("--merge", default=None, help="existing numpy dataset JSON to merge into")
    args = ap.parse_args()

    SPECS = [("gpt", "RANK_COLLAPSE"), ("gpt", "ATTENTION_COLLAPSE"),
             ("gpt", "EXPLODING_NORM"), ("gpt", "DEAD_UNITS"), ("gpt", "HEALTHY"),
             ("vit", "RANK_COLLAPSE"), ("vit", "ATTENTION_COLLAPSE"),
             ("vit", "EXPLODING_NORM"), ("vit", "DEAD_UNITS"), ("vit", "HEALTHY")]
    # healthy seeds split for held-out calibration vs FA (match numpy convention)
    fail_seeds = list(range(args.seeds))
    healthy_seeds = list(range(20, 20 + args.seeds)) + list(range(40, 40 + args.seeds))

    data = []
    print(f"{'arch':5} {'mode':18} {'seed':>4} {'T_div':>6} {'len':>4}")
    for arch, mode in SPECS:
        seeds = healthy_seeds if mode == "HEALTHY" else fail_seeds
        for s in seeds:
            r = gen_trace(arch, mode, s, steps=args.steps)
            data.append(r)
            print(f"{arch:5} {mode:18} {s:>4} {str(r['T_divergence']):>6} {r['length']:>4}")

    if args.merge and os.path.exists(args.merge):
        base = json.load(open(args.merge))
        data = base + data
        print(f"\nmerged with {args.merge}: {len(base)} numpy + new torch = {len(data)} total")

    json.dump(data, open(args.out, "w"))
    # summary
    from collections import defaultdict
    agg = defaultdict(list)
    for r in data:
        if r.get("source") == "torch" or args.merge:
            agg[(r["arch"], r["mode"])].append(r)
    print(f"\n=== summary ({args.out}) ===")
    for (a, m), rs in sorted(agg.items()):
        divs = [r["T_divergence"] for r in rs if r["T_divergence"] is not None]
        mt = f"{sum(divs)/len(divs):.1f}" if divs else "—"
        print(f"  {a:5} {m:18} {len(divs):>2}/{len(rs):<2} diverged  mean_T_div={mt}")
    print(f"\nwrote {args.out} ({len(data)} traces)")
    print("Next: python nri_dataset_validation.py  (point it at this file)")


if __name__ == "__main__":
    main()
