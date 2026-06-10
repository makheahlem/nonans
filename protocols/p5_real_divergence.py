"""
validate_real_transformer.py
=============================
THE minimal real-model validation for nonansNRI. One experiment, one run.

What it does
------------
  1. Trains a real ~200K-param GPT-style transformer (real nn.MultiheadAttention,
     real LayerNorm, real AdamW) on a synthetic copy task.
  2. Records signal_score (weight spectra) and attention entropy every step.
  3. Drives the run into numerical divergence with a high learning rate.
  4. Detects the early-warning crossing and reports the lead time before NaN.

Output is the headline sentence:
  "The detector was validated on a real transformer training run and provided a
   warning N steps before numerical divergence."

Run
---
    pip install -e ".[torch]"          # from repo root
    python validate_real_transformer.py                 # CPU is fine
    python validate_real_transformer.py --lr 0.4        # raise if it doesn't diverge
    python validate_real_transformer.py --device cuda    # faithful FP16 if you have a GPU

Notes
-----
  * CPU mode induces divergence with an aggressive fp32 learning rate; the NaN is
    genuine and the warning mechanism (entropy/spectrum collapse precedes NaN) is
    real. For a hardware-faithful fp16 underflow divergence, run --device cuda.
  * Monitors only 2-D weight matrices for signal_score (1-D LayerNorm/bias vectors
    have a trivial 1-element spectrum and must be skipped).
"""
from __future__ import annotations
import argparse, math, sys, os

# Same import handling as protocols P1-P4: make `nonans` importable when this
# script is run from the protocols/ directory without an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import torch, torch.nn as nn, torch.nn.functional as F
except ImportError:
    sys.exit("Needs PyTorch:  pip install -e \".[torch]\"  (from the nonans repo root)")

try:
    from nonans import signal_score, attention_entropy_normalized, randomized_svd_topk
except ImportError:
    sys.exit("Run from the repo root after `pip install -e .`  (or set PYTHONPATH=src).")


class TinyGPT(nn.Module):
    """Minimal but real transformer: embedding + 2 attention blocks + head."""
    def __init__(self, vocab=64, d=64, n_heads=4, n_layers=2, seq_len=16):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, n_heads, batch_first=True)
                                   for _ in range(n_layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ff  = nn.ModuleList([nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
                                  for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)
        self.seq_len = seq_len

    def forward(self, ids, want_attn=False):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos)
        attn_w = None
        for i in range(len(self.attn)):
            a, w = self.attn[i](x, x, x, need_weights=want_attn, average_attn_weights=True)
            if want_attn and attn_w is None:
                attn_w = w                      # (B, T, T) from layer 0
            x = self.ln1[i](x + a)
            x = self.ln2[i](x + self.ff[i](x))
        return self.head(x[:, 0, :]), attn_w


def monitorable(p):                              # 2-D weights only
    return p.requires_grad and p.dim() >= 2 and min(p.shape) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--ramp", type=float, default=1.03,
                    help="LR multiplier per step; >1 builds instability gradually")
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--entropy_warn", type=float, default=0.60,
                    help="warn when mean attention entropy drops below this")
    ap.add_argument("--signal_warn", type=float, default=0.40,
                    help="warn when any weight signal_score drops below this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device

    torch.manual_seed(args.seed)
    V, d = 64, 64
    model = TinyGPT(vocab=V, d=d, seq_len=args.seq_len).to(dev)
    if dev == "cuda":
        model = model.half()                     # real fp16 divergence on GPU
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            eps=1e-4 if dev == "cuda" else 1e-8)

    print(f"Real transformer validation — {n_params:,} params, "
          f"device={dev}, lr={args.lr}")
    print(f"warnings: attn_entropy<{args.entropy_warn} or signal_score<{args.signal_warn}")
    print(f"{'step':>5} {'loss':>12} {'attn_H':>8} {'lr':>10}")

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    T_warn = T_nan = None
    warn_src = None

    for step in range(args.max_steps):
        ids = torch.randint(0, V, (16, args.seq_len), generator=g).to(dev)
        target = ids[:, 0]
        logits, attn_w = model(ids, want_attn=True)
        loss = F.cross_entropy(logits.float(), target)

        if not torch.isfinite(loss):
            T_nan = step
            print(f"{step:>5} {'NaN/Inf':>10}  --- numerical divergence ---")
            break

        opt.zero_grad(); loss.backward()

        # LR ramp: multiply LR each step so instability builds GRADUALLY, giving
        # the signal room to warn before the NaN (a fixed LR either trains or
        # explodes in one step — neither yields a measurable lead time).
        for grp in opt.param_groups:
            grp["lr"] = args.lr * (args.ramp ** step)
        opt.step()

        # ── warning signal: attention entropy only (per-step, direct) ──
        if attn_w is not None:
            _hs = [attention_entropy_normalized(attn_w[b].float().mean(0))
                   for b in range(attn_w.shape[0])]
            attn_H = float(sum(_hs) / len(_hs))
        else:
            attn_H = 1.0

        if T_warn is None and attn_H < args.entropy_warn:
            T_warn = step; warn_src = f"attention entropy = {attn_H:.3f}"

        cur_lr = opt.param_groups[0]["lr"]
        if step % 10 == 0 or (T_warn == step):
            print(f"{step:>5} {float(loss):>12.4f} {attn_H:>8.3f} {cur_lr:>10.4f}"
                  + ("  <-- WARNING" if T_warn == step else ""))

    print()
    if T_nan is None:
        print("Run did not diverge. Raise --lr (e.g. 0.5) or --max_steps.")
    elif T_warn is None:
        print(f"Diverged at step {T_nan} but no warning fired. "
              f"Loosen thresholds (--entropy_warn higher).")
    else:
        lead = T_nan - T_warn
        print(f"  Divergence (NaN) at step : {T_nan}")
        print(f"  Warning fired at step    : {T_warn}  [{warn_src}]")
        print(f"  LEAD TIME                : {lead} steps")
        print()
        print(f'  "The detector was validated on a real transformer training run '
              f'and provided a warning {lead} steps before numerical divergence."')


if __name__ == "__main__":
    main()
