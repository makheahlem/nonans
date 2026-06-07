"""
Protocol P3: Early-warning signal on a real FP16 training run
==============================================================
Trains a small transformer on a synthetic copy task in FP16 (mixed
precision, no gradient clipping, no loss scaling) until it produces
a NaN. Measures the lead time of the entropy-based alarm.

This protocol crosses the instability boundary by:
  1. Using FP16 (float16) throughout — the primary source of training
     instability in practice (loss-of-significance in softmax keys)
  2. Setting a high learning rate without gradient clipping
  3. Disabling AMP loss scaling intentionally

You may need to tune --lr upward if your hardware does not diverge
in the default max_steps. Typical divergence LR: 0.05–0.3.

Usage
-----
    python protocols/run_early_warning_torch.py --lr 0.1 --max_steps 500

Dependencies: torch ≥ 1.12  (CPU works; GPU optional)
"""
import torch, torch.nn as nn, argparse, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nonans import signal_score, attention_entropy_normalized, randomized_svd_topk

parser = argparse.ArgumentParser()
parser.add_argument("--lr",        type=float, default=0.1)
parser.add_argument("--max_steps", type=int,   default=500)
parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--alarm_threshold", type=float, default=0.40)
args = parser.parse_args()
device = args.device


# ─── Model ────────────────────────────────────────────────────────────────────

class TinyCopyTransformer(nn.Module):
    def __init__(self, V=64, d=64, n_heads=4, seq_len=16):
        super().__init__()
        self.emb    = nn.Embedding(V, d)
        self.attn   = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln1    = nn.LayerNorm(d)
        self.ff1    = nn.Linear(d, 4 * d)
        self.ff2    = nn.Linear(4 * d, d)
        self.ln2    = nn.LayerNorm(d)
        self.head   = nn.Linear(d, V)
        self.seq_len = seq_len

    def forward(self, ids):
        x = self.emb(ids).to(torch.float16)              # FP16 embedding
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff2(torch.relu(self.ff1(x)))
        x = self.ln2(x + ff_out)
        logits = self.head(x[:, 0, :])                   # predict from first token
        return logits, attn_weights                       # attn_weights: (B, T, T)


V, d, seq_len = 64, 64, 16
model = TinyCopyTransformer(V=V, d=d, seq_len=seq_len).to(device)
# Cast entire model to FP16 for maximal instability exposure
model = model.half()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
criterion = nn.CrossEntropyLoss()

print(f"Protocol P3 — Early warning on FP16 training")
print(f"  device={device}  lr={args.lr}  max_steps={args.max_steps}")
print(f"  alarm threshold: signal_score < {args.alarm_threshold}")
print()

rng = torch.Generator()
rng.manual_seed(42)
T_alarm, T_nan = None, None
alarm_source   = None

for step in range(args.max_steps):
    ids    = torch.randint(0, V, (16, seq_len), generator=rng, device=device)
    target = ids[:, 0]

    try:
        logits, attn_weights = model(ids)
        # attn_weights: (B, T, T) averaged over heads
        loss = criterion(logits.float(), target)
    except Exception as e:
        print(f"  [step {step:>4}] Exception in forward: {e}")
        T_nan = step; break

    if not torch.isfinite(loss):
        print(f"  [step {step:>4}] Non-finite loss detected. T_nan = {step}")
        T_nan = step; break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ── Monitor ──
    for name, param in model.named_parameters():
        if param.requires_grad and param.numel() >= 16:
            sv  = randomized_svd_topk(param.data.float(), k=8)
            sig = signal_score(sv)
            if sig < args.alarm_threshold and T_alarm is None:
                T_alarm = step
                alarm_source = f"signal_score({name}) = {sig:.3f}"

    if attn_weights is not None:
        for b in range(attn_weights.shape[0]):
            H = attention_entropy_normalized(attn_weights[b].float().mean(0))
            if (H < 0.2 or H > 0.95) and T_alarm is None:
                T_alarm = step
                alarm_source = f"attention_entropy = {H:.3f}"

    if step % 50 == 0:
        print(f"  step {step:>4}  loss={float(loss):.4f}")

print()
if T_nan is not None and T_alarm is not None:
    lead = T_nan - T_alarm
    print(f"  ✓ Diverged at step {T_nan}")
    print(f"  ✓ Alarm fired at step {T_alarm}  [{alarm_source}]")
    print(f"  ✓ Lead time: {lead} steps")
elif T_nan is not None and T_alarm is None:
    print(f"  ⚠ Diverged at step {T_nan} but no alarm fired — decrease alarm_threshold")
elif T_nan is None:
    print(f"  ⚠ Run converged (no NaN). Increase --lr to cross instability boundary")
    print(f"     Current T_alarm = {T_alarm}")
print()
print("NumPy reference (bench4): mean lead = 68.80 steps  (30/30 diverged, 30/30 alarmed)")
