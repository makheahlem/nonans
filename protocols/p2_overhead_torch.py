"""
Protocol P2: Entropy-monitoring overhead on real GPU attention
==============================================================
Measures the wall-clock fraction that entropy monitoring adds to the
attention forward pass on your hardware.

The NumPy reference (bench3) measured 10–25% on CPU, trending lower
for larger configs. GPU is expected to be lower because:
  (a) entropy is a single fused reduction over cache-resident softmax output
  (b) attention (QK^T + softmax) is FLOP-dense; entropy is not

Usage
-----
    python protocols/run_overhead_torch.py [--device cuda|cpu]

Reports: overhead_% per config, plus mean and median.
"""
import torch, time, argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nonans import attention_entropy_batched

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--n_iter", type=int, default=500)
args = parser.parse_args()
device = args.device
print(f"Device: {device}  (torch {torch.__version__})")
if device == "cuda":
    print(f"  {torch.cuda.get_device_name(0)}")
print()

def softmax_attn(Q, K, d_k):
    scores = (Q @ K) / (d_k ** 0.5)
    return torch.softmax(scores, dim=-1)

def time_fn(fn, *args, n_iter=200, warmup=20):
    for _ in range(warmup):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(n_iter):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter_ns() - t0) / n_iter / 1000.0   # µs

configs = [
    (8,  128, 64),
    (8,  256, 64),
    (16, 256, 64),
    (16, 512, 64),
    (32, 512, 64),
    (32, 1024, 64),
]

gen = torch.Generator(device="cpu")
gen.manual_seed(0)
overheads = []
print(f"  {'config':<12}  {'fwd µs':>10}  {'ent µs':>10}  {'overhead':>10}")
print(f"  {'─'*60}")
for n_heads, seq_len, d_k in configs:
    Q = torch.randn(n_heads, 1, d_k,       device=device)
    K = torch.randn(n_heads, d_k, seq_len, device=device)
    A = softmax_attn(Q, K, d_k)[:, 0, :]   # (n_heads, seq_len)

    fwd_us = time_fn(softmax_attn, Q, K, d_k, n_iter=args.n_iter)
    ent_us = time_fn(attention_entropy_batched, A,  n_iter=args.n_iter)
    pct    = 100.0 * ent_us / fwd_us
    overheads.append(pct)
    print(f"  {n_heads}h×{seq_len:<6}    {fwd_us:>10.2f}  {ent_us:>10.2f}  {pct:>9.2f}%")

import statistics
print(f"\n  mean overhead   : {statistics.mean(overheads):.2f}%")
print(f"  median overhead : {statistics.median(overheads):.2f}%")
print()
print("NumPy CPU reference (bench3): mean=15.85%  median=14.82%  (n=5 configs)")
print("Note: GPU numbers are typically 3–8× lower due to memory bandwidth advantage.")
