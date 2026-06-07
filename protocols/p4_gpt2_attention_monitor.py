"""
Protocol P4: Inference-time attention monitoring on GPT-2 small
===============================================================
Uses the nonans.RuntimeMonitor to monitor attention health during
GPT-2 small inference on a synthetic prompt. This is the "famous-
model validation" referenced in the paper.

What it demonstrates
--------------------
  1. Correct library integration: wrap GPT-2's attention weights
     with RuntimeMonitor.check_layer().
  2. Calibration pass: 50 prompts from the validation distribution.
  3. Live inference: classify each layer/head on a test prompt.
  4. Optional injection: replace one head with a one-hot distribution
     to verify COLLAPSE is detected.

Usage
-----
    pip install transformers torch
    python protocols/run_gpt2_attention_monitor.py

    # Force collapse injection on layer 4 head 0:
    python protocols/run_gpt2_attention_monitor.py --inject_collapse 4 0

Dependencies: transformers ≥ 4.30, torch ≥ 1.12
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from nonans import RuntimeMonitor

parser = argparse.ArgumentParser()
parser.add_argument("--inject_collapse", nargs=2, type=int, default=None,
                    metavar=("LAYER", "HEAD"),
                    help="Inject a synthetic collapse into this (layer, head)")
parser.add_argument("--n_cal_prompts", type=int, default=50)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

try:
    from transformers import GPT2Model, GPT2Tokenizer
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    sys.exit(1)

print("Loading GPT-2 small…")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model     = GPT2Model.from_pretrained("gpt2",
                                       output_attentions=True,
                                       attn_implementation="eager")
model = model.to(args.device).eval()
print(f"  n_layers = {model.config.n_layer}  n_heads = {model.config.n_head}")
print()

monitor = RuntimeMonitor(
    collapse_threshold_abs = 0.05,
    sequence_length        = 1024,
    deviation_sigma        = 3.0,
)

CAL_TEXTS = [
    "The capital of France is Paris, which is known for",
    "In recent years, artificial intelligence has made significant",
    "The quick brown fox jumps over the lazy dog and then",
    "Modern transformer architectures rely on the self-attention mechanism",
    "Water boils at 100 degrees Celsius at standard atmospheric",
] * (args.n_cal_prompts // 5 + 1)

# ─── Calibration pass ─────────────────────────────────────────────────────────
print(f"Calibration pass  ({args.n_cal_prompts} prompts)…")
with torch.no_grad():
    for text in CAL_TEXTS[:args.n_cal_prompts]:
        inputs = tokenizer(text, return_tensors="pt").to(args.device)
        out    = model(**inputs)
        for layer_i, layer_attn in enumerate(out.attentions):
            # layer_attn: (B, n_heads, seq_len, seq_len)
            for head_i in range(layer_attn.shape[1]):
                attn_vec = layer_attn[0, head_i, -1, :]  # last query position
                monitor.update_calibration(attn_vec, layer=layer_i, head=head_i)

monitor.finalize_calibration()
n_calibrated = len(monitor._calibration)
print(f"  Calibrated {n_calibrated} (layer, head) pairs")
print()

# ─── Test inference ────────────────────────────────────────────────────────────
TEST_TEXT = ("The attention mechanism in transformers computes "
             "a weighted sum of values based on the similarity between "
             "query and key vectors, enabling the model to focus on "
             "different parts of the input.")

print(f"Test prompt: '{TEST_TEXT[:60]}…'")
inputs = tokenizer(TEST_TEXT, return_tensors="pt").to(args.device)
with torch.no_grad():
    out = model(**inputs)

attentions = list(out.attentions)

# Optional collapse injection
if args.inject_collapse is not None:
    inj_layer, inj_head = args.inject_collapse
    if inj_layer < len(attentions):
        n_seq = attentions[inj_layer].shape[-1]
        onehot = torch.zeros(n_seq, device=args.device)
        onehot[0] = 1.0
        # Replace that head's last-position attention
        attentions[inj_layer][0, inj_head, -1, :] = onehot
        print(f"\n[INJECTION] One-hot collapse injected at layer {inj_layer} head {inj_head}")

# ─── Classification ────────────────────────────────────────────────────────────
print("\nLayer-by-layer health report:")
print(f"  {'layer':>5}  {'head':>4}  {'entropy':>8}  {'state':<12}  {'conf':>5}")
print(f"  {'─' * 52}")

total_heads = 0
faults      = {"COLLAPSE": 0, "MAXIMUM": 0, "DEVIATION": 0}

for layer_i, layer_attn in enumerate(attentions):
    n_heads = layer_attn.shape[1]
    # Stack per-head last-query-position attention vectors
    attn_per_head = torch.stack(
        [layer_attn[0, h, -1, :] for h in range(n_heads)]
    )   # (n_heads, seq_len)

    results = monitor.check_layer(attn_per_head, layer=layer_i)
    for head_i, state, conf, H in sorted(results, key=lambda x: x[1].severity):
        total_heads += 1
        if state.name in faults:
            faults[state.name] += 1
        marker = " ◄ FAULT" if state.name in faults else ""
        print(f"  {layer_i:>5}  {head_i:>4}  {H:>8.4f}  "
              f"{state.icon} {state.name:<10}  {conf:>5.2f}{marker}")

print(f"\n  Total heads    : {total_heads}")
print(f"  COLLAPSE faults: {faults['COLLAPSE']}")
print(f"  MAXIMUM  faults: {faults['MAXIMUM']}")
print(f"  DEVIATION faults: {faults['DEVIATION']}")
print()
if args.inject_collapse is not None:
    expected = (faults["COLLAPSE"] > 0)
    print(f"  Injection detection: {'✓ DETECTED' if expected else '✗ MISSED'}")
