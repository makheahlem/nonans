# GPU Validation Plan

The benchmark protocols in `protocols/` are written, validated on CPU
where possible, and ready to run on accelerator hardware. This document
states exactly what each protocol measures, what it produces, what
hardware it needs, and what reviewers can check against the published
claims afterward.

The plan is sized to **~250 GPU-hours** total across the three
protocols. Each protocol's exit criteria are explicit, so partial
completion is interpretable.

---

## Protocol P2 — GPU overhead measurement

**Script:** `protocols/run_overhead_torch.py`
**What it measures:** entropy-computation overhead relative to the
attention forward pass, on accelerator hardware, across five model
configurations (heads × sequence length). The CPU counterpart is
already in the preprint as B3.

**Required hardware:** any modern CUDA device (T4, V100, A100, H100,
RTX 30/40-series all work; H100/A100 preferred for currency).
**Required compute:** under one hour wall-clock.
**Exit criterion:** JSON output with per-configuration overhead
percentages and complete hardware metadata block (GPU model, CUDA
version, cuDNN version, capability, memory). Reviewers can verify
the result reproduces by re-running.

**Expected outcome:** measured GPU overhead, to be reported in the
recompiled preprint *with* the hardware block. The CPU result of
15.39% (Intel Xeon @ 2.80 GHz) bounds the worst case; the GPU result
will sit below it due to bandwidth advantage on already-resident
tensors, but the exact number is what the protocol is designed to
determine — not a number to be asserted in advance.

**Failure modes documented:** if overhead exceeds 20% on GPU, the
"already-resident" cost model is wrong and the architecture needs
revision. The protocol returns a clear pass/fail signal against a
threshold the paper can defend.

---

## Protocol P3 — Real-training-run lead time

**Script:** `protocols/run_early_warning_torch.py`
**What it measures:** the structural precursor lead between an NRI
alarm fire and an actual `NaN`/`Inf` event in a real PyTorch
transformer training loop in FP16 with no AMP loss scaling (forced
divergence). The B4 figure (`68.8` steps, controlled protocol)
becomes a real-training validation point.

**Required hardware:** any GPU with ≥16 GB memory for the small
transformer configuration; ≥40 GB for the medium configuration.
**Required compute:** ~24 GPU-hours for the small configuration with
n=30 forced-divergence runs; ~96 GPU-hours for the medium.
**Exit criterion:** JSON output with per-run `T_alarm`, `T_NaN`, lead
time, plus aggregate statistics. Detection rate, false-alarm rate,
and mean lead time directly comparable to B4.

**Expected outcome:** a second benchmark table in the paper labeled
"real-training validation" supplementing the controlled-protocol
result. Honest expectation: real-training lead times will differ from
controlled-protocol numbers, possibly substantially. The protocol is
designed to *measure* this, not to confirm a predetermined answer.

---

## Protocol P4 — Attention monitoring on a real model

**Script:** `protocols/run_gpt2_attention_monitor.py`
**What it measures:** calibration and inference-time classification
on GPT-2 small (117M parameters), with optional collapse injection to
verify detection. Demonstrates the inference-time half of the
unified-identity claim on a real model rather than synthetic
distributions.

**Required hardware:** any GPU with ≥8 GB memory for GPT-2 small;
proportionally more for medium/large.
**Required compute:** ~12 GPU-hours for the small-model validation
including calibration sweep; substantially more if extended to
GPT-2 medium or larger.
**Exit criterion:** JSON output with per-layer per-head calibration
statistics, OOD detection rates on injected collapse, false-alarm
rate on in-distribution generation. Comparable to B6.

**Expected outcome:** the inference-side of the framework demonstrated
on a real production-class model, completing the regime-spanning
claim.

---

## Timeline assuming continuous compute allocation

| Phase | Duration | Output |
|---|---|---|
| Week 1 | P2 + paper update | GPU overhead row in B3; recompiled preprint |
| Weeks 2–3 | P3 small configuration | Real-training B4 supplement |
| Week 4 | P4 GPT-2 small | Inference validation |
| Week 5 | Aggregate, supplementary release | v0.5.0 release, updated DOI |

Total wall-clock: five weeks. Total compute: ~250 GPU-hours plus
overhead.

## What this plan does not cover

- **FSDP / multi-GPU distributed validation.** Out of scope for this
  allocation; would require multi-node access and is a separate ask.
- **Resolution-layer benchmarks.** The resolution layer is sealed; its
  evaluation belongs to a separate forthcoming paper.
- **Comparative evaluation against σReparam under matched conditions.**
  Worth doing but requires careful protocol design and is a
  follow-up rather than a validation step.

## Reporting back

All measurement scripts produce JSON output with hardware metadata.
The recompiled preprint will include the measured numbers with full
hardware disclosure, and the JSON outputs will be deposited alongside
the paper on Zenodo as supplementary materials, so reviewers can
verify each reported number by re-running the protocols on equivalent
hardware.
