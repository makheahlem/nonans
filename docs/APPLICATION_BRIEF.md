# Application Brief

**Project:** **nonansNRI** — Numerical Runtime Intelligence (NRI)
**Commercial entity:** nonans
**Applicant:** Ahlem Makhebi, Independent Researcher (Germany) · ORCID 0009-0007-7010-3282 · ahlemmakhebi@protonmail.com
**Repository:** https://github.com/makheahlem/nonans
**Preprint:** [Zenodo DOI on assignment]
**Status:** v0.4.0 released. Six benchmarks reproducible bit-for-bit on
IEEE-754 hardware. Self-validating PyTorch protocols ready for GPU
execution.

## The contribution

NRI is a runtime instrumentation layer for the real-time classification
of legitimate numerical events — non-finite values that emerge from the
computation itself (overflow, structural rank collapse, attention
concentration), distinct from the bit-flip soft errors targeted by
algorithm-based fault tolerance (Huang & Abraham 1984 onward) and from
static floating-point precision analysis (Liu et al., TOSEM 2025).

The published contribution is the open observability and classification
layer: a unified-identity entropy interface across training and
inference, verified to `3.18×10⁻⁷` (B1); a propagation-regime
classifier discriminating transient from persistent faults (B2);
structural precursor lead of `84.5` steps over gradient-norm monitoring
at equal false-alarm rate (B5). A proprietary resolution layer at the
runtime boundary is maintained in a separate private codebase and is the
subject of separate forthcoming work.

## Why this work is at this stage

Everything that can be done without accelerator hardware has been done.
The mathematical claims are verified on CPU against an independently
implemented NumPy reference. The PyTorch protocols are written and
self-validating. The preprint is positioned against the closest prior
art (σReparam, spectral-dynamics work, SentryCam, the SVD-entropy
lineage in HPC/nuclear monitoring) with the contribution narrowed to
what the benchmarks support.

## What I cannot do alone, and what the resource unlocks

Three measurements are missing from the preprint and are explicitly
listed as outstanding. Each is matched to a protocol; each requires
hardware I do not have access to as an independent researcher.

1. **GPU overhead measurement (Protocol P2).**
   The CPU overhead figure (`15.39%` on Intel Xeon @ 2.80 GHz) is in
   the preprint. The GPU figure is not, and on already-resident tensors
   the bandwidth advantage of on-device computation is expected to
   change the result substantially. *Required:* any modern CUDA device.
   *Output:* one verified table row, recompiled preprint.
   *Time:* under one hour wall-clock.

2. **Real-training-run lead time (Protocol P3).**
   The `68.8`-step lead in B4 is on a controlled instability protocol,
   not a real training run. Validating on actual GPT-2-class training
   in FP16 turns the precursor claim from "verified on synthetic
   structural perturbation" into "verified on real divergence."
   *Required:* GPU for transformer training, ~24 hours of compute.
   *Output:* second benchmark table in the paper, supplementary
   release.

3. **Large-model attention monitoring (Protocol P4).**
   Calibration and inference-time classification on GPT-2 small (117M);
   extension to GPT-2 medium/large with adequate compute.
   *Required:* GPU memory sufficient for the chosen model class.
   *Output:* third validation result, demonstration of
   regime-spanning operation claimed in the paper.

All three protocols are checked-in, self-validating, and assert their
results against locked reference tolerances. They produce JSON output
suitable for direct inclusion in a recompiled paper.

## Independent-researcher accountability

I am unaffiliated. The paper is CC BY 4.0; the code is Apache-2.0; the
benchmarks are deterministic and seed-locked (`20260525`). A reviewer
can clone the repository and reproduce the headline number in under
five minutes on any laptop. The ORCID iD links the preprint, the
repository, and this application. Citation is requested under academic
norm and supported by a `CITATION.cff` file at the repository root.

## Specific ask

*For GPU/compute-credit programs:* ~250 GPU-hours on H100/A100-class
hardware to execute Protocols P2, P3, P4 and recompile the preprint
with measured results. No additional financial support requested.

*For independent-researcher fellowships:* the above compute, plus
modest stipend support for the next twelve months covering Paper 2 (on
the residual-continuation direction in inference and real-time
settings) and travel to one systems venue for in-person presentation.

For all enquiries: ahlemmakhebi@protonmail.com.
