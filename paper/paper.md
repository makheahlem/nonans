---
title: 'nonansNRI: Numerical Runtime Intelligence for Transformer Training and Inference'
tags:
  - Python
  - machine learning
  - transformers
  - numerical stability
  - runtime monitoring
  - observability
authors:
  - name: Ahlem Makhebi
    orcid: 0009-0007-7010-3282
    email: ahlemmakhebi@protonmail.com
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher, Germany
    index: 1
date: 10 June 2026
bibliography: paper.bib
---

# Summary

`nonansNRI` is a Python library for **runtime numerical-health monitoring** of
transformer models during training and inference. It instruments a model and
classifies, in real time, the structural state of its weight matrices and
attention distributions into discrete health categories — for example healthy,
rank-collapsed, or attention-concentrated — using lightweight information-theoretic
signals (Shannon entropy of singular-value and attention distributions) computed
on data already resident in memory.

The library exposes a small, dependency-light API. Its mathematical primitives
run on either NumPy arrays or PyTorch tensors; PyTorch is required only for the
training- and inference-time integration layers. A single entry point,
`nonans.wrap(model, optimizer)`, attaches monitoring to a training loop, and a
`RuntimeMonitor` class provides per-head attention classification at inference
time. The goal is to make numerical instability **observable before it becomes a
failure** — to surface a structural warning while a run is still recoverable,
rather than after it has produced a `NaN`.

# Statement of need

Transformer training is prone to numerical instability — loss spikes, gradient
explosions, rank collapse, and attention degeneration — that frequently
terminate in non-finite values (`NaN`/`Inf`) and a lost run [@dong2021attention].
In practice these failures are detected late, usually by monitoring the loss or
gradient norm, by which point the run is often unrecoverable and compute has been
wasted. Practitioners lack a lightweight, model-agnostic way to **observe the
structural precursors** of these failures as they develop.

`nonansNRI` addresses this gap. It provides a reusable instrumentation layer that
computes interpretable, bounded health scores from quantities a transformer
already produces (singular spectra of weights, attention probability
distributions), at negligible overhead, and classifies them into actionable
states. It is intended for:

- **ML researchers and engineers** training transformers who want an early,
  interpretable signal of structural degradation during a run;
- **Systems and infrastructure engineers** building training-stability tooling
  or observability dashboards who need a small, embeddable monitoring primitive;
- **Researchers studying training dynamics** who want a consistent, quantitative
  measure of attention and weight-spectrum health over time.

The library is deliberately scoped to **observation and classification**. It does
not modify training or attempt automated recovery; it produces typed health
records that downstream tooling can consume.

# State of the field

Existing approaches to training stability fall into a few categories. **Loss- and
gradient-norm monitors** (common in training frameworks) are simple and
widely used, but they react to instability rather than anticipate it — by the
time a gradient norm spikes, the run is often already diverging. **General-purpose
tensor debuggers and anomaly detectors** (e.g. PyTorch's `autograd` anomaly mode)
locate where a `NaN` was produced, but operate after the fact and are intended
for debugging rather than continuous, low-overhead monitoring. **Numerical-precision
analysis tools** focus on representation error rather than structural health of
attention or weight spectra.

`nonansNRI` differs in monitoring the **structural state** of the model —
spectral diffuseness of weights and entropy of attention — as a continuous,
bounded signal computed cheaply on already-resident data. This provides an
interpretable health measure that is independent of the loss value and available
throughout a run, complementing rather than replacing existing loss/gradient
monitors.

# Software design

The package is organized as a small set of focused modules, layered as
*observation → classification → a typed interface seam*:

- **`primitives`** — the mathematical core: Shannon entropy of a distribution,
  normalized attention entropy, a shape/rank score, and a randomized truncated
  SVD for efficient top-$k$ singular values. All functions accept NumPy arrays
  (the reference path) or PyTorch tensors; this is the only module needed for the
  core signals and it has no required dependency beyond NumPy.
- **`decision`** — classifiers that map structural signals to discrete health
  states: a training-time classifier over weight-spectrum signals and an
  inference-time attention classifier with optional per-head calibration.
- **`engine`** — a three-gate training-time monitoring engine that sequences
  cheap-to-expensive checks (norm deviation, then spectral analysis) so that the
  costlier singular-value computation runs only when a cheaper trigger fires
  (requires PyTorch).
- **`runtime`** — the inference-time `RuntimeMonitor`: per-head attention
  classification with a batched hot path and an optional calibration pass that
  learns a per-head nominal entropy band.
- **`telemetry`** — a telemetry bus and the resolver-interface protocol, defining
  how structured health records are emitted to any downstream consumer.
- **`protocol`** — the typed interface contracts (health enums, a structured
  `FaultContext` record, a fixed-capacity ring buffer) that form a stable seam
  between the open monitoring layer and a downstream consumer.
- **`bindings`** — the `nonans.wrap(model, optimizer)` integration entry point.
- **`bench`** — six self-contained, deterministic benchmarks runnable via
  `python -m nonans.bench`.
- **`protocols`** — runnable PyTorch validation scripts (P1–P5), including a
  real-transformer divergence demonstration.

Design choices: a minimal required dependency surface (NumPy only for the core,
so the primitives are testable without PyTorch), optional PyTorch for the
training/inference integration, bounded and interpretable scores in $[0, 1]$, a
staged-cost monitoring engine to keep per-step overhead low, and a stable typed
interface (`protocol`) so the monitoring layer can be embedded in larger systems
without coupling to internal implementation details. The library implements the
open observation-and-classification layer only; the typed `FaultContext` seam is
the documented boundary at which a separate resolution component could consume
its output.

# Quality control

The repository includes:

- A **pytest** test suite covering the mathematical primitives, the decision
  logic, and the typed protocol contracts.
- **Deterministic benchmarks** (fixed seed) runnable individually or together
  via `python -m nonans.bench all`, each writing machine-readable JSON output.
- A **five-minute reproducibility script** (`reproduce.sh`) that re-runs core
  benchmarks and asserts their headline numbers, so a reviewer can verify the
  central claims quickly.
- **Continuous-integration gates** that assert the locked benchmark numbers,
  preventing silent numerical regressions.

Installation is standard (`pip install -e .`, or `pip install -e ".[test]"` for
the test extras), and the primitives run without PyTorch so the core is testable
in a minimal environment.

# Usage and demonstration

Basic integration:

```python
import nonans

# Training-time instrumentation (requires PyTorch)
model = nonans.wrap(model, optimizer)

# Inference-time attention monitoring (requires PyTorch)
monitor = nonans.RuntimeMonitor(sequence_length=512)
monitor.finalize_calibration()
state, conf, H = monitor.classify_head(attn_weights, layer=0, head=0)

# Primitives work directly on NumPy or PyTorch
import numpy as np
sv = np.array([1.0, 0.8, 0.5, 0.3, 0.1], dtype=np.float32)
nonans.signal_score(sv)        # bounded spectral-health score in [0, 1]
```

As a **demonstration of functionality** (not a scientific result), the repository
includes a runnable script (`protocols/p5_real_divergence.py`) that trains a small
real transformer (a 2-layer, 4-head GPT-style model built on PyTorch's
`nn.MultiheadAttention`) until it reaches a non-finite loss, while logging the
monitor's attention-entropy signal at every step. In a representative run, the
signal declines as instability develops and crosses a warning threshold before the
run reaches `NaN`. This is included to show the instrumentation operates correctly
on a real model and integrates with a standard training loop; it is a usage example,
not a controlled empirical study, and the divergence is induced deliberately via
learning-rate escalation.

# AI usage disclosure

AI-based tools were used **only to assist with the writing and structuring** of
documentation and this manuscript (organizing sections, drafting prose, and
formatting). All research ideas, the underlying mathematics, the software design,
and every code fix and correction are the author's own work and were
independently verified by the author. No AI tool contributed novel methods,
results, or claims to the software.

# Availability

- **Source code:** <https://github.com/makheahlem/nonans>
- **Archived release (Zenodo):** <https://doi.org/10.5281/zenodo.20573423>
- **License:** Apache-2.0 (code). The accompanying preprint is CC BY 4.0.

# Acknowledgements

The author thanks the open-source scientific Python and PyTorch communities,
on whose tools this work builds.

# References
