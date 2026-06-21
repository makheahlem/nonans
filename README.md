<div align="center">

<a href="https://nonans.com"><img src="https://img.shields.io/badge/nonans-NRI-7c6bff?style=for-the-badge&labelColor=0d0d11" alt="nonans NRI"></a>

# nonansNRI

### See the divergence before the NaN.

**Numerical Runtime Intelligence** — open, reproducible runtime instrumentation
for transformer numerical health.

[![Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-2ec87a)](LICENSE)
[![Paper CC BY 4.0](https://img.shields.io/badge/paper-CC%20BY%204.0-a599ff)](https://doi.org/10.5281/zenodo.20573423)
[![Preprint II](https://img.shields.io/badge/preprint%20II-10.5281%2Fzenodo.20773398-7c6bff)](https://doi.org/10.5281/zenodo.20773398)
[![Preprint I](https://img.shields.io/badge/preprint%20I-10.5281%2Fzenodo.20573423-9898ac)](https://doi.org/10.5281/zenodo.20573423)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--7010--3282-a6ce39)](https://orcid.org/0009-0007-7010-3282)
[![Benchmarks](https://img.shields.io/badge/benchmarks-6%2F6%20reproduce-2ec87a)](#benchmarks)
[![Real-model](https://img.shields.io/badge/real--model-203%20step%20lead-7c6bff)](#real-model-validation)
[![Version](https://img.shields.io/badge/version-0.5.0-9898ac)](#)

[**Website**](https://nonans.com) · [**Researcher**](https://makhebiahlem.nonans.com) · [**Preprint II**](https://doi.org/10.5281/zenodo.20773398) · [**Preprint I**](https://doi.org/10.5281/zenodo.20573423) · [**Cite**](#cite)

</div>

---

**Numerical Runtime Intelligence (NRI)** is a runtime-instrumentation
**framework** for the real-time classification of legitimate numerical
events (overflow, structural rank collapse, attention concentration) in
transformer training and inference. This repository is its open
**reference implementation** (the Python package `nonans`).

The published contribution is the **observability and classification
layer**. A proprietary resolution layer at the runtime boundary is
maintained in a separate private codebase; its method is the subject
of separate forthcoming work.

> Project: **nonansNRI** · Commercial entity: **nonans**
> Independent research project. Reference implementation: `nonans` v0.5.0.
> Paper: CC BY 4.0 · Code: Apache-2.0 (open) · Resolver: proprietary

---

## Five-minute reproducibility test

```bash
git clone https://github.com/makheahlem/nonans.git
cd nonans
pip install -e .          # numpy is the only required dep
python reproduce.py       # cross-platform: installs if needed, runs core checks
```

Expected output (deterministic, seed `20260525`, IEEE-754):

```
B1 unified identity: max_error = 3.1834430658239654e-07   PASS
B2 detection: 100% on HEALTHY, COLLAPSE, MAXIMUM           PASS
```

If a single number fails, the rest of this work has nothing to stand on,
and a reviewer should stop here.

## Full test suite

```bash
pip install -e ".[test]"
pytest
```

The suite runs **56 tests** (with torch installed; 54 without): math
primitives, decision logic, the protocol contract, and the
authoritative benchmark gates that assert every locked headline number
from the preprint.

## Benchmarks

Run all six benchmarks; outputs go to `figures/{json,png}`:

```bash
python -m nonans.bench all
# or one at a time
python -m nonans.bench b1
```

| Benchmark | Claim | Locked result |
|---|---|---|
| B1 | Unified identity: same Shannon functional, two regimes | max error `3.18×10⁻⁷`, N=10,000 |
| B2 | Three canonical fault classes | 100% on HEALTHY/COLLAPSE/MAXIMUM |
| B3 | CPU entropy overhead vs attention forward | `~15%` mean (hardware-dependent) |
| B4 | Structural precursor lead before NaN | `68.8` steps mean, `30/30`, `0%` FA |
| B5 | Signal score vs gradient-norm at equal FA | `84.5` vs `54.9` (`29.6`-step advantage) |
| B6 | Calibration enables DEVIATION detection | OOD detection `100%` both modes |

## Real-model validation

Beyond the synthetic benchmarks, the precursor signal is validated on a **real
transformer trained to numerical divergence**: a 2-layer, 4-head GPT-style model
(~109K parameters, real `nn.MultiheadAttention`). Training is driven unstable by
learning-rate escalation until the loss reaches a non-finite value (NaN). The
mean attention entropy declines monotonically as instability builds and crosses
the warning threshold **203 steps before** the NaN.

| Model | Divergence | Warning (entropy < 0.6) | Lead time |
|---|---|---|---|
| 2-layer GPT, 4 heads (~109K params) | NaN @ step 351 | step 148 | **203 steps** |

```
attn_H 0.99 ●●●●●●●●●●●●●●●●●●●●●               healthy training
       0.60 ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ WARN(148) ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄  threshold
            ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● │ NaN(351)
            0          100      148      200       351  step
                                 └──── 203 steps ────┘
   loss:  0.7 ──► 185 (at warning) ──► 1.2e12 ──► NaN
```

```bash
python protocols/p5_real_divergence.py        # ~2 min on CPU, no GPU required
```

> Divergence here is induced by learning-rate escalation in fp32 (CPU); this
> demonstrates the precursor **mechanism** (attention entropy collapses before
> numerical failure). A hardware-faithful **FP16-underflow** divergence on GPU
> remains documented future work (see below and `docs/GPU_VALIDATION_PLAN.md`).

## What's new in v0.5.0 — silent structural degradation

v0.5.0 adds a multi-architecture failure-trace dataset and held-out-calibrated
validation, behind a second preprint that extends the foundational work.

**Central finding.** On real transformer architectures (causal GPT-style and
bidirectional ViT-style), NRI's spectral signal registers structural degradation
while loss and gradient norms remain within their healthy range — a class of
degradation that standard loss/gradient monitoring cannot, by construction,
observe. The result is held-out-calibrated with **0 false alarms** across both
architectures (dead-unit injection: 6/6 GPT, 5/5 ViT; rank-collapse: 3/6 GPT,
6/6 ViT; attention concentration is a characterized blind spot, 0/6 both).

**Robustness finding.** Of four induced failure modes on real transformers, only
unbounded weight-norm growth reliably reaches a non-finite value (12/12 across
both architectures); injected rank collapse, attention concentration, and unit
death are largely absorbed by the architecture's normalization and residual
structure — reframing where training instability actually originates.

All results are reproducible on CPU:

```bash
pip install -e .
python datasets/nri_failure_dataset_generator.py     # multi-arch failure dataset
python datasets/nri_dataset_validation.py            # held-out-calibrated validation
python datasets/silent_degradation_validated.py      # silent-degradation test
```

Second preprint: *Numerical Runtime Intelligence: Observing Structural
Degradation in Transformer Training Beyond Loss and Gradient Signals* —
[doi.org/10.5281/zenodo.20773398](https://doi.org/10.5281/zenodo.20773398).

## What is not yet validated

- **GPU overhead** — measured on CPU only; Protocol P2
  (`protocols/p2_overhead_torch.py`) runs on accelerator hardware and
  captures full hardware metadata when executed.
- **Real-model lead time (FP16 path)** — a real-transformer lead time of
  **203 steps** is demonstrated via learning-rate escalation
  (`protocols/p5_real_divergence.py`, above). The **FP16-underflow** divergence
  path (Protocol P3, `protocols/p3_early_warning_torch.py`) and larger-scale
  runs are not yet executed by the author.
- **FSDP / model-parallel settings** — compatibility not tested.

These gaps are documented in `paper/preprint.pdf` §6 and motivate the
`docs/GPU_VALIDATION_PLAN.md` included with this repository.

## Repository layout

```
src/nonans/              Installable package (Apache-2.0)
  primitives.py          Shannon entropy, randomized SVD; numpy primary, torch optional
  decision.py            Ghost/real classifier + runtime attention classifier
  engine.py              Three-gate monitoring engine (requires torch)
  runtime.py             Inference-time RuntimeMonitor (requires torch)
  telemetry.py           TelemetryBus, ResolverProtocol (the resolver interface)
  protocol.py            FaultContext, Health/RuntimeHealth enums, RingBuffer
  bindings.py            nonans.wrap(model, optimizer) entry point
  _torch_bindings.py     Small torch-only surface: grad + Adam state access
  bench/                 Six benchmarks runnable via `python -m nonans.bench`
datasets/                Multi-architecture failure dataset + validation (v0.5.0)
  nri_failure_dataset_generator.py   bare-matrix failure traces (numpy, CPU)
  nri_failure_dataset_torch.py       real GPT/ViT failure traces (torch)
  nri_dataset_validation.py          held-out-calibrated cross-arch validation
  silent_degradation_validated.py    silent-degradation test (the v0.5.0 finding)
protocols/               PyTorch protocols for validation
  p1..p4_*.py            unified-identity, overhead, FP16 early-warning, GPT-2 monitor
  p5_real_divergence.py  real-transformer divergence: 203-step lead before NaN (CPU/GPU)
tests/                   pytest suite; CI-gate on the locked benchmark numbers
paper/                   preprint.pdf + LaTeX source
docs/                    Application brief, GPU validation plan
```

## Quick integration

```python
import nonans

# Training-time (requires PyTorch)
model = nonans.wrap(model, optimizer)

# Inference-time (requires PyTorch)
monitor = nonans.RuntimeMonitor(sequence_length=512)
monitor.finalize_calibration()
state, conf, H = monitor.classify_head(attn_weights, layer=0, head=0)

# Math primitives work with numpy or torch tensors
sv = np.array([1.0, 0.8, 0.5, 0.3, 0.1], dtype=np.float32)
print(nonans.signal_score(sv))        # 0.8741
print(nonans.shape_score(sv))         # 1.0
print(nonans.attention_entropy(sv))   # 1.4612
```

## Licensing & attribution

Attribution is **mandatory** for the code and the paper, and **requested**
for academic reuse. The binding mechanism is the [`NOTICE`](NOTICE) file together with the [`LICENSE`](LICENSE).

- **Code (this repository):** Apache License 2.0. Redistribution requires
  retention of copyright, patent, trademark, and attribution notices;
  marking of modified files; inclusion of the license; **and a readable
  copy of the [`NOTICE`](NOTICE) file** (Apache-2.0 §4(d)) — this is the
  binding attribution mechanism. Apache 2.0 includes an explicit patent grant.
- **Paper:** CC BY 4.0. Any reuse or adaptation must credit the author,
  link the source, state the license, and indicate changes.
- **Resolution layer:** proprietary, maintained in a separate private
  codebase by **nonans** (the commercial entity associated with this
  project), not in this repository. Use-time attribution obligations for
  the resolver are set by its separate commercial license.

Citation is requested as a matter of academic norm; see `CITATION.cff` and
the recommended-citation block in the preprint.

## Links

- Project site: [nonans.com](https://nonans.com)
- Researcher: [makhebiahlem.nonans.com](https://makhebiahlem.nonans.com)
- Preprint: [doi.org/10.5281/zenodo.20573423](https://doi.org/10.5281/zenodo.20573423)
- Citation metadata: [`CITATION.cff`](CITATION.cff) (drives GitHub "Cite this repository")

Canonical citation name **Makhebi, Ahlem**, anchored by ORCID. Cite the
**concept DOI** for "any version" and the **version DOI**
`10.5281/zenodo.20573423` for this exact release.

## Cite

```bibtex
@article{makhebi2026nri,
  title  = {Numerical Runtime Intelligence: A Runtime Instrumentation
            Layer for Real-Time Classification of Legitimate Numerical
            Events in Transformer Training and Inference},
  author = {Makhebi, Ahlem},
  year   = {2026},
  doi    = {10.5281/zenodo.20573423},
  url    = {https://github.com/makheahlem/nonans}
}
```

## Author

**Makhebi, Ahlem** · Independent Researcher (Germany) · ORCID [0009-0007-7010-3282](https://orcid.org/0009-0007-7010-3282) · [makhebiahlem.nonans.com](https://makhebiahlem.nonans.com) · [LinkedIn](https://www.linkedin.com/company/nonans) · ahlem.makhebi@nonans.com

---

<sub>© 2026 nonans / Ahlem Makhebi. Website content and brand: all rights reserved. Code: Apache-2.0. Paper: CC BY 4.0.</sub>
