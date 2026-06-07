<div align="center">

<a href="https://nonans.com"><img src="https://img.shields.io/badge/nonans-NRI-7c6bff?style=for-the-badge&labelColor=0d0d11" alt="nonans NRI"></a>

# nonansNRI

### See the divergence before the NaN.

**Numerical Runtime Intelligence** — open, reproducible runtime instrumentation
for transformer numerical health.

[![Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-2ec87a)](LICENSE)
[![Paper CC BY 4.0](https://img.shields.io/badge/paper-CC%20BY%204.0-a599ff)](https://doi.org/10.5281/zenodo.20573423)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20573423-7c6bff)](https://doi.org/10.5281/zenodo.20573423)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--7010--3282-a6ce39)](https://orcid.org/0009-0007-7010-3282)
[![Benchmarks](https://img.shields.io/badge/benchmarks-6%2F6%20reproduce-2ec87a)](#benchmarks)
[![Version](https://img.shields.io/badge/version-0.4.0-9898ac)](#)

[**Website**](https://nonans.com) · [**Researcher**](https://makhebiahlem.nonans.com) · [**Reel demo**](demo.html) · [**Pitch deck**](deck.html) · [**Preprint**](https://doi.org/10.5281/zenodo.20573423) · [**Cite**](#cite)

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
> Independent research project. Reference implementation: `nonans` v0.4.0.
> Paper: CC BY 4.0 · Code: Apache-2.0 (open) · Resolver: proprietary

---

## Five-minute reproducibility test

```bash
git clone https://github.com/makheahlem/nonans.git
cd nonans
pip install -e .          # numpy is the only required dep
./reproduce.sh            # runs B1 and B2 with locked-number assertions
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

## What is not yet validated

- **GPU overhead** — measured on CPU only; Protocol P2
  (`protocols/p2_overhead_torch.py`) runs on accelerator hardware and
  captures full hardware metadata when executed.
- **Real-model lead times** — Protocol P3
  (`protocols/p3_early_warning_torch.py`) runs a real PyTorch transformer
  divergence in FP16; not yet executed by the author at scale.
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
protocols/               PyTorch protocols (P1-P4) for GPU validation
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
for academic reuse. Full policy with copy-paste blocks: [`ATTRIBUTION.md`](ATTRIBUTION.md).

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

## Web, media & metadata

| Surface | Where |
|---|---|
| Project site | [nonans.com](https://nonans.com) — `index.html` |
| Researcher site | [makhebiahlem.nonans.com](https://makhebiahlem.nonans.com) — `researcher.html` |
| Reel demo | `demo.html` — 60-second auto-playing walkthrough |
| Pitch deck | `deck.html` — keyboard-navigable research/partnership deck |
| Policies | `policies.html` — copyright, attribution, privacy, terms |

Metadata is optimized for discoverability, attribution retention, and
long-term citation propagation. Strategy and rationale: [`METADATA.md`](METADATA.md).
Implementing files:

- [`CITATION.cff`](CITATION.cff) — drives GitHub "Cite this repository"; software + paper citation, version lineage, ORCID.
- [`codemeta.json`](codemeta.json) — schema.org/SoftwareSourceCode for CodeMeta-aware indexers and Software Heritage.
- [`zenodo.json`](zenodo.json) — DataCite deposit metadata: tiered keywords, subjects, related identifiers, version lineage.
- [`NOTICE`](NOTICE) / [`ATTRIBUTION.md`](ATTRIBUTION.md) — the binding attribution mechanism and full policy.

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

<sub>© 2026 nonans / Ahlem Makhebi. Website content and brand: all rights reserved. Code: Apache-2.0. Paper: CC BY 4.0. See [`policies.html`](policies.html).</sub>
