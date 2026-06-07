# nonansNRI

**Numerical Runtime Intelligence (NRI)** — a runtime instrumentation
layer for the real-time classification of legitimate numerical events
(overflow, structural rank collapse, attention concentration) in
transformer training and inference.

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

## Licensing

- **Paper:** CC BY 4.0. Reuse requires attribution per the license terms.
- **Code (this repository):** Apache License 2.0. Redistribution
  requires retention of copyright, patent, trademark, and attribution
  notices; marking of modified files; and inclusion of the license.
  Apache 2.0 includes an explicit patent grant.
- **Resolution layer:** proprietary, maintained in a separate private
  codebase by **nonans** (the commercial entity associated with this
  project), not in this repository.

Citation is requested as a matter of academic norm; see `CITATION.cff`
and the recommended-citation block in the preprint.

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

Makhebi Ahlem · Independent Researcher (Germany) · ORCID 0009-0007-7010-3282 · ahlemmakhebi@protonmail.com
