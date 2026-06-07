"""
nonans.bench
============

Six deterministic benchmarks (seed 20260525) validating the locked
numbers in the preprint. Run individually or all at once:

    python -m nonans.bench b1     # unified identity
    python -m nonans.bench all    # run all six in order

Outputs JSON + PNG to ``figures/`` in the current working directory.
"""
from ._harness import SEED, hardware_metadata, output_dir

__all__ = ["SEED", "hardware_metadata", "output_dir"]
