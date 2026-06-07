"""
Entry point for ``python -m nonans.bench``.
"""
from __future__ import annotations

import importlib
import sys

BENCHMARKS = [
    "b1_unified_identity",
    "b2_detection",
    "b3_overhead",
    "b4_early_warning",
    "b5_comparison",
    "b6_calibration",
]


def _run(name: str) -> None:
    full = f"nonans.bench.{name}"
    # Importing the module runs it (the benchmark scripts execute at import)
    importlib.import_module(full)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m nonans.bench {b1|b2|b3|b4|b5|b6|all}", file=sys.stderr)
        return 2
    target = argv[1].lower()
    if target == "all":
        for name in BENCHMARKS:
            print(f"\n===== {name} =====")
            _run(name)
        return 0
    matches = [n for n in BENCHMARKS if n.startswith(target)]
    if len(matches) != 1:
        print(f"unknown benchmark: {target!r}; available: {BENCHMARKS + ['all']}", file=sys.stderr)
        return 2
    _run(matches[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
