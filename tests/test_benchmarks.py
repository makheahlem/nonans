"""
Tests for nonans.bench — the authoritative validation gate.

Runs each benchmark and asserts the headline result against the value
locked in the preprint (seed 20260525). Any failure here means the
mathematics has drifted and the paper's claims no longer reproduce.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile

import pytest


# Module-level temp directory: created once, shared by all tests in this file.
# (Equivalent to pytest's tmp_path_factory but simpler and explicit.)
_FIGURES_DIR = tempfile.mkdtemp(prefix="nonans_test_figures_")


def _run_benchmark(name: str) -> dict:
    """Run a benchmark module in the shared temp dir and return its JSON result."""
    prev_cwd = os.getcwd()
    os.chdir(_FIGURES_DIR)
    try:
        mod_name = f"nonans.bench.{name}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        importlib.import_module(mod_name)
        # b1_unified_identity -> figures/bench1_unified_identity.json
        bench_num = name[1]
        json_basename = f"bench{bench_num}_{name[3:]}.json"
        json_path = os.path.join(_FIGURES_DIR, "figures", json_basename)
        with open(json_path) as f:
            return json.load(f)
    finally:
        os.chdir(prev_cwd)


# ─── B1: unified identity ────────────────────────────────────────────────────

def test_b1_unified_identity_locked():
    """B1: max pointwise error of the unified-identity claim.
    Locked value (seed 20260525): 3.1834430658239654e-07."""
    result = _run_benchmark("b1_unified_identity")
    assert result["verdict"] == "VERIFIED"
    assert result["n_samples"] == 10000
    assert result["max_error"] == pytest.approx(3.1834430658239654e-07, abs=1e-10), \
        f"B1 drifted: got {result['max_error']!r}, expected 3.1834430658239654e-07"


# ─── B2: detection on three canonical fault classes ──────────────────────────

def test_b2_detection_locked():
    """B2: 100% classification accuracy on three canonical fault classes."""
    result = _run_benchmark("b2_detection")
    acc = result["accuracy_per_class"]
    for cls in ("HEALTHY", "COLLAPSE", "MAXIMUM"):
        assert acc[cls] == 1.0, f"B2 {cls} accuracy drifted: {acc[cls]}"


# ─── B3: CPU overhead (hardware-dependent — only check structure) ────────────

def test_b3_overhead_structure():
    """B3: CPU overhead is hardware-dependent. We assert the result is
    in a plausible range, not a specific number, and that all five
    configurations were measured."""
    result = _run_benchmark("b3_overhead")
    assert "configurations" in result
    assert len(result["configurations"]) == 5
    mean_overhead = result.get("mean_overhead_%") or result.get("mean_overhead_pct")
    assert mean_overhead is not None, "B3 mean overhead key missing"
    # Wide envelope: from ~5% on fast CPUs to ~50% on slow ones.
    assert 1.0 <= mean_overhead <= 60.0, f"B3 mean overhead implausible: {mean_overhead}"


# ─── B4: early-warning lead time ─────────────────────────────────────────────

def test_b4_early_warning_locked():
    """B4: detection rate, alarm rate, and mean lead time on controlled protocol."""
    result = _run_benchmark("b4_early_warning")
    assert result["n_runs_total"] == 30, "B4 expected 30 total runs"
    assert result["n_diverged"] == 30, "B4 expected all 30 runs to diverge"
    assert result["n_with_alarm"] == 30, "B4 expected alarm in all 30 runs"
    lead = result["lead_time_overall"]
    # Mean lead time is deterministic at this seed; locked value 68.8 (controlled protocol)
    assert lead["mean"] == pytest.approx(68.8, abs=0.5), \
        f"B4 mean lead drifted: {lead['mean']}"


# ─── B5: comparative detector evaluation ─────────────────────────────────────

def test_b5_comparison_locked():
    """B5: signal-score lead vs gradient-norm at equal false-alarm rate."""
    result = _run_benchmark("b5_comparison")
    # Keys are the detector names with format strings; iterate to find ours.
    signal_key = next(k for k in result if "signal_score" in k.lower())
    gradient_key = next(k for k in result if "gradient-norm" in k.lower())
    signal_lead = result[signal_key].get("mean_lead") or result[signal_key].get("mean_lead_steps")
    gradient_lead = result[gradient_key].get("mean_lead") or result[gradient_key].get("mean_lead_steps")
    assert signal_lead == pytest.approx(84.5, abs=0.5), f"B5 signal_score lead drifted: {signal_lead}"
    assert gradient_lead == pytest.approx(54.9, abs=0.5), f"B5 gradient_norm lead drifted: {gradient_lead}"
    # The headline claim: signal_score leads by 29.6 steps
    assert (signal_lead - gradient_lead) == pytest.approx(29.6, abs=1.0)


# ─── B6: calibration effect on false-alarm rate ──────────────────────────────

def test_b6_calibration_locked():
    """B6: calibration pass enables DEVIATION detection.

    Output schema: ``results.{uncalibrated,calibrated}.{in-distribution,
    ood-collapse,ood-max}.accuracy``. The test verifies that both modes
    achieve 100% OOD detection (collapse and maximum) and that
    calibration produces a calibration record with the expected fields.
    """
    result = _run_benchmark("b6_calibration")
    res = result["results"]
    cal = result["calibration"]
    # Calibration record must contain the four nominal-distribution stats
    for key in ("mean", "std", "p5", "p95"):
        assert key in cal, f"B6 calibration missing key: {key}"
    # Both modes must achieve 100% OOD detection on the two OOD types
    for mode in ("uncalibrated", "calibrated"):
        for ood in ("ood-collapse", "ood-max"):
            acc = res[mode][ood]["accuracy"]
            assert acc == 1.0, f"B6 {mode} {ood} drifted: {acc}"
