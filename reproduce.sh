#!/usr/bin/env bash
# reproduce.sh — five-minute reviewer reproducibility test.
#
# Runs benchmarks B1 and B2 from the installed nonans package and
# asserts each result against the value locked in the preprint
# (seed 20260525). All assertions are checked from Python.
#
# Exit code 0 = both PASSED. Non-zero = at least one FAILED.

set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$here"

# Run benchmarks from a temporary directory so figures/ doesn't pollute repo
work="$(mktemp -d -t nonans_reproduce_XXXXXX)"
trap 'rm -rf "$work"' EXIT
cd "$work"

echo "=== nonans reproducibility test ==="
echo "seed: 20260525"
echo "work dir: $work"
echo

# Prefer the installed package; fall back to the in-repo src/ if not installed
export PYTHONPATH="${here}/src:${PYTHONPATH:-}"

# B1
python3 -m nonans.bench b1 > /dev/null
python3 - <<'PY'
import json, sys
d = json.load(open("figures/bench1_unified_identity.json"))
expected = 3.1834430658239654e-07
got = d["max_error"]
ok = abs(got - expected) < 1e-10 and d["verdict"] == "VERIFIED"
print(f"  B1 unified identity: max_error = {got:.16e}")
print(f"    expected:                     {expected:.16e}")
print(f"    verdict:                      {d['verdict']}")
print(f"    -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY

echo

# B2
python3 -m nonans.bench b2 > /dev/null
python3 - <<'PY'
import json, sys
d = json.load(open("figures/bench2_detection.json"))
acc = d["accuracy_per_class"]
ok = all(acc[c] == 1.0 for c in ("HEALTHY", "COLLAPSE", "MAXIMUM"))
print(f"  B2 detection: per-class accuracy = {acc}")
print(f"    -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY

echo
echo "=== ALL PASSED ==="
echo "Headline mathematical claim reproduced:"
echo "  signal_score(sv) == attention_entropy_normalized(sv/sum(sv))"
echo "  max error 3.1834430658239654e-07 over N=10,000 (seed 20260525)"
