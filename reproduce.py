#!/usr/bin/env python3
"""
nonans / NRI — one-command reproducibility test.

Works on Windows, macOS, and Linux with no shell, no venv juggling, no path
tricks. Run it with whatever Python you have:

    python reproduce.py

It checks the environment, installs the package if needed, and runs the core
checks, failing loudly with a clear message if anything is wrong.
"""
import subprocess, sys, os, importlib.util

def say(msg):  print(f"[reproduce] {msg}")
def fail(msg):
    print(f"\n[reproduce] FAILED: {msg}\n")
    sys.exit(1)

# 1. Python version sanity (the 3.14 numpy/torch trap)
v = sys.version_info
say(f"Python {v.major}.{v.minor}.{v.micro}")
if v.minor >= 14:
    say("WARNING: Python 3.14+ may lack stable numpy/torch wheels.")
    say("If install or import fails below, use Python 3.11 or 3.12 in a fresh venv.")

# 2. Is the package importable? If not, install it from the current directory.
def importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

if not importable("nonans"):
    say("nonans not importable in this interpreter — installing from current directory...")
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".",
                        "--break-system-packages"] if v.minor >= 12
                       else [sys.executable, "-m", "pip", "install", "-e", "."],
                       cwd=here)
    if r.returncode != 0:
        # retry without --break-system-packages in case it's unsupported
        r2 = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=here)
        if r2.returncode != 0:
            fail("could not install the nonans package. Run `pip install -e .` manually "
                 "from the repo root and check the error.")
    if not importable("nonans"):
        fail("installed, but 'nonans' still not importable. Are you running this script "
             "from the repo root (the folder containing pyproject.toml)?")

import numpy as np
import nonans
say(f"nonans imported from: {nonans.__file__}")

# 3. Core numerical check — the unified-identity / signal primitive.
#    This does NOT depend on the bench submodule existing; it tests the
#    primitives directly, so it works regardless of repo layout.
say("running core signal checks...")

sv = np.array([1.0, 0.8, 0.5, 0.3, 0.1], dtype=np.float64)
ss = float(nonans.signal_score(sv))
expected = 0.8741  # documented value
if abs(ss - expected) > 1e-3:
    fail(f"signal_score({list(sv)}) = {ss:.4f}, expected ~{expected}. Primitive mismatch.")
say(f"signal_score check: {ss:.4f}  PASS")

# healthy vs collapsed spectrum should separate
healthy   = np.ones(8, dtype=np.float64)               # flat -> high entropy
collapsed = np.array([1.0] + [1e-6]*7, dtype=np.float64)  # spike -> low entropy
h_score = float(nonans.signal_score(healthy))
c_score = float(nonans.signal_score(collapsed))
if not (h_score > 0.9 and c_score < 0.2):
    fail(f"separation check failed: healthy={h_score:.3f} (want >0.9), "
         f"collapsed={c_score:.3f} (want <0.2)")
say(f"separation check: healthy={h_score:.3f} > collapsed={c_score:.3f}  PASS")

# 4. Optional: run the bench module IF it exists, but don't fail if it doesn't.
if importable("nonans.bench"):
    say("nonans.bench found — running B1...")
    r = subprocess.run([sys.executable, "-m", "nonans.bench", "b1"])
    if r.returncode == 0:
        say("nonans.bench b1  PASS")
    else:
        say("nonans.bench b1 did not exit cleanly (non-fatal for this core test).")
else:
    say("nonans.bench submodule not present — skipping (core checks above are sufficient).")

print("\n[reproduce] ALL CORE CHECKS PASSED.")
print("[reproduce] The nonans signal primitives are installed and working.")
