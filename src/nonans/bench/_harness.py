"""
nonans.bench._harness
=====================

Shared infrastructure for the six benchmark scripts: deterministic
seeding, output directory resolution, hardware metadata capture, and a
small JSON-writer that emits a uniform record per benchmark.
"""
from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict


SEED = 20260525


def output_dir() -> str:
    """Resolve the project-root ``figures/`` directory and ensure it exists.

    Output is written to the current working directory's ``figures/``
    folder so reviewers running ``python -m nonans.bench b1`` from the
    repo root find results in a predictable place.
    """
    out = os.path.join(os.getcwd(), "figures")
    os.makedirs(out, exist_ok=True)
    return out


def hardware_metadata() -> Dict[str, Any]:
    """Capture host hardware metadata for honest timing benchmarks.

    Timing results are hardware-dependent; this block must be reported
    alongside any latency or throughput number.
    """
    import numpy as np  # local import: don't impose at module load
    md: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cpu": platform.processor() or "unknown",
        "machine": platform.machine(),
    }
    try:
        import torch  # noqa: F401
        md["torch"] = torch.__version__
        md["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            md["gpu_name"] = torch.cuda.get_device_name(0)
            md["cuda_version"] = torch.version.cuda
            md["gpu_capability"] = f"{props.major}.{props.minor}"
            md["gpu_total_memory_GB"] = round(props.total_memory / 1e9, 2)
    except ImportError:
        md["torch"] = None
    return md
