"""
nonans.protocol
===============

Public interface contracts: ``FaultContext``, ``Health`` and
``RuntimeHealth`` enums, ``RingBuffer``.

This module defines the typed seam between the open layers
(observability + classification) and the proprietary resolution layer.
The Resolution Layer consumes ``FaultContext`` records and returns
defined values; its implementation is outside this repository and is
the subject of separate forthcoming work.

These names and field layouts are the long-term attribution anchors of
the framework. Stability of this module across versions is a deliberate
commitment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional


# ─── Classification states ───────────────────────────────────────────────────

class Health(Enum):
    """Training-time structural-health classification.

    ``UNKNOWN`` is emitted before sufficient trajectory history exists
    (controlled by ``MIN_HISTORY`` in :mod:`nonans.decision`).
    """
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    GHOST = "ghost"
    REAL = "real"


class RuntimeHealth(Enum):
    """Inference-time attention-pathology classification."""
    HEALTHY = "healthy"
    COLLAPSE = "collapse"
    MAXIMUM = "maximum"
    DEVIATION = "deviation"
    UNKNOWN = "unknown"


# ─── Ring buffer ─────────────────────────────────────────────────────────────

class RingBuffer:
    """Fixed-capacity deque of float samples with O(1) push and slicing.

    Used for the shape-score trajectory window and the norm history.
    Capacity is bounded; oldest samples drop on overflow.
    """

    __slots__ = ("_buf", "_capacity")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = int(capacity)
        self._buf: Deque[float] = deque(maxlen=self._capacity)

    def push(self, value: float) -> None:
        self._buf.append(float(value))

    def to_list(self) -> List[float]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._capacity

    def clear(self) -> None:
        self._buf.clear()


# ─── FaultContext: the typed interface ───────────────────────────────────────

@dataclass
class FaultContext:
    """Structured record handed to the resolution boundary.

    Fields fall into five groups:
      * identity: which tensor, at what step
      * classification: discrete state + confidence
      * structural signals: the five quantities the classifier consumes
      * geometry: current top-k singular values + last-healthy snapshot
      * history: trajectory window + Gate-2 scan latency

    Confidence decays linearly with step age via
    :meth:`adjusted_confidence`, preventing stale contexts from driving
    over-confident downstream actions.
    """
    # identity
    tensor_id: str
    shape: tuple
    step: int
    # classification
    health: Health
    confidence: float
    # structural signals
    signal: float
    shape_value: float
    collapse_rate: float
    restoring_force: float
    momentum_align: float
    # geometry
    sigma_topk: List[float]
    sigma_last_healthy: Optional[List[float]] = None
    # history + cost
    shape_trajectory: List[float] = field(default_factory=list)
    scan_us: float = 0.0

    def adjusted_confidence(self, current_step: int, max_age: int = 50) -> float:
        """Confidence linearly decayed by step age (zero at ``max_age``)."""
        age = max(0, int(current_step) - int(self.step))
        if age >= max_age:
            return 0.0
        return float(self.confidence) * (1.0 - age / float(max_age))


# ─── Thresholds (locked; do not change without rerunning B2/B4/B5) ───────────

SHAPE_HEALTHY = 0.72
SHAPE_CRITICAL = 0.15
RATE_THRESHOLD = -0.025
FORCE_THRESHOLD = 1e-4
MOMENTUM_THRESHOLD = -0.3
MIN_HISTORY = 5
EVIDENCE_REAL = 2
DEFAULT_TRAJECTORY_WINDOW = 12
DEFAULT_NORM_HISTORY = 24
DEFAULT_SCAN_EVERY = 25
DEFAULT_GATE1_TAU = 0.12
