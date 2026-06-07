"""
Tests for nonans.protocol.

Covers:
  * FaultContext dataclass fields and confidence decay
  * RingBuffer push/length/slice/clear semantics
  * Health and RuntimeHealth enum membership
"""
from __future__ import annotations

import pytest

from nonans.protocol import (
    FaultContext,
    Health,
    RingBuffer,
    RuntimeHealth,
)


class TestRingBuffer:
    def test_zero_capacity_rejected(self):
        with pytest.raises(ValueError):
            RingBuffer(0)

    def test_push_and_length(self):
        buf = RingBuffer(4)
        for v in (1.0, 2.0, 3.0):
            buf.push(v)
        assert len(buf) == 3
        assert buf.to_list() == [1.0, 2.0, 3.0]

    def test_overflow_drops_oldest(self):
        buf = RingBuffer(3)
        for v in (1.0, 2.0, 3.0, 4.0):
            buf.push(v)
        assert buf.to_list() == [2.0, 3.0, 4.0]
        assert len(buf) == 3

    def test_clear(self):
        buf = RingBuffer(4)
        buf.push(1.0)
        buf.clear()
        assert len(buf) == 0

    def test_capacity_immutable(self):
        buf = RingBuffer(8)
        assert buf.capacity == 8


class TestFaultContext:
    def _ctx(self, step: int = 100, confidence: float = 1.0) -> FaultContext:
        return FaultContext(
            tensor_id="layer_0.weight",
            shape=(64, 64),
            step=step,
            health=Health.GHOST,
            confidence=confidence,
            signal=0.4,
            shape_value=0.5,
            collapse_rate=-0.03,
            restoring_force=1e-5,
            momentum_align=-0.4,
            sigma_topk=[1.0, 0.5, 0.25],
            sigma_last_healthy=[1.0, 0.8, 0.6],
            shape_trajectory=[0.7, 0.6, 0.5],
            scan_us=312.0,
        )

    def test_required_fields_present(self):
        ctx = self._ctx()
        assert ctx.tensor_id == "layer_0.weight"
        assert ctx.health is Health.GHOST
        assert isinstance(ctx.shape, tuple)

    def test_confidence_decay_zero_age(self):
        ctx = self._ctx(step=100, confidence=0.8)
        assert ctx.adjusted_confidence(current_step=100) == pytest.approx(0.8)

    def test_confidence_decay_partial_age(self):
        ctx = self._ctx(step=100, confidence=1.0)
        # age = 25, max_age = 50 → decay = 0.5
        assert ctx.adjusted_confidence(current_step=125, max_age=50) == pytest.approx(0.5)

    def test_confidence_decay_max_age(self):
        ctx = self._ctx(step=100, confidence=1.0)
        assert ctx.adjusted_confidence(current_step=150, max_age=50) == 0.0

    def test_confidence_decay_past_max_age(self):
        ctx = self._ctx(step=100, confidence=1.0)
        assert ctx.adjusted_confidence(current_step=200, max_age=50) == 0.0

    def test_confidence_decay_negative_age_clipped(self):
        # Current step earlier than fault step → no negative age
        ctx = self._ctx(step=100, confidence=0.7)
        assert ctx.adjusted_confidence(current_step=50) == pytest.approx(0.7)


class TestEnums:
    def test_health_states(self):
        assert {h.value for h in Health} == {"unknown", "healthy", "ghost", "real"}

    def test_runtime_health_states(self):
        assert {h.value for h in RuntimeHealth} == {
            "healthy", "collapse", "maximum", "deviation", "unknown"
        }
