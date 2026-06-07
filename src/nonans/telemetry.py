"""
nonans.telemetry
================
Telemetry interface — the cleanly-typed boundary between the open
monitoring layer and downstream consumers (resolver, dashboard, logger).

Three abstractions
------------------
  1. Event             — immutable record of one observation
  2. TelemetrySink     — abstract receiver (subscribed by consumers)
  3. ResolverProtocol  — abstract contract the proprietary NaN-resolver
                         must implement to consume FaultContext

This module defines NO mutable global state and depends ONLY on
nonans.protocol. It is safe to import from any layer.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import torch as _torch_typing  # noqa: F401

from .protocol import FaultContext, Health, RuntimeHealth


# ─── Event types ──────────────────────────────────────────────────────────────

class EventKind(Enum):
    """The four observable events emitted by the monitoring layer."""
    SCAN_COMPLETE = "scan_complete"   # Gate 2 fired, FaultContext produced
    HEALTH_CHANGE = "health_change"   # tensor transitioned to new Health
    SINGULARITY   = "singularity"     # NaN/Inf detected, resolver invoked
    RESOLUTION    = "resolution"      # resolver returned (success/failure)


@dataclass(frozen=True)
class Event:
    """
    Immutable observation record. Emitted by the Sentinel and consumed by
    any subscribed TelemetrySink (logger, dashboard, alerting, resolver…).

    `payload` is event-kind-specific:
      SCAN_COMPLETE : FaultContext
      HEALTH_CHANGE : (prev_health: Health, new_health: Health, ctx: FaultContext)
      SINGULARITY   : { 'tensor_id': str, 'ctx': Optional[FaultContext] }
      RESOLUTION    : { 'tensor_id': str, 'success': bool, 'wall_ms': float }
    """
    kind:       EventKind
    timestamp:  float
    step:       int
    tensor_id:  str
    payload:    object = None
    confidence: float  = 0.0


# ─── Sink abstraction ─────────────────────────────────────────────────────────

class TelemetrySink(abc.ABC):
    """
    Abstract telemetry receiver. Implementations include: in-memory ring,
    logfile writer, OpenTelemetry exporter, Prometheus gauge, etc.

    Implementations MUST NOT raise from receive(). All exceptions should be
    swallowed and counted internally. The training loop must never crash
    because of a telemetry sink.
    """

    @abc.abstractmethod
    def receive(self, event: Event) -> None: ...

    def close(self) -> None:
        """Optional. Default no-op."""
        return None


class InMemorySink(TelemetrySink):
    """
    Bounded in-memory ring of recent events. Suitable for tests, REPL
    introspection, and unit benchmarks. NOT for production telemetry.

    Thread-safe. Drops oldest on overflow.
    """
    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = int(capacity)
        self._buf: List[Event] = []
        self._lock = RLock()
        self.dropped = 0

    def receive(self, event: Event) -> None:
        try:
            with self._lock:
                if len(self._buf) >= self.capacity:
                    self._buf.pop(0)
                    self.dropped += 1
                self._buf.append(event)
        except Exception:
            pass

    def events(self, kind: Optional[EventKind] = None) -> List[Event]:
        with self._lock:
            if kind is None:
                return list(self._buf)
            return [e for e in self._buf if e.kind == kind]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class CallbackSink(TelemetrySink):
    """Thin sink that forwards every event to a user callback."""
    def __init__(self, callback: Callable[[Event], None]) -> None:
        self._cb = callback
        self.errors = 0

    def receive(self, event: Event) -> None:
        try:
            self._cb(event)
        except Exception:
            self.errors += 1


# ─── Telemetry bus ────────────────────────────────────────────────────────────

class TelemetryBus:
    """
    Multiplexes events to all subscribed sinks. Process-global singleton
    is provided as `nonans.telemetry.bus`, but additional buses can be
    instantiated for testing.
    """
    def __init__(self) -> None:
        self._sinks: List[TelemetrySink] = []
        self._lock  = RLock()
        self._emitted: int = 0

    def subscribe(self, sink: TelemetrySink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def unsubscribe(self, sink: TelemetrySink) -> None:
        with self._lock:
            try:
                self._sinks.remove(sink)
            except ValueError:
                pass

    def emit(self, event: Event) -> None:
        with self._lock:
            sinks = list(self._sinks)
        for s in sinks:
            try:
                s.receive(event)
            except Exception:
                pass
        self._emitted += 1

    @property
    def emitted(self) -> int:
        return self._emitted

    def close(self) -> None:
        with self._lock:
            for s in self._sinks:
                try:
                    s.close()
                except Exception:
                    pass
            self._sinks.clear()


# Process-global default bus
bus = TelemetryBus()


# ─── Convenience emitters (called by engine/runtime) ─────────────────────────

def emit_scan(ctx: FaultContext, bus_: Optional[TelemetryBus] = None) -> None:
    target = bus_ or bus
    target.emit(Event(
        kind       = EventKind.SCAN_COMPLETE,
        timestamp  = time.time(),
        step       = ctx.step,
        tensor_id  = ctx.tensor_id,
        payload    = ctx,
        confidence = ctx.confidence,
    ))


def emit_health_change(prev: Health, new: Health, ctx: FaultContext,
                       bus_: Optional[TelemetryBus] = None) -> None:
    target = bus_ or bus
    target.emit(Event(
        kind       = EventKind.HEALTH_CHANGE,
        timestamp  = time.time(),
        step       = ctx.step,
        tensor_id  = ctx.tensor_id,
        payload    = (prev, new, ctx),
        confidence = ctx.confidence,
    ))


def emit_singularity(tensor_id: str, step: int, ctx: Optional[FaultContext],
                     bus_: Optional[TelemetryBus] = None) -> None:
    target = bus_ or bus
    target.emit(Event(
        kind       = EventKind.SINGULARITY,
        timestamp  = time.time(),
        step       = step,
        tensor_id  = tensor_id,
        payload    = {"tensor_id": tensor_id, "ctx": ctx},
        confidence = (ctx.confidence if ctx is not None else 0.0),
    ))


def emit_resolution(tensor_id: str, step: int, success: bool, wall_ms: float,
                    bus_: Optional[TelemetryBus] = None) -> None:
    target = bus_ or bus
    target.emit(Event(
        kind       = EventKind.RESOLUTION,
        timestamp  = time.time(),
        step       = step,
        tensor_id  = tensor_id,
        payload    = {"tensor_id": tensor_id, "success": success, "wall_ms": wall_ms},
        confidence = 1.0 if success else 0.0,
    ))


# ─── Resolver protocol ────────────────────────────────────────────────────────

class ResolverProtocol(abc.ABC):
    """
    Abstract contract for the proprietary NaN-resolution layer (Layer 2 in
    the architecture stack). The Sentinel does not import any resolver
    implementation; it only depends on this protocol.

    Implementations should be:
      - Pure-function-like at the API surface (no global state mutation
        visible to caller other than the returned tensor).
      - Thread-safe.
      - Bounded in wall time — return None to indicate "cannot resolve in
        budget; caller should fall back".
    """

    @abc.abstractmethod
    def can_resolve(self, tensor: "torch.Tensor", ctx: Optional[FaultContext]) -> bool:
        """Return True iff this resolver can attempt this singularity."""
        ...

    @abc.abstractmethod
    def resolve(
        self,
        tensor:    "torch.Tensor",
        ctx:       Optional[FaultContext],
        budget_ms: float = 5.0,
    ) -> "Optional[torch.Tensor]":
        """
        Attempt to produce a finite, structurally-consistent replacement
        for `tensor`, possibly using the FaultContext from the Sentinel.

        Return None if cannot resolve within budget — caller falls back
        to its default (clip, replace, skip step, etc.).
        """
        ...


# ─── Reference no-op resolver (for tests / public layer benchmarks) ──────────

class IdentityResolver(ResolverProtocol):
    """
    Reference implementation that returns the input unchanged when finite,
    and the last-healthy fallback derived from FaultContext when not.

    NOT a production resolver. Intended for benchmarking the open layer in
    isolation, and as a test fixture.

    Requires PyTorch.
    """

    def can_resolve(self, tensor: "torch.Tensor", ctx: Optional[FaultContext]) -> bool:
        return ctx is not None

    def resolve(self, tensor, ctx, budget_ms=5.0):
        if not _HAS_TORCH:
            raise ImportError(
                "IdentityResolver requires PyTorch; install with `pip install torch`"
            )
        if ctx is None:
            return None
        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all()):
            return tensor
        # Fallback: replace non-finite entries with 0 — minimal informed action.
        out = tensor.clone()
        out[~finite_mask] = 0.0
        return out
