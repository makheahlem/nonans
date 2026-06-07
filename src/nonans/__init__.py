"""
nonans
======

Numerical Runtime Intelligence (NRI) — reference implementation.

A runtime instrumentation layer for the real-time classification of
legitimate numerical events (overflow, structural rank collapse,
attention concentration) in transformer training and inference.

Public API
----------

Training-time::

    import nonans
    model = nonans.wrap(model, optimizer)
    # ... train as usual; FaultContext records are produced internally.

Inference-time::

    monitor = nonans.RuntimeMonitor(sequence_length=512)
    # ... calibration pass ...
    monitor.finalize_calibration()
    state, conf, H = monitor.classify_head(attn_weights, layer=0, head=0)

Telemetry::

    nonans.bus.subscribe(nonans.InMemorySink(capacity=1024))

The proprietary Resolution Layer that consumes ``FaultContext`` records
at the runtime boundary is deployed in a private production setting and
is NOT included in this repository.
"""
from __future__ import annotations

from ._version import __version__
from .protocol import (
    FaultContext,
    Health,
    RingBuffer,
    RuntimeHealth,
)
from .telemetry import (
    Event,
    InMemorySink,
    ResolverProtocol,
    TelemetryBus,
    bus,
)

# Public re-exports of the math primitives
from .primitives import (
    attention_entropy,
    attention_entropy_batched,
    attention_entropy_normalized,
    collapse_rate,
    norm_deviation,
    randomized_svd_topk,
    shape_score,
    signal_score,
)

# Decision primitives
from .decision import classify, classify_attention


def wrap(model, optimizer, **kwargs):
    """Single integration line for training-time monitoring.

    Wraps a torch model and optimizer with the Sentinel three-gate
    monitoring engine. Requires PyTorch.

    See :func:`nonans.bindings.wrap` for keyword arguments.
    """
    from .bindings import wrap as _wrap
    return _wrap(model, optimizer, **kwargs)


def RuntimeMonitor(*args, **kwargs):
    """Inference-time attention monitor. Requires PyTorch.

    See :class:`nonans.runtime.RuntimeMonitor` for full signature.
    """
    from .runtime import RuntimeMonitor as _RuntimeMonitor
    return _RuntimeMonitor(*args, **kwargs)


__all__ = [
    "__version__",
    # protocol
    "FaultContext",
    "Health",
    "RingBuffer",
    "RuntimeHealth",
    # telemetry
    "Event",
    "InMemorySink",
    "ResolverProtocol",
    "TelemetryBus",
    "bus",
    # primitives
    "attention_entropy",
    "attention_entropy_batched",
    "attention_entropy_normalized",
    "collapse_rate",
    "norm_deviation",
    "randomized_svd_topk",
    "shape_score",
    "signal_score",
    # decision
    "classify",
    "classify_attention",
    # entry points
    "wrap",
    "RuntimeMonitor",
]
