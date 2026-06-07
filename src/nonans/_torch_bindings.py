"""
nonans._torch_bindings
======================

The (small) torch-only surface of the library: functions that must read
gradients or optimizer state, which only exist in a torch context.

This module is imported only by :mod:`nonans.engine` and
:mod:`nonans.bindings` when ``nonans.wrap(model, optimizer)`` is called.
The math primitives in :mod:`nonans.primitives` do not depend on torch.

If torch is not installed, importing this module raises ``ImportError``;
users on numpy-only paths never trigger that import.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def restoring_force(param: torch.Tensor) -> float:
    """L2 norm of a parameter's current gradient.

    Returns 0.0 if no gradient is present yet (pre-backward step) or on
    failure to materialize. Detaches before reading; monitoring must not
    affect autograd.
    """
    g = getattr(param, "grad", None)
    if g is None:
        return 0.0
    try:
        return float(g.detach().to(torch.float32).norm())
    except (RuntimeError, TypeError):
        return 0.0


def momentum_aligned_to_collapse(
    param: torch.Tensor,
    optimizer,
    threshold: float = -0.3,
) -> bool:
    """True when the Adam first moment opposes the current gradient.

    Cos(m, g) < threshold signals that the optimizer is reinforcing the
    direction of collapse — the third fault signal in the ghost/real
    classifier. Returns False for non-Adam optimizers or when state is
    not yet populated; this is a graceful no-vote rather than a
    misleading positive.
    """
    try:
        state = optimizer.state.get(param)
        if state is None or "exp_avg" not in state:
            return False
        m = state["exp_avg"]
        g = param.grad
        if g is None or m.shape != g.shape:
            return False
        m_f = m.detach().to(torch.float32).flatten()
        g_f = g.detach().to(torch.float32).flatten()
        if float(m_f.norm()) < 1e-8 or float(g_f.norm()) < 1e-8:
            return False
        cos = float(F.cosine_similarity(m_f.unsqueeze(0), g_f.unsqueeze(0)).squeeze())
        return cos < threshold
    except (RuntimeError, KeyError, AttributeError):
        return False
