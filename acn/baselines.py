"""Non-optimizing baselines. Both are causal: they only use what is known up to slot t.

Uncontrolled: each car charges at its max rate from plug-in until its energy is delivered.

Equal sharing: at each slot the site cap C is split equally among cars that are
present and still need energy. A car never gets more than its own limit
(max rate x fraction of the slot present, or its remaining need); what it cannot use
is redistributed to the others ("water-filling"). Cars that are done get nothing.
"""

from __future__ import annotations

import numpy as np

from acn.slots import Instance


def uncontrolled(inst: Instance) -> np.ndarray:
    R = np.zeros((inst.n, inst.T))
    remaining = inst.energy.copy()
    cap = inst.max_power()
    for t in range(inst.T):
        R[:, t] = np.minimum(cap[:, t], remaining / inst.dt_h)
        remaining -= R[:, t] * inst.dt_h
    return R


def water_fill(limits: np.ndarray, total: float) -> np.ndarray:
    """Split `total` equally, each share capped at its limit; leftovers go to the rest.

    Returns x with x_i = min(limit_i, level) and sum(x) = min(total, sum(limits)).
    """
    if limits.sum() <= total:
        return limits.copy()
    lim = np.sort(limits)
    k = len(lim)
    used = 0.0
    for j, li in enumerate(lim):
        level = (total - used) / (k - j)  # equal share for the k-j cars still uncapped
        if li >= level:
            return np.minimum(limits, level)
        used += li
    return limits.copy()  # unreachable: sum(limits) > total guarantees a return above


def equal_share(inst: Instance, cap_kw: float | np.ndarray) -> np.ndarray:
    """cap_kw: one site cap, or an array of per-operational-day caps (len n_days)."""
    caps = np.broadcast_to(np.asarray(cap_kw, float), (inst.n_days,))
    R = np.zeros((inst.n, inst.T))
    remaining = inst.energy.copy()
    cap = inst.max_power()
    for t in range(inst.T):
        limits = np.minimum(cap[:, t], remaining / inst.dt_h)
        R[:, t] = water_fill(limits, caps[inst.day_index[t]])
        remaining -= R[:, t] * inst.dt_h
    return R
