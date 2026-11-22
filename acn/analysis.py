"""Helpers for result notebooks: rebuild per-day schedules and baseline cap sweeps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from acn.baselines import equal_share, uncontrolled
from acn.clean import load_clean
from acn.metrics import evaluate
from acn.models import min_peak
from acn.slots import Instance, daily_instances, usable_sessions

UNMET_TOL_KWH = 0.01


def load_day_instances(site: str, start: str, end: str, min_sessions: int = 1) -> dict:
    """{date: Instance} for operational days with at least min_sessions sessions."""
    df = usable_sessions(load_clean(site))
    out = {}
    for inst in daily_instances(df, site, start, end):
        if inst.n >= min_sessions:
            out[inst.slot_starts[0].date()] = inst
    return out


def day_schedules(inst: Instance, smooth: bool = True) -> dict[str, np.ndarray]:
    """Schedules of all three methods; equal sharing capped at the LP's optimal peak.

    smooth=True picks the evenest optimal LP schedule (same peak; see models.min_peak)."""
    lp = min_peak(inst, smooth=smooth).R
    return {"lp": lp, "uncontrolled": uncontrolled(inst),
            "equal_share": equal_share(inst, lp.sum(0).max())}


def unmet_kwh(inst: Instance, R: np.ndarray) -> float:
    return evaluate(inst, R)["unmet_kwh"]


def min_cap_equal_share(inst: Instance, hi: float, iters: int = 25) -> float:
    """Smallest site cap at which equal sharing leaves (almost) no energy unmet, by bisection.

    At cap >= the uncontrolled peak, equal sharing reproduces uncontrolled charging
    (every car gets its full limit), so `hi` = uncontrolled peak is a valid upper bracket.
    Assumes unmet energy is non-increasing in the cap; `cap_sweep` lets you eyeball that.
    """
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if unmet_kwh(inst, equal_share(inst, mid)) <= UNMET_TOL_KWH:
            hi = mid
        else:
            lo = mid
    return hi


def cap_sweep(instances: dict, lp_peaks: dict, multipliers: np.ndarray) -> pd.DataFrame:
    """Equal-sharing unmet energy when its cap is m x (that day's LP peak)."""
    rows = []
    for d, inst in instances.items():
        for m in multipliers:
            u = unmet_kwh(inst, equal_share(inst, m * lp_peaks[d]))
            rows.append({"date": d, "multiplier": m, "cap_kw": m * lp_peaks[d],
                         "unmet_kwh": u, "unmet_frac": u / inst.energy.sum()})
    return pd.DataFrame(rows)


def occupancy(inst: Instance) -> np.ndarray:
    """(T,) cars plugged in per slot (fractional at arrival/departure slots)."""
    return inst.avail.sum(axis=0)
