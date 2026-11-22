"""Evaluate a schedule R (n x T, kW average power per slot) on an Instance."""

from __future__ import annotations

import numpy as np

from acn.slots import Instance

TOL = 1e-6


def site_load(R: np.ndarray) -> np.ndarray:
    """(T,) total site power per slot. Equals the 15-min average demand utilities bill on."""
    return R.sum(axis=0)


def daily_peaks(inst: Instance, R: np.ndarray) -> np.ndarray:
    load = site_load(R)
    return np.array([load[inst.day_index == d].max(initial=0.0) for d in range(inst.n_days)])


def check_feasible(inst: Instance, R: np.ndarray, tol: float = 1e-5) -> dict:
    """Physical-limit violations (should all be zero for any valid schedule)."""
    over = R - inst.max_power()
    return {"neg_rate": int((R < -tol).sum()),
            "over_rate_or_absent": int((over > tol).sum()),
            "max_violation_kw": float(max(over.max(initial=0.0), -R.min(initial=0.0)))}


def evaluate(inst: Instance, R: np.ndarray) -> dict:
    delivered = R.sum(axis=1) * inst.dt_h
    unmet = np.clip(inst.energy - delivered, 0, None)
    peaks = daily_peaks(inst, R)
    return {
        "n_sessions": inst.n,
        "energy_kwh": float(inst.energy.sum()),
        "delivered_kwh": float(delivered.sum()),
        "peak_kw": float(site_load(R).max(initial=0.0)),
        "sum_daily_peaks_kw": float(peaks.sum()),
        "unmet_kwh": float(unmet.sum()),
        "unmet_frac": float(unmet.sum() / inst.energy.sum()) if inst.n else 0.0,
        "n_sessions_unmet": int((unmet > 0.01).sum()),
        **check_feasible(inst, R),
    }
