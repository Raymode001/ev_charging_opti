"""Peak-minimizing charging LP (perfect information: actual arrival, departure, energy).

Decision r[i,t] >= 0: average power (kW) to car i in slot t, only where avail[i,t] > 0.

    r[i,t] <= avail[i,t] * p_max                         (rate limit; car absent -> 0)
    sum_t r[i,t] * dt = E_i                              (deliver kWhDelivered before leaving)
    sum_i r[i,t] <= peak[day(t)]                         (site load vs that day's peak)

objective="daily": minimize sum_d peak[d]   (each day's peak; separable unless sessions span days)
objective="max":   minimize max_t load      (one peak for the whole horizon, e.g. a billing period),
                   plus eps * sum_d peak[d] so non-peak days are not left arbitrary.

Optional minimum rate (r_min_kw > 0), modelled as minimum *progress* so it stays an LP:
    sum_{s<=t} r[i,s] dt >= min(E_i, r_min * sum_{s<=t} avail[i,s] dt)
i.e. a car that leaves at any moment has at least what charging at r_min would have given it.
A literal per-slot "r >= r_min while charging" is semi-continuous and needs binaries (MIP extension).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pulp

from acn.slots import Instance


@dataclass
class Result:
    R: np.ndarray
    status: str
    objective: float
    peaks: np.ndarray  # (n_days,) peak variable values
    solve_s: float


def solver(msg: bool = False) -> pulp.LpSolver:
    # Bundled CBC is x86-only on Apple Silicon; HiGHS (pip install highspy) works everywhere.
    return pulp.HiGHS(msg=msg)


def min_peak(inst: Instance, objective: str = "daily", r_min_kw: float = 0.0,
             eps: float = 1e-4, msg: bool = False, smooth: bool = False) -> Result:
    """Solve the peak LP. smooth=True adds a second, lexicographic stage that keeps every
    day's peak at its optimum and minimizes the sum over cars of each car's highest
    instantaneous power (r_it / a_it), so each car charges as evenly as the peak allows.
    The first-stage optimum is not unique; this only picks a nicer optimal schedule."""
    if objective not in ("daily", "max"):
        raise ValueError("objective must be 'daily' or 'max'")

    ub = inst.max_power()
    idx = list(zip(*np.nonzero(ub > 0), strict=True))
    prob = pulp.LpProblem("min_peak", pulp.LpMinimize)
    r = {(i, t): pulp.LpVariable(f"r_{i}_{t}", lowBound=0, upBound=float(ub[i, t])) for i, t in idx}
    peak = [pulp.LpVariable(f"peak_{d}", lowBound=0) for d in range(inst.n_days)]

    if objective == "daily":
        prob += pulp.lpSum(peak)
    else:
        z = pulp.LpVariable("peak_max", lowBound=0)
        for p in peak:
            prob += p <= z
        prob += z + eps * pulp.lpSum(peak)

    by_car: dict[int, list[int]] = {}
    by_slot: dict[int, list[int]] = {}
    for i, t in idx:
        by_car.setdefault(i, []).append(t)
        by_slot.setdefault(t, []).append(i)

    for i, ts in by_car.items():
        prob += pulp.lpSum(r[i, t] for t in ts) * inst.dt_h == float(inst.energy[i]), f"energy_{i}"
        if r_min_kw > 0:  # ts is ascending: np.nonzero walks row-major
            cum_avail = 0.0
            for k, t in enumerate(ts[:-1]):  # last slot is implied by the energy equality
                cum_avail += inst.avail[i, t] * inst.dt_h
                # A car slower than r_min can only be asked to keep pace at its own max rate.
                need = min(float(inst.energy[i]), min(r_min_kw, inst.p_max[i]) * cum_avail)
                if need > 0:
                    prob += pulp.lpSum(r[i, s] for s in ts[:k + 1]) * inst.dt_h >= need
    for t, cars in by_slot.items():
        prob += pulp.lpSum(r[i, t] for i in cars) <= peak[inst.day_index[t]], f"load_{t}"

    t0 = time.perf_counter()
    prob.solve(solver(msg))
    solve_s = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]

    if status == "Optimal" and smooth:
        peak_star = [p.value() or 0.0 for p in peak]
        for d, v in enumerate(peak_star):
            prob += peak[d] <= v * (1 + 1e-7) + 1e-6, f"fix_peak_{d}"
        m = [pulp.LpVariable(f"m_{i}", lowBound=0) for i in range(inst.n)]
        for (i, t), var in r.items():
            prob += var <= float(inst.avail[i, t]) * m[i]  # instantaneous power <= m_i
        prob.setObjective(pulp.lpSum(m))
        prob.solve(solver(msg))
        status = pulp.LpStatus[prob.status]
        solve_s = time.perf_counter() - t0

    R = np.zeros((inst.n, inst.T))
    if status == "Optimal":
        for (i, t), v in r.items():
            R[i, t] = v.value() or 0.0
        R = np.clip(R, 0, ub)  # remove solver round-off outside bounds
    peaks = np.array([p.value() or 0.0 for p in peak])
    obj = float(peaks.sum()) if smooth else float(pulp.value(prob.objective) or np.nan)
    return Result(R, status, obj, peaks, solve_s)
