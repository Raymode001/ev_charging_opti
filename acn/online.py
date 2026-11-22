"""Online (rolling-horizon / MPC) charging under departure-time uncertainty.

At the start of every slot t the controller sees only cars that have arrived by the end
of slot t (sub-slot lookahead, consistent with fractional availability), their remaining
energy, and an *estimated* departure. It solves

    min  P  +  M * sum_i u_i  +  eps * sum_{i,s} s * r_is * dt
    s.t. sum_{s>=t} r_is dt + u_i = E_i^rem                   (soft: u = energy it expects to miss)
         0 <= r_is <= a^est_is * p_i                          (estimated availability)
         sum_i r_is <= P,   P >= peak so far                  (no penalty for re-using the realized peak)

applies the slot-t rates, and moves on. The eps term breaks ties toward charging earlier
(ACN's "quick charge" idea), which protects against arrivals the controller cannot see.

Reality differs from the plan only through departures: a car that leaves earlier than
estimated stops charging (it gets rate x (actual / estimated presence) in its last slot) and
any remaining energy is unmet; a car that stays past its estimate is treated as leaving
"now + 1 slot", so it is charged as fast as possible.

Optional minimum progress (r_min_kw > 0), the online form of the "minimum rate protects
early leavers" idea: at every step, the slot-t rate of each car gets a lower bound so that
energy delivered since plug-in >= min(E_i, min(r_min, p_i) * hours plugged in). Because the
controller re-plans every slot, enforcing it on the current slot enforces it throughout.

Energy needs are assumed known at plug-in (E_i = kWhDelivered), isolating *departure*
uncertainty. Estimators (`departure_estimates`):
    "true"         actual departure (online arrivals, perfect departures)
    "driver"       driver's first stated requestedDeparture; if missing, arrival + median dwell
                   for that arrival hour (computed on the window; mild lookahead)
    "conservative" driver estimate shifted by the q-quantile of (actual - stated) errors, i.e.
                   assume people leave earlier than they say
    "none"         no departure info: assume every car leaves at the end of the operational day
"""

from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np
import pandas as pd

from acn.slots import Instance, availability, ns

INF = highspy.kHighsInf
BIG_M = 1e3  # $/kWh-scale penalty: delivering energy dominates the peak in the MPC objective
EPS = 1e-4


@dataclass
class OnlineResult:
    R: np.ndarray  # (n, T) realized average power
    unmet: np.ndarray  # (n,) kWh not delivered by actual departure
    n_solves: int


def departure_estimates(sessions: pd.DataFrame, how: str, q: float = 0.25) -> pd.Series:
    """Estimated departure (tz-aware) per sessionID for one estimator."""
    s = sessions.set_index("sessionID")
    if how == "true":
        return s["disconnectTime"]
    if how == "none":
        return pd.Series(pd.NaT, index=s.index)  # filled with end of horizon by the simulator
    hour = s["connectionTime"].dt.hour
    med_dwell = (s["disconnectTime"] - s["connectionTime"]).groupby(hour).median()
    fallback = s["connectionTime"] + hour.map(med_dwell)
    est = s["requestedDeparture_first"].fillna(fallback)
    if how == "driver":
        return est
    if how == "conservative":
        err = (s["disconnectTime"] - s["requestedDeparture_first"]).dropna()
        return est + err.quantile(q)
    raise ValueError(f"unknown estimator {how!r}")


def _solve_window(avail: np.ndarray, rem: np.ndarray, p: np.ndarray, dt_h: float,
                  peak_floor: float, lb0: np.ndarray | None = None) -> np.ndarray:
    """One MPC step on the window [t, T): returns planned rates (n, W).

    lb0: optional (n,) lower bound on each car's rate in the first (current) slot.
    """
    n, W = avail.shape
    ub = avail * p[:, None]
    ri, rt = np.nonzero(ub > 0)
    m = len(ri)
    # Columns: r_k (energy row i, load row n+t), u_i (energy row i), P (all load rows, -1).
    start = np.r_[np.arange(0, 2 * m + 1, 2), 2 * m + 1 + np.arange(n), 2 * m + n + W]
    index = np.r_[np.column_stack([ri, n + rt]).ravel(), np.arange(n), n + np.arange(W)]
    value = np.r_[np.tile([dt_h, 1.0], m), np.ones(n), -np.ones(W)]
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = m + n + 1, n + W
    lp.col_cost_ = np.r_[EPS * dt_h * (rt + 1), np.full(n, BIG_M), 1.0]
    r_lower = np.zeros(m)
    if lb0 is not None:
        first = rt == 0
        r_lower[first] = np.minimum(lb0[ri[first]], ub[ri[first], 0])
    lp.col_lower_ = np.r_[r_lower, np.zeros(n), peak_floor]
    lp.col_upper_ = np.r_[ub[ri, rt], np.full(n, INF), INF]
    lp.row_lower_ = np.r_[rem, np.full(W, -INF)]
    lp.row_upper_ = np.r_[rem, np.zeros(W)]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = index
    lp.a_matrix_.value_ = value
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"MPC LP: {h.modelStatusToString(h.getModelStatus())}")
    x = np.asarray(h.getSolution().col_value)
    plan = np.zeros((n, W))
    plan[ri, rt] = x[:m]
    return np.clip(plan, 0, ub)


def simulate(inst: Instance, arrive: np.ndarray, depart: np.ndarray, est_depart: np.ndarray,
             r_min_kw: float = 0.0) -> OnlineResult:
    """Rolling-horizon control of one day. Times as epoch ns, aligned with inst.session_ids."""
    T, dt = inst.T, pd.Timedelta(hours=inst.dt_h)
    step = int(dt / pd.Timedelta(nanoseconds=1))
    s0 = ns(inst.slot_starts)
    horizon_end = s0[-1] + step
    est = np.where(np.isnan(est_depart), horizon_end, est_depart).astype(np.int64)

    R = np.zeros((inst.n, T))
    rem = inst.energy.copy()
    gone = np.zeros(inst.n, bool)
    unmet = np.zeros(inst.n)
    present_h = np.zeros(inst.n)  # hours plugged in so far (for minimum progress)
    peak, solves = 0.0, 0
    for t in range(T):
        slot_end = s0[t] + step
        # Cars whose actual departure was before this slot have left: book unmet energy.
        leaving = ~gone & (depart <= s0[t])
        unmet[leaving] = rem[leaving]
        rem[leaving] = 0.0
        gone |= leaving
        known = ~gone & (arrive < slot_end) & (rem > 1e-9)
        if not known.any():
            continue
        idx = np.nonzero(known)[0]
        # Believed departure: the estimate; once that has passed and the car is still here,
        # assume it leaves at the end of this slot (charge it as fast as possible).
        believed = np.where(est[idx] > s0[t], est[idx], slot_end)
        a_est = availability(arrive[idx], np.minimum(believed, horizon_end), inst.slot_starts[t:], dt)
        lb0 = None
        if r_min_kw > 0:
            hours_by_slot_end = present_h[idx] + a_est[:, 0] * inst.dt_h
            target = np.minimum(inst.energy[idx], np.minimum(r_min_kw, inst.p_max[idx]) * hours_by_slot_end)
            delivered = inst.energy[idx] - rem[idx]
            lb0 = np.clip((target - delivered) / inst.dt_h, 0, None)
        plan = _solve_window(a_est, rem[idx], inst.p_max[idx], inst.dt_h, peak, lb0)
        solves += 1
        # Realized slot-t power: charging stops when the car actually leaves.
        a_act = availability(arrive[idx], depart[idx], inst.slot_starts[t:t + 1], dt)[:, 0]
        frac = np.divide(a_act, a_est[:, 0], out=np.zeros_like(a_act), where=a_est[:, 0] > 0)
        r = np.minimum(plan[:, 0] * np.minimum(frac, 1.0), rem[idx] / inst.dt_h)
        R[idx, t] = r
        present_h[idx] += a_act * inst.dt_h
        rem[idx] -= r * inst.dt_h
        peak = max(peak, R[:, t].sum())
    unmet[~gone] = rem[~gone]  # cars still plugged in at the end of the horizon
    return OnlineResult(R, unmet, solves)
