"""Two-stage stochastic model: commit to a site power cap, then schedule each day under it.

Stage 1 (here-and-now, before any day is known):   choose site cap C >= 0 (kW).
Stage 2 (wait-and-see, per scenario s = one historical day):
    schedule r^s under the cap; energy not delivered, u^s_i >= 0, is penalized.

    min_C   c * C  +  (1/S) * sum_s  lam * U_s(C)
    U_s(C) = min  sum_i u_i
             s.t. 0 <= r_it <= a_it * p_i,      sum_t r_it dt + u_i = E_i,
                  sum_i r_it <= C  for all t,   u_i >= 0

c   = demand charge per weekday ($/kW): monthly $/kW-month divided by weekdays per month.
lam = penalty per kWh not delivered ($/kWh).

The recourse is *offline within the day*: once a scenario is revealed, its arrivals,
departures and energies are all known. The uncertainty modelled is which day you get
(day-to-day demand), not intraday departure times (that is the online/rolling extension).

Solution method. C is the only first-stage variable, and each U_s is convex piecewise-linear
and non-increasing in C (an LP value function in a right-hand side). So we tabulate U_s on a
1 kW grid once (warm-started HiGHS, changing only the cap's row bounds), after which any
(c, lam) is a 1-D minimization over the grid. `extensive_form` solves the textbook single LP
with all scenarios for validation; tests check that both give the same objective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import highspy
import numpy as np
import pandas as pd
import pulp

from acn.clean import load_clean
from acn.models import min_peak, solver
from acn.rates import RATE_COL, estimate_car_rates
from acn.slots import Instance, daily_instances, usable_sessions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "two_stage"
WEEKDAYS_PER_MONTH = 21.0
INF = highspy.kHighsInf


class StageTwo:
    """Stage-2 LP for one day; `unmet(C)` re-solves with a new cap, warm-started."""

    def __init__(self, inst: Instance):
        self.inst = inst
        n, T = inst.n, inst.T
        ub = inst.max_power()
        ri, rt = np.nonzero(ub > 0)
        m = len(ri)
        # Columns: r_k (k < m) appear in energy row i and slot row n + t; then u_i in energy row i.
        start = np.r_[np.arange(0, 2 * m + 1, 2), 2 * m + 1 + np.arange(n)]
        index = np.r_[np.column_stack([ri, n + rt]).ravel(), np.arange(n)]
        value = np.r_[np.tile([inst.dt_h, 1.0], m), np.ones(n)]

        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = m + n, n + T
        lp.col_cost_ = np.r_[np.zeros(m), np.ones(n)]
        lp.col_lower_ = np.zeros(m + n)
        lp.col_upper_ = np.r_[ub[ri, rt], np.full(n, INF)]
        lp.row_lower_ = np.r_[inst.energy, np.full(T, -INF)]
        lp.row_upper_ = np.r_[inst.energy, np.full(T, INF)]
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = start
        lp.a_matrix_.index_ = index
        lp.a_matrix_.value_ = value

        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.passModel(lp)
        self.slot_rows = np.arange(n, n + T, dtype=np.int32)
        self._lower = np.full(T, -INF)

    def unmet(self, cap_kw: float) -> float:
        self.h.changeRowsBounds(len(self.slot_rows), self.slot_rows, self._lower,
                                np.full(len(self.slot_rows), float(cap_kw)))
        self.h.run()
        if self.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"stage-2 LP not optimal: {self.h.modelStatusToString(self.h.getModelStatus())}")
        return max(self.h.getInfo().objective_function_value, 0.0)


def unmet_curves(instances: dict, grid: np.ndarray, peaks: dict) -> pd.DataFrame:
    """U[date, C] = min unmet kWh at cap C. Zero at and above the day's optimal peak."""
    rows = {}
    for d, inst in instances.items():
        st = StageTwo(inst)
        rows[d] = [st.unmet(c) if c < peaks[d] - 1e-9 else 0.0 for c in grid]
    return pd.DataFrame.from_dict(rows, orient="index", columns=grid)


@dataclass
class Prices:
    demand_charge_month: float  # $/kW-month
    lam: float  # $/kWh not delivered

    @property
    def c(self) -> float:
        return self.demand_charge_month / WEEKDAYS_PER_MONTH


def cost_curve(U: pd.DataFrame, pr: Prices) -> pd.Series:
    """Expected daily cost f(C) = c*C + lam*mean_s U_s(C) for every grid C."""
    grid = U.columns.to_numpy(float)
    return pd.Series(pr.c * grid + pr.lam * U.mean(axis=0).to_numpy(), index=grid)


def cost_at(U: pd.DataFrame, C: float, pr: Prices) -> float:
    """Expected cost of committing to cap C (nearest grid point at or above C)."""
    grid = U.columns.to_numpy(float)
    j = min(int(np.searchsorted(grid, C - 1e-9)), len(grid) - 1)
    return pr.c * grid[j] + pr.lam * float(U.iloc[:, j].mean())


def solve_saa(U: pd.DataFrame, pr: Prices) -> tuple[float, float]:
    f = cost_curve(U, pr)
    return float(f.idxmin()), float(f.min())


def wait_and_see(U: pd.DataFrame, pr: Prices) -> float:
    """Each day picks its own best cap with full knowledge: a lower bound (WS)."""
    grid = U.columns.to_numpy(float)
    per_day = pr.c * grid[None, :] + pr.lam * U.to_numpy()
    return float(per_day.min(axis=1).mean())


def policy_table(U_train: pd.DataFrame, U_test: pd.DataFrame, peaks_train: pd.Series,
                 pr: Prices) -> pd.DataFrame:
    """Compare cap policies in-sample (train) and out-of-sample (test)."""
    C_saa, _ = solve_saa(U_train, pr)
    policies = {
        "two-stage (SAA)": C_saa,
        "mean-day plan (EV)": float(peaks_train.mean()),
        "median-day plan": float(peaks_train.median()),
        "worst-case plan (robust)": float(peaks_train.max()),
    }
    rows = []
    for name, C in policies.items():
        for split, U in [("train", U_train), ("test", U_test)]:
            grid = U.columns.to_numpy(float)
            j = min(int(np.searchsorted(grid, C - 1e-9)), len(grid) - 1)
            C = float(grid[j])  # policies commit to a grid cap (rounded up), consistent with cost_at
            unmet = U.iloc[:, j]
            rows.append({"policy": name, "split": split, "cap_kw": C,
                         "expected_cost": cost_at(U, C, pr),
                         "demand_cost": pr.c * C, "unmet_cost": pr.lam * unmet.mean(),
                         "mean_unmet_kwh": unmet.mean(),
                         "days_with_unmet": float((unmet > 0.01).mean())})
    for split, U in [("train", U_train), ("test", U_test)]:
        rows.append({"policy": "wait-and-see (perfect info)", "split": split, "cap_kw": np.nan,
                     "expected_cost": wait_and_see(U, pr)})
    return pd.DataFrame(rows)


def evpi_vss(U_train: pd.DataFrame, peaks_train: pd.Series, pr: Prices) -> dict:
    """Textbook quantities on the training scenarios."""
    _, rp = solve_saa(U_train, pr)
    ws = wait_and_see(U_train, pr)
    eev = cost_at(U_train, float(peaks_train.mean()), pr)
    return {"RP": rp, "WS": ws, "EEV": eev, "EVPI": rp - ws, "VSS": eev - rp}


def extensive_form(instances: dict, pr: Prices, msg: bool = False) -> tuple[float, float]:
    """The full two-stage LP with all scenarios in one model (for validation; small S only)."""
    S = len(instances)
    prob = pulp.LpProblem("two_stage", pulp.LpMinimize)
    C = pulp.LpVariable("C", lowBound=0)
    unmet_terms = []
    for s, inst in enumerate(instances.values()):
        ub = inst.max_power()
        r = {(i, t): pulp.LpVariable(f"r_{s}_{i}_{t}", lowBound=0, upBound=float(ub[i, t]))
             for i, t in zip(*np.nonzero(ub > 0), strict=True)}
        u = [pulp.LpVariable(f"u_{s}_{i}", lowBound=0) for i in range(inst.n)]
        unmet_terms += u
        by_car, by_slot = {}, {}
        for (i, t), var in r.items():
            by_car.setdefault(i, []).append(var)
            by_slot.setdefault(t, []).append(var)
        for i in range(inst.n):
            prob += pulp.lpSum(by_car.get(i, [])) * inst.dt_h + u[i] == float(inst.energy[i])
        for terms in by_slot.values():
            prob += pulp.lpSum(terms) <= C
    prob += pr.c * C + (pr.lam / S) * pulp.lpSum(unmet_terms)
    prob.solve(solver(msg))
    return float(C.value()), float(pulp.value(prob.objective))


# ---------- data ----------

def load_scenarios(site: str, start: str, end: str, rates: str = "uniform",
                   weekdays_only: bool = True) -> tuple[dict, pd.Series]:
    """{date: Instance} and each day's optimal (wait-and-see) peak."""
    df = usable_sessions(load_clean(site))
    kw = {}
    if rates == "observed":
        df = estimate_car_rates(df)
        kw = {"rate_col": RATE_COL}
    inst, peaks = {}, {}
    for x in daily_instances(df, site, start, end, **kw):
        d = x.slot_starts[0]
        if x.n == 0 or (weekdays_only and d.dayofweek >= 5):
            continue
        inst[d.date()] = x
        peaks[d.date()] = float(min_peak(x).R.sum(0).max())
    return inst, pd.Series(peaks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", default="jpl")
    p.add_argument("--train", nargs=2, default=["2019-01-01", "2019-12-31"])
    p.add_argument("--test", nargs=2, default=["2020-01-01", "2020-02-29"])
    p.add_argument("--rates", choices=("uniform", "observed"), default="uniform")
    p.add_argument("--step-kw", type=float, default=1.0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.site}_{a.rates}"
    for split, (s, e) in {"train": a.train, "test": a.test}.items():
        inst, peaks = load_scenarios(a.site, s, e, a.rates)
        grid = np.arange(0, np.ceil(peaks.max()) + a.step_kw, a.step_kw)
        U = unmet_curves(inst, grid, peaks.to_dict())
        U.to_csv(OUT / f"unmet_{tag}_{split}.csv")
        peaks.rename("peak_lp_kw").to_csv(OUT / f"peaks_{tag}_{split}.csv")
        print(f"{split}: {len(inst)} weekdays, grid 0..{grid[-1]:.0f} kW -> {OUT.relative_to(ROOT)}", flush=True)


def load_curves(tag: str = "jpl_uniform") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """U_train, U_test, peaks_train, peaks_test, aligned to a common cap grid."""
    def read(split):
        U = pd.read_csv(OUT / f"unmet_{tag}_{split}.csv", index_col=0)
        U.columns = U.columns.astype(float)
        pk = pd.read_csv(OUT / f"peaks_{tag}_{split}.csv", index_col=0).iloc[:, 0]
        return U, pk
    (Utr, ptr), (Ute, pte) = read("train"), read("test")
    grid = sorted(set(Utr.columns) | set(Ute.columns))
    # Beyond a split's own grid every day is fully served: unmet = 0.
    return Utr.reindex(columns=grid, fill_value=0.0), Ute.reindex(columns=grid, fill_value=0.0), ptr, pte


if __name__ == "__main__":
    main()
