"""Core experiment: uncontrolled vs equal sharing vs peak-minimizing LP, per operational day.

Modes:
  daily   one LP per 03:00-03:00 day; sessions crossing the boundary are dropped (counted).
  weekly  one LP per Monday-Monday week, objective 'daily' (sum of daily peaks) or 'max'
          (single weekly peak). Overnight sessions are kept; only week-boundary crossers drop.

Equal sharing is run with its site cap set to the LP's optimal peak for that day, so
both schedules have the same peak and the comparison is on unmet energy.

Usage:
    python -m acn.run_core --site jpl --start 2019-01-01 --end 2020-02-29
    python -m acn.run_core --site caltech --start 2019-01-01 --end 2020-02-29 --mode weekly --objective max
Writes results/core_<site>_<mode>[_<objective>]_<start>_<end>.csv (one row per day).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from acn.baselines import equal_share, uncontrolled
from acn.clean import load_clean
from acn.metrics import daily_peaks, evaluate
from acn.models import min_peak
from acn.slots import Instance, daily_instances, usable_sessions, weekly_instances

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def day_rows(inst: Instance, objective: str, r_min_kw: float) -> list[dict]:
    """Run all methods on one instance; return one row per operational day."""
    if inst.n == 0:
        return []
    res = min_peak(inst, objective, r_min_kw)
    if res.status != "Optimal":
        print(f"  {inst.slot_starts[0]:%Y-%m-%d}: LP {res.status}")
        return []
    R = {"unc": uncontrolled(inst), "lp": res.R}
    pk = {k: daily_peaks(inst, v) for k, v in R.items()}

    # Equal sharing with each day's cap = that day's LP peak (same peak, compare unmet energy).
    Req = equal_share(inst, pk["lp"])
    eq_peak = daily_peaks(inst, Req)
    # Attribute unmet energy to the day the session departs (its last available slot).
    last_slot = inst.T - 1 - np.argmax(inst.avail[:, ::-1] > 0, axis=1)
    unmet = np.clip(inst.energy - Req.sum(1) * inst.dt_h, 0, None)
    eq_unmet = np.bincount(inst.day_index[last_slot], weights=unmet, minlength=inst.n_days)

    first_slot = np.argmax(inst.avail > 0, axis=1)
    arr_day = inst.day_index[first_slot]
    lp_eval = evaluate(inst, res.R)
    rows = []
    for d in range(inst.n_days):
        on = arr_day == d
        rows.append({
            "date": inst.slot_starts[inst.day_index == d][0].date(),
            "n_sessions": int(on.sum()),
            "energy_kwh": float(inst.energy[on].sum()),
            "peak_uncontrolled_kw": pk["unc"][d],
            "peak_lp_kw": pk["lp"][d],
            "peak_equal_share_kw": eq_peak[d],
            "unmet_equal_share_kwh": eq_unmet[d],
            "lp_unmet_kwh": lp_eval["unmet_kwh"] if inst.n_days == 1 else np.nan,
            "lp_violations": lp_eval["over_rate_or_absent"] + lp_eval["neg_rate"],
            # Per-instance counts go on the instance's first day only, so column sums are right.
            "n_dropped_boundary": inst.info["n_dropped_boundary"] if d == 0 else 0,
            "n_clipped": inst.info["n_clipped"] if d == 0 else 0,
            "solve_s": res.solve_s,
        })
    return rows


def run(site: str, start: str, end: str, mode: str = "daily", objective: str = "daily",
        r_min_kw: float = 0.0, boundary_hour: int = 3) -> pd.DataFrame:
    df = usable_sessions(load_clean(site))
    gen = daily_instances if mode == "daily" else weekly_instances
    rows = []
    for inst in gen(df, site, start, end, boundary_hour):
        rows += day_rows(inst, objective, r_min_kw)
    out = pd.DataFrame(rows)
    if len(out):
        out["weekday"] = pd.to_datetime(out["date"]).dt.day_name()
        out["peak_reduction"] = 1 - out["peak_lp_kw"] / out["peak_uncontrolled_kw"]
        out["unmet_equal_share_frac"] = out["unmet_equal_share_kwh"] / out["energy_kwh"]
    return out


def summarize(out: pd.DataFrame) -> None:
    wk = out[~out["weekday"].isin(["Saturday", "Sunday"]) & (out["n_sessions"] >= 5)]
    print(f"days: {len(out)} | weekdays with >=5 sessions: {len(wk)}")
    cols = ["n_sessions", "peak_uncontrolled_kw", "peak_lp_kw", "peak_reduction",
            "unmet_equal_share_frac"]
    print(wk[cols].describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_string())
    print(f"LP violations total: {int(out['lp_violations'].sum())} | "
          f"sessions dropped at boundary: {int(out['n_dropped_boundary'].sum())} | "
          f"clipped: {int(out['n_clipped'].sum())}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", required=True, choices=("caltech", "jpl", "office001"))
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="last local date, inclusive")
    p.add_argument("--mode", choices=("daily", "weekly"), default="daily")
    p.add_argument("--objective", choices=("daily", "max"), default="daily")
    p.add_argument("--r-min-kw", type=float, default=0.0)
    p.add_argument("--boundary-hour", type=int, default=3)
    a = p.parse_args()
    out = run(a.site, a.start, a.end, a.mode, a.objective, a.r_min_kw, a.boundary_hour)
    RESULTS.mkdir(exist_ok=True)
    tag = a.mode + (f"_{a.objective}" if a.mode == "weekly" else "")
    f = RESULTS / f"core_{a.site}_{tag}_{a.start}_{a.end}.csv"
    out.to_csv(f, index=False)
    summarize(out)
    print(f"-> {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
