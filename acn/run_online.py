"""Departure-uncertainty experiment: offline LP vs online MPC with different departure estimates.

For every operational day, with per-car observed rates (rates.py):
    offline_lp        perfect information (knows all arrivals and departures): lower bound on peak
    mpc_<estimator>   rolling horizon, sees cars only once they arrive; departures from the estimator
    mpc_*_rmin<A>     same, plus an online minimum-progress guarantee (8 A = 1.66 kW, 16 A = 3.33 kW)
    uncontrolled      full speed from plug-in
    equal_share       site cap = the mpc_driver peak of that day (same peak, compare unmet energy)

Writes results/online/online_<site>_<start>_<end>.csv, one row per (day, policy).

Usage:
    python -m acn.run_online --site jpl --start 2019-01-01 --end 2020-02-29
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from acn.baselines import equal_share, uncontrolled
from acn.clean import load_clean
from acn.metrics import evaluate
from acn.models import min_peak
from acn.online import departure_estimates, simulate
from acn.rates import RATE_COL, estimate_car_rates
from acn.slots import Instance, daily_instances, ns, usable_sessions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "online"
ESTIMATORS = ("true", "driver", "conservative", "none")
AMPS_TO_KW = 208 / 1000
# (policy name, estimator, minimum-progress rate in kW)
POLICIES = [(f"mpc_{e}", e, 0.0) for e in ESTIMATORS] + [
    (f"mpc_{e}_rmin{a}A", e, a * AMPS_TO_KW) for e in ("driver", "none") for a in (8, 12, 16)
]
# Equal sharing is causal and needs no departure info; trace it at caps = m x offline LP peak.
EQ_MULTIPLIERS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)


def _ns_or_nan(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), np.nan)
    ok = s.notna().to_numpy()
    if ok.any():
        out[ok] = ns(s[ok]).astype(float)
    return out


def day_rows(inst: Instance, sess: pd.DataFrame, estimates: dict) -> list[dict]:
    s = sess.set_index("sessionID").loc[inst.session_ids]
    arrive, depart = ns(s["connectionTime"]), ns(s["disconnectTime"])
    base = {"date": inst.slot_starts[0].date(), "weekday": inst.slot_starts[0].day_name(),
            "n_sessions": inst.n, "energy_kwh": float(inst.energy.sum())}

    def row(policy: str, R: np.ndarray, unmet: np.ndarray | None = None) -> dict:
        ev = evaluate(inst, R)
        u = ev["unmet_kwh"] if unmet is None else float(unmet.sum())
        n_u = ev["n_sessions_unmet"] if unmet is None else int((unmet > 0.01).sum())
        return {**base, "policy": policy, "peak_kw": ev["peak_kw"], "unmet_kwh": u,
                "unmet_frac": u / inst.energy.sum(), "n_sessions_unmet": n_u,
                "violations": ev["over_rate_or_absent"] + ev["neg_rate"]}

    rows = [row("offline_lp", min_peak(inst).R), row("uncontrolled", uncontrolled(inst))]
    est_ns = {e: _ns_or_nan(est.loc[inst.session_ids]) for e, est in estimates.items()}
    for name, e, r_min in POLICIES:
        res = simulate(inst, arrive, depart, est_ns[e], r_min_kw=r_min)
        rows.append(row(name, res.R, res.unmet))
    cap = next(r["peak_kw"] for r in rows if r["policy"] == "mpc_driver")
    rows.append(row("equal_share_at_mpc_driver_peak", equal_share(inst, cap)))
    lp_peak = rows[0]["peak_kw"]
    for m in EQ_MULTIPLIERS:
        rows.append(row(f"equal_share_x{m:.1f}", equal_share(inst, m * lp_peak)))
    return rows


def run(site: str, start: str, end: str, weekdays_only: bool = True) -> pd.DataFrame:
    df = estimate_car_rates(usable_sessions(load_clean(site)))
    estimates = {e: departure_estimates(df, e) for e in ESTIMATORS}
    rows = []
    for inst in daily_instances(df, site, start, end, rate_col=RATE_COL):
        if inst.n == 0 or (weekdays_only and inst.slot_starts[0].dayofweek >= 5):
            continue
        rows += day_rows(inst, df, estimates)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", default="jpl")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2020-02-29")
    a = p.parse_args()
    t = time.perf_counter()
    out = run(a.site, a.start, a.end)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"online_{a.site}_{a.start}_{a.end}.csv"
    out.to_csv(f, index=False)
    wk = out[out["n_sessions"] >= 5]
    print(f"{out['date'].nunique()} weekdays in {time.perf_counter() - t:.0f}s -> {f.relative_to(ROOT)}")
    print(wk.groupby("policy")[["peak_kw", "unmet_frac", "n_sessions_unmet", "violations"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main()
