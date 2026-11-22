"""Sensitivity runs: how the core result moves with car rates, minimum rate, and site.

Each scenario runs every operational day in the window (weekends included; filter later)
and writes results/sensitivity/<scenario>.csv with one row per day:
    peak_uncontrolled_kw, peak_lp_kw, peak_equal_share_min_kw (smallest cap at which equal
    sharing delivers all energy), unmet_equal_share_at_lp_frac, kwh_clipped, ...

Usage:
    python -m acn.sensitivity                 # all scenarios (skips finished ones)
    python -m acn.sensitivity --only jpl_observed caltech_observed --force
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from acn.analysis import min_cap_equal_share, unmet_kwh
from acn.baselines import equal_share, uncontrolled
from acn.clean import load_clean
from acn.models import min_peak
from acn.rates import RATE_COL, estimate_car_rates
from acn.slots import P_MAX_KW, daily_instances, usable_sessions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sensitivity"
START, END = "2019-01-01", "2020-02-29"
AMPS_TO_KW = 208 / 1000


@dataclass(frozen=True)
class Scenario:
    site: str
    rates: str = "uniform"  # "uniform" (every car p_max) or "observed" (per-car estimate)
    p_max: float = P_MAX_KW
    r_min_kw: float = 0.0
    note: str = ""


SCENARIOS = {
    "jpl_base": Scenario("jpl", note="every car 6.656 kW (32 A)"),
    "jpl_observed": Scenario("jpl", "observed", note="per-car rate from observed charging"),
    "jpl_charger_5.0kW": Scenario("jpl", p_max=5.0, note="every car 5.0 kW"),
    "jpl_charger_3.3kW": Scenario("jpl", p_max=3.3, note="every car 3.3 kW"),
    "jpl_observed_rmin_6A": Scenario("jpl", "observed", r_min_kw=6 * AMPS_TO_KW, note="observed + min 6 A (1.25 kW)"),
    "jpl_observed_rmin_8A": Scenario("jpl", "observed", r_min_kw=8 * AMPS_TO_KW, note="observed + min 8 A (1.66 kW)"),
    "jpl_observed_rmin_16A": Scenario("jpl", "observed", r_min_kw=16 * AMPS_TO_KW, note="observed + min 16 A (3.33 kW)"),
    "caltech_base": Scenario("caltech", note="every car 6.656 kW"),
    "caltech_observed": Scenario("caltech", "observed", note="per-car rate from observed charging"),
}


def sessions_for(sc: Scenario) -> pd.DataFrame:
    df = usable_sessions(load_clean(sc.site))
    if sc.rates == "observed":
        lo, hi = (pd.Timestamp(x, tz="America/Los_Angeles") for x in (START, END))
        window = df[(df["connectionTime"] >= lo) & (df["connectionTime"] < hi + pd.Timedelta(days=2))]
        df = estimate_car_rates(window, sc.p_max)
    return df


def run_scenario(name: str, sc: Scenario) -> pd.DataFrame:
    df = sessions_for(sc)
    kw = {"rate_col": RATE_COL} if sc.rates == "observed" else {"p_max": sc.p_max}
    rows = []
    for inst in daily_instances(df, sc.site, START, END, **kw):
        if inst.n == 0:
            continue
        res = min_peak(inst, r_min_kw=sc.r_min_kw)
        if res.status != "Optimal":
            print(f"  {name} {inst.slot_starts[0]:%Y-%m-%d}: LP {res.status}")
            continue
        unc = uncontrolled(inst).sum(0).max()
        lp = res.R.sum(0).max()
        rows.append({
            "scenario": name, "site": sc.site, "date": inst.slot_starts[0].date(),
            "weekday": inst.slot_starts[0].day_name(), "n_sessions": inst.n,
            "energy_kwh": inst.energy.sum(), "kwh_clipped": inst.info["kwh_clipped"],
            "mean_car_rate_kw": inst.p_max.mean(),
            "peak_uncontrolled_kw": unc, "peak_lp_kw": lp,
            "peak_equal_share_min_kw": min_cap_equal_share(inst, hi=unc + 1e-6),
            "unmet_equal_share_at_lp_frac": unmet_kwh(inst, equal_share(inst, lp)) / inst.energy.sum(),
        })
    out = pd.DataFrame(rows)
    out["peak_reduction"] = 1 - out["peak_lp_kw"] / out["peak_uncontrolled_kw"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=list(SCENARIOS))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.only or SCENARIOS:
        f = OUT / f"{name}.csv"
        if f.exists() and not a.force:
            print(f"{name}: exists, skipping")
            continue
        t = time.perf_counter()
        out = run_scenario(name, SCENARIOS[name])
        out.to_csv(f, index=False)
        wk = out[~out["weekday"].isin(["Saturday", "Sunday"]) & (out["n_sessions"] >= 5)]
        print(f"{name}: {len(out)} days in {time.perf_counter() - t:.0f}s | weekday median: "
              f"unc {wk['peak_uncontrolled_kw'].median():.0f} kW, LP {wk['peak_lp_kw'].median():.0f} kW, "
              f"reduction {wk['peak_reduction'].median():.1%}, eq-share min cap "
              f"{wk['peak_equal_share_min_kw'].median():.0f} kW", flush=True)


if __name__ == "__main__":
    main()
