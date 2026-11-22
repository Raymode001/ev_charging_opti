"""Tests for analysis helpers, plotting smoke tests, and the online (MPC) controller."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from acn import analysis, plots
from acn.baselines import equal_share, uncontrolled
from acn.metrics import evaluate
from acn.models import min_peak
from acn.online import departure_estimates, simulate
from acn.slots import Instance, ns

TZ = "America/Los_Angeles"
START = pd.Timestamp("2019-06-03 03:00", tz=TZ)


def toy(avail, energy, p_max=4.0) -> Instance:
    avail = np.asarray(avail, float)
    slots = pd.date_range(START, periods=avail.shape[1], freq="15min")
    e = np.asarray(energy, float)
    return Instance("toy", slots, 0.25, np.arange(len(e)), e, e, avail, p_max,
                    np.zeros(avail.shape[1], int))


def times(inst: Instance, first: list[int], last: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Epoch-ns arrival at slot `first`, departure at end of slot `last`."""
    s = inst.slot_starts
    a = ns(pd.Series([s[i] for i in first]))
    d = ns(pd.Series([s[i] + pd.Timedelta(minutes=15) for i in last]))
    return a, d


# ---------- analysis ----------

def test_min_cap_equal_share_by_hand():
    # car0: 1 slot, needs 1 kWh (4 kW). car1: 4 slots, needs 1 kWh.
    # Equal sharing at cap C gives each min(limit, C/2) in slot 0, so car0 gets C/2 * 0.25 kWh.
    # Exact delivery needs C = 8; the search stops once unmet <= UNMET_TOL_KWH (0.01 kWh),
    # i.e. at C = 2 * (1 - 0.01) / 0.25 = 7.92.
    inst = toy([[1, 0, 0, 0], [1, 1, 1, 1]], [1.0, 1.0])
    c = analysis.min_cap_equal_share(inst, hi=uncontrolled(inst).sum(0).max() + 1e-6)
    assert c == pytest.approx(2 * (1 - analysis.UNMET_TOL_KWH) / 0.25, abs=1e-4)
    assert evaluate(inst, equal_share(inst, c))["unmet_kwh"] <= analysis.UNMET_TOL_KWH


def test_cap_sweep_monotone_and_zero_at_large_cap():
    inst = toy([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1]], [1.5, 2.0, 1.0])
    lp = min_peak(inst).R.sum(0).max()
    sw = analysis.cap_sweep({"d": inst}, {"d": lp}, np.array([1.0, 1.5, 2.0, 5.0]))
    assert (np.diff(sw["unmet_kwh"]) <= 1e-9).all()
    assert sw["unmet_kwh"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_occupancy_counts_fractional_presence():
    inst = toy([[1, 0.5], [0, 1]], [0.1, 0.1])
    np.testing.assert_allclose(analysis.occupancy(inst), [1.0, 1.5])


# ---------- plots (smoke: every figure renders) ----------

def test_all_plots_render():
    plots.style()
    inst = toy([[1, 1, 1, 0], [0, 1, 1, 1]], [1.0, 1.0])
    sched = {"lp": min_peak(inst).R, "uncontrolled": uncontrolled(inst), "equal_share": equal_share(inst, 4.0)}
    res = pd.DataFrame({
        "date": pd.date_range("2019-01-07", periods=10, freq="B").date,
        "weekday": [d.day_name() for d in pd.date_range("2019-01-07", periods=10, freq="B")],
        "n_sessions": np.arange(10, 20), "peak_uncontrolled_kw": np.linspace(200, 250, 10),
        "peak_lp_kw": np.linspace(70, 90, 10), "peak_equal_share_min_kw": np.linspace(120, 150, 10)})
    res["peak_reduction"] = 1 - res["peak_lp_kw"] / res["peak_uncontrolled_kw"]
    sweep = pd.DataFrame({"multiplier": np.repeat([1.0, 1.5], 3), "unmet_frac": [0.1, 0.08, 0.12, 0, 0, 0.01]})
    U = pd.DataFrame(np.maximum(0, 50 - np.arange(6)[None, :] * 10 + np.arange(4)[:, None]), columns=np.arange(6.0))
    figs = [
        plots.day_profile(inst, sched, analysis.occupancy(inst), "t"),
        plots.schedule_heatmaps(inst, sched, "t"),
        plots.peaks_over_time(res), plots.peak_distributions(res, res["peak_equal_share_min_kw"]),
        plots.tradeoff(sweep, res), plots.reduction_vs_sessions(res), plots.by_weekday(res),
        plots.unmet_curve_plot(U, 3.0), plots.cost_decomposition(U, 1.0, 2.0),
        plots.cap_vs_penalty(pd.Series([2.0, 4.0], index=[0.1, 10]), pd.Series([1.0, 3.0, 5.0])),
    ]
    assert all(isinstance(f, plt.Figure) for f in figs)
    plt.close("all")


# ---------- online MPC ----------

def test_mpc_with_true_departures_delivers_everything():
    inst = toy([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]], [2.0, 2.0])
    a, d = times(inst, [0, 2], [3, 5])
    res = simulate(inst, a, d, d.astype(float))
    assert res.unmet.sum() == pytest.approx(0.0, abs=1e-6)
    assert evaluate(inst, res.R)["over_rate_or_absent"] == 0


def test_mpc_never_beats_offline_peak():
    inst = toy([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], [2.0, 2.0, 3.0])
    a, d = times(inst, [0, 2, 0], [3, 5, 5])
    online = simulate(inst, a, d, d.astype(float)).R.sum(0).max()
    offline = min_peak(inst).R.sum(0).max()
    assert online >= offline - 1e-6


def test_optimistic_estimate_causes_unmet_and_min_progress_helps():
    # Car stays 2 slots but the estimate says 8: the MPC spreads charging thinly, then the car leaves.
    inst = toy([[1, 1, 0, 0, 0, 0, 0, 0]], [1.5])
    a, d = times(inst, [0], [1])
    late = ns(pd.Series([inst.slot_starts[0] + pd.Timedelta(hours=2)])).astype(float)
    plain = simulate(inst, a, d, late)
    hedged = simulate(inst, a, d, late, r_min_kw=3.0)
    assert plain.unmet.sum() > 0.5
    assert hedged.unmet.sum() < plain.unmet.sum()
    # Minimum progress: 3 kW for 0.5 h = 1.5 kWh, i.e. everything.
    assert hedged.unmet.sum() == pytest.approx(0.0, abs=1e-6)


def test_departure_estimates_conservative_is_earlier():
    t0 = pd.Timestamp("2019-06-03 08:00", tz=TZ)
    df = pd.DataFrame({
        "sessionID": list("abcd"),
        "connectionTime": [t0] * 4,
        "disconnectTime": [t0 + pd.Timedelta(hours=h) for h in (6, 8, 9, 10)],
        "requestedDeparture_first": [t0 + pd.Timedelta(hours=8)] * 3 + [pd.NaT],
    })
    drv = departure_estimates(df, "driver")
    con = departure_estimates(df, "conservative", q=0.0)
    assert drv["a"] == t0 + pd.Timedelta(hours=8)
    assert pd.notna(drv["d"])  # fallback: arrival + median dwell for that arrival hour
    assert con["a"] == drv["a"] - pd.Timedelta(hours=2)  # min error = 6 - 8 = -2 h
    assert departure_estimates(df, "none").isna().all()
