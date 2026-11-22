"""Tests for per-car rates, rate estimation, and the two-stage stochastic model."""

import numpy as np
import pandas as pd
import pytest

from acn.baselines import uncontrolled
from acn.metrics import evaluate
from acn.models import min_peak
from acn.rates import RATE_COL, estimate_car_rates
from acn.slots import Instance
from acn.stochastic import (
    Prices,
    StageTwo,
    cost_at,
    extensive_form,
    solve_saa,
    unmet_curves,
    wait_and_see,
)

TZ = "America/Los_Angeles"


def toy(avail, energy, p_max, dt_h=0.25) -> Instance:
    avail = np.asarray(avail, float)
    slots = pd.date_range("2019-06-03 03:00", periods=avail.shape[1], freq="15min", tz=TZ)
    e = np.asarray(energy, float)
    return Instance("toy", slots, dt_h, np.arange(len(e)), e, e, avail, p_max,
                    np.zeros(avail.shape[1], int))


# ---------- per-car rates ----------

def test_scalar_p_max_is_broadcast():
    inst = toy([[1, 1], [1, 1]], [0.5, 0.5], 4.0)
    assert inst.p_max.tolist() == [4.0, 4.0]


def test_uncontrolled_uses_each_cars_rate():
    inst = toy([[1, 1, 1], [1, 1, 1]], [0.5, 0.5], [2.0, 4.0])
    R = uncontrolled(inst)
    np.testing.assert_allclose(R[:, 0], [2.0, 2.0])  # car1 needs only 0.5 kWh = 2 kW for one slot
    np.testing.assert_allclose(R[0], [2.0, 0, 0])


def test_lp_respects_slow_car():
    # Slow car (1 kW) needs 0.5 kWh over 2 slots: it must charge at 1 kW in both slots.
    inst = toy([[1, 1], [1, 1]], [0.5, 0.5], [1.0, 4.0])
    res = min_peak(inst)
    ev = evaluate(inst, res.R)
    assert ev["unmet_kwh"] == pytest.approx(0, abs=1e-6) and ev["over_rate_or_absent"] == 0
    assert ev["peak_kw"] == pytest.approx(2.0, abs=1e-6)  # 1 kW + 1 kW each slot


def test_r_min_above_slow_car_rate_stays_feasible():
    inst = toy([[1] * 4], [0.5], [1.0])
    assert min_peak(inst, r_min_kw=3.0).status == "Optimal"


# ---------- rate estimation ----------

def test_estimate_car_rates_sources_and_bounds():
    df = pd.DataFrame({
        "userID": ["a", "a", "a", "b", None, None],
        "avg_kw_charging": [3.0, 3.2, 3.4, 5.0, 6.0, np.nan],
        "charging_h": [2, 2, 2, 2, 2, np.nan],
        "kWhDelivered": [6, 6, 6, 10, 12, 1],
        "dwell_h": [8, 8, 8, 8, 8, 8],
    })
    out = estimate_car_rates(df, p_cap=6.656)
    assert out["rate_source"].tolist() == ["user"] * 3 + ["session", "session", "median"]
    assert out.loc[0, RATE_COL] == pytest.approx(pd.Series([3.0, 3.2, 3.4]).quantile(0.9))
    assert out.loc[3, RATE_COL] == pytest.approx(5.0)
    assert (out[RATE_COL] <= 6.656).all()
    assert (out[RATE_COL] >= out["kWhDelivered"] / out["dwell_h"] - 1e-12).all()


# ---------- two-stage ----------

def two_days():
    # Day A: one car, 1 kWh over 2 slots at 4 kW max -> optimal peak 2 kW.
    # Day B: two cars, 1 kWh each over 2 slots -> optimal peak 4 kW.
    return {"A": toy([[1, 1]], [1.0], 4.0), "B": toy([[1, 1], [1, 1]], [1.0, 1.0], 4.0)}


def test_stage_two_unmet_by_hand():
    st = StageTwo(two_days()["B"])
    assert st.unmet(0) == pytest.approx(2.0)
    assert st.unmet(2) == pytest.approx(1.0)  # 2 kW x 2 slots x 0.25 h = 1 kWh delivered
    assert st.unmet(4) == pytest.approx(0.0)
    assert st.unmet(10) == pytest.approx(0.0)


def test_grid_saa_matches_extensive_form():
    inst = two_days()
    peaks = {"A": 2.0, "B": 4.0}
    grid = np.arange(0, 4.01, 0.05)
    U = unmet_curves(inst, grid, peaks)
    for lam in [0.1, 1.0, 10.0]:
        pr = Prices(demand_charge_month=21.0, lam=lam)  # c = $1 per kW-day
        _Cg, fg = solve_saa(U, pr)
        Ce, fe = extensive_form(inst, pr)
        assert fg == pytest.approx(fe, abs=0.05)  # within one grid step of cost
        assert cost_at(U, Ce, pr) == pytest.approx(fe, abs=0.05)


def test_wait_and_see_is_a_lower_bound():
    inst = two_days()
    U = unmet_curves(inst, np.arange(0, 4.01, 0.05), {"A": 2.0, "B": 4.0})
    pr = Prices(21.0, 5.0)
    assert wait_and_see(U, pr) <= solve_saa(U, pr)[1] + 1e-9


def test_saa_cap_rises_with_penalty():
    inst = two_days()
    U = unmet_curves(inst, np.arange(0, 4.01, 0.05), {"A": 2.0, "B": 4.0})
    caps = [solve_saa(U, Prices(21.0, lam))[0] for lam in [0.1, 2.0, 100.0]]
    assert caps == sorted(caps) and caps[-1] == pytest.approx(4.0)
