"""Tests for slotting, baselines, the peak LP and metrics, on hand-checkable instances."""

import numpy as np
import pandas as pd
import pytest

from acn.baselines import equal_share, uncontrolled, water_fill
from acn.metrics import evaluate
from acn.models import min_peak
from acn.slots import (
    P_MAX_KW,
    Instance,
    availability,
    build_instance,
    daily_instances,
    day_starts,
    ns,
    weekly_instances,
)

TZ = "America/Los_Angeles"


def frame(rows):
    """rows: (id, local arrive, local depart, kWh)."""
    df = pd.DataFrame(rows, columns=["sessionID", "connectionTime", "disconnectTime", "kWhDelivered"])
    for c in ["connectionTime", "disconnectTime"]:
        df[c] = pd.to_datetime(df[c]).dt.tz_localize(TZ)
    return df


def toy(avail, energy, p_max=4.0, dt_h=0.25, day_index=None) -> Instance:
    avail = np.asarray(avail, float)
    T = avail.shape[1]
    slots = pd.date_range("2019-06-03 03:00", periods=T, freq="15min", tz=TZ)
    return Instance("toy", slots, dt_h, np.arange(len(energy)), np.asarray(energy, float),
                    np.asarray(energy, float), avail, p_max,
                    np.zeros(T, int) if day_index is None else np.asarray(day_index))


# ---------- slotting ----------

def test_partial_slot_fractions_sum_to_dwell():
    slots = pd.date_range("2019-06-03 08:00", periods=4, freq="15min", tz=TZ)
    a = ns(pd.Series(pd.to_datetime(["2019-06-03 08:05"]).tz_localize(TZ)))
    b = ns(pd.Series(pd.to_datetime(["2019-06-03 08:40"]).tz_localize(TZ)))
    f = availability(a, b, slots, pd.Timedelta(minutes=15))[0]
    np.testing.assert_allclose(f, [10 / 15, 1, 10 / 15, 0])
    assert f.sum() * 0.25 == pytest.approx(35 / 60)


@pytest.mark.parametrize("day,slots", [("2019-03-09", 92), ("2019-11-02", 100), ("2019-06-03", 96)])
def test_dst_operational_day_lengths(day, slots):
    # The 03:00-03:00 day that contains a DST switch is 23 h (spring) or 25 h (fall).
    df = frame([("x", f"{day} 09:00", f"{day} 10:00", 1.0)])
    inst = next(daily_instances(df, "t", day, day))
    assert inst.T == slots


def test_boundary_crossing_session_dropped_and_counted():
    df = frame([("in", "2019-06-03 08:00", "2019-06-03 17:00", 10.0),
                ("cross", "2019-06-03 22:00", "2019-06-04 06:00", 8.0)])
    s, e = day_starts("2019-06-03", "2019-06-04")
    inst = build_instance(df, "t", s, e)
    assert list(inst.session_ids) == ["in"]
    assert inst.info["n_dropped_boundary"] == 1


def test_week_keeps_overnight_session_and_labels_days():
    df = frame([("cross", "2019-06-03 22:00", "2019-06-04 06:00", 8.0)])  # Monday night
    w = next(weekly_instances(df, "t", "2019-06-03", "2019-06-09"))
    assert list(w.session_ids) == ["cross"] and w.n_days == 7
    assert w.day_index[0] == 0 and w.day_index[-1] == 6


def test_energy_clipped_to_physical_max():
    df = frame([("x", "2019-06-03 08:00", "2019-06-03 09:00", 10.0)])  # 1 h * 6.656 kW max
    s, e = day_starts("2019-06-03", "2019-06-04")
    inst = build_instance(df, "t", s, e)
    assert inst.energy[0] == pytest.approx(P_MAX_KW) and inst.info["n_clipped"] == 1


# ---------- baselines ----------

def test_water_fill():
    np.testing.assert_allclose(water_fill(np.array([1.0, 5.0, 5.0]), 7.0), [1, 3, 3])
    np.testing.assert_allclose(water_fill(np.array([1.0, 2.0]), 10.0), [1, 2])
    assert water_fill(np.array([3.0, 3.0, 3.0]), 6.0).sum() == pytest.approx(6.0)


def test_uncontrolled_front_loads_and_meets_energy():
    inst = toy([[1, 1, 1, 1]], [1.5])  # 4 kW max, 0.25 h slots -> 1 kWh per full slot
    R = uncontrolled(inst)
    np.testing.assert_allclose(R[0], [4, 2, 0, 0])
    assert evaluate(inst, R)["unmet_kwh"] == pytest.approx(0)


def test_equal_share_respects_cap_and_redistributes():
    inst = toy([[1, 1], [1, 1]], [0.25, 2.0])  # car0 needs only 1 kW for one slot
    R = equal_share(inst, cap_kw=5.0)
    np.testing.assert_allclose(R[:, 0], [1, 4])  # car0's unused share goes to car1 (max 4)
    assert (R.sum(axis=0) <= 5 + 1e-9).all()


# ---------- LP ----------

def test_lp_hand_example():
    # car0 present 1 h, needs 2 kWh; car1 present 2 h, needs 2 kWh.
    # Optimum: car0 at 2 kW in hour 1, car1 at 2 kW in hour 2 -> peak 2 kW.
    inst = toy([[1] * 4 + [0] * 4, [1] * 8], [2.0, 2.0])
    res = min_peak(inst)
    assert res.status == "Optimal"
    assert evaluate(inst, res.R)["peak_kw"] == pytest.approx(2.0, abs=1e-6)


def test_lp_meets_energy_never_worse_than_uncontrolled():
    rng = np.random.default_rng(0)
    n, T = 15, 40
    avail = np.zeros((n, T))
    for i in range(n):
        a = rng.integers(0, 20)
        avail[i, a:a + rng.integers(4, 20)] = 1
    energy = avail.sum(1) * 4 * 0.25 * rng.uniform(0.2, 0.9, n)
    inst = toy(avail, energy)
    res, unc = evaluate(inst, min_peak(inst).R), evaluate(inst, uncontrolled(inst))
    assert res["unmet_kwh"] == pytest.approx(0, abs=1e-5)
    assert res["over_rate_or_absent"] == 0 and res["neg_rate"] == 0
    assert res["peak_kw"] <= unc["peak_kw"] + 1e-6


def test_daily_vs_max_objective_on_two_days():
    # Day 0 needs a 4 kW peak; day 1 needs 1 kW. Daily objective keeps day 1 at 1 kW.
    inst = toy([[1, 1, 0, 0], [0, 0, 1, 1]], [2.0, 0.5], day_index=[0, 0, 1, 1])
    d = min_peak(inst, "daily")
    m = min_peak(inst, "max")
    np.testing.assert_allclose(d.peaks, [4, 1], atol=1e-6)
    assert m.peaks.max() == pytest.approx(4, abs=1e-6)
    np.testing.assert_allclose(m.peaks, [4, 1], atol=1e-3)  # eps tie-break


def test_min_progress_protects_early_leaver():
    # Without r_min, the LP may postpone car0 entirely; with r_min it must keep pace.
    inst = toy([[1] * 8, [1] * 4 + [0] * 4], [2.0, 1.0])
    res = min_peak(inst, r_min_kw=1.0)
    cum = np.cumsum(res.R[0]) * 0.25
    need = np.minimum(2.0, 1.0 * 0.25 * np.arange(1, 9))
    assert (cum >= need - 1e-6).all()


def test_smooth_keeps_peak_and_flattens_cars():
    # Two cars, both present 4 slots, 1 kWh each: optimal peak 2 kW; smooth = 1 kW each, every slot.
    inst = toy([[1, 1, 1, 1], [1, 1, 1, 1]], [1.0, 1.0])
    plain, smooth = min_peak(inst), min_peak(inst, smooth=True)
    assert smooth.R.sum(0).max() == pytest.approx(plain.R.sum(0).max(), abs=1e-5)
    np.testing.assert_allclose(smooth.R, 1.0, atol=1e-5)
