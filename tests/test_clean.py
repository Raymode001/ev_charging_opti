"""Unit tests for acn.clean and acn.fetch flattening, on synthetic sessions."""

from pathlib import Path

import pandas as pd
import pytest

from acn import clean
from acn.fetch import flatten, to_frame


def session(sid="s1", station="A", conn="2019-06-03 15:00:00", disc="2019-06-03 23:00:00",
            done="2019-06-03 18:00:00", kwh=10.0, **extra) -> dict:
    """One raw row with UTC times, as acn.fetch writes them."""
    return {"sessionID": sid, "siteID": "0002", "clusterID": "0039", "stationID": station,
            "spaceID": "X", "userID": None, "connectionTime": conn, "disconnectTime": disc,
            "doneChargingTime": done, "kWhDelivered": kwh, "timezone": clean.TZ, **extra}


def run(rows: list[dict], tmp_path: Path, max_kw: float = 7.0) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    f = tmp_path / "s.csv"
    df.to_csv(f, index=False)
    return clean.clean(f, max_kw)


def test_utc_to_local_summer(tmp_path):
    df = run([session(conn="2019-06-03 15:00:00")], tmp_path)  # PDT = UTC-7
    assert df.loc[0, "connectionTime"].hour == 8
    assert df.loc[0, "local_date"] == pd.Timestamp("2019-06-03").date()


def test_utc_to_local_winter(tmp_path):
    df = run([session(conn="2019-12-02 16:00:00", disc="2019-12-02 20:00:00",
                      done="2019-12-02 17:00:00")], tmp_path)  # PST = UTC-8
    assert df.loc[0, "connectionTime"].hour == 8


def test_late_utc_evening_is_previous_local_day(tmp_path):
    # 03:00 UTC on Jun 4 is 20:00 local on Jun 3: slotting must use the local date.
    df = run([session(conn="2019-06-04 03:00:00", disc="2019-06-04 05:00:00",
                      done="2019-06-04 04:00:00")], tmp_path)
    assert df.loc[0, "local_date"] == pd.Timestamp("2019-06-03").date()


def test_dwell_across_fall_back_uses_elapsed_time(tmp_path):
    # 2019-11-03 fall-back: 00:30 PDT -> 03:30 PST is 4 elapsed hours, not 3 wall-clock hours.
    df = run([session(conn="2019-11-03 07:30:00", disc="2019-11-03 11:30:00",
                      done="2019-11-03 08:30:00", kwh=1.0)], tmp_path)
    assert df.loc[0, "dwell_h"] == pytest.approx(4.0)
    assert df.loc[0, "flag_dst_day"]


def test_dst_dates_2019():
    d = clean.dst_dates(range(2019, 2020))
    assert {pd.Timestamp("2019-03-10").date(), pd.Timestamp("2019-11-03").date()} == d


def test_nonpositive_dwell_is_hard_flag(tmp_path):
    df = run([session(disc="2019-06-03 15:00:00", done=None)], tmp_path)
    assert df.loc[0, "flag_nonpositive_dwell"] and not df.loc[0, "ok"]


def test_negative_dwell(tmp_path):
    df = run([session(disc="2019-06-03 14:00:00", done=None)], tmp_path)
    assert df.loc[0, "dwell_h"] < 0 and df.loc[0, "flag_nonpositive_dwell"]


def test_zero_energy_flag(tmp_path):
    df = run([session(kwh=0.0)], tmp_path)
    assert df.loc[0, "flag_nonpositive_energy"] and not df.loc[0, "ok"]


def test_infeasible_energy_vs_max_rate(tmp_path):
    # 2 h dwell at 7 kW can deliver at most 14 kWh.
    rows = [session("ok", kwh=14.0, disc="2019-06-03 17:00:00", done="2019-06-03 17:00:00"),
            session("bad", station="B", kwh=14.5, disc="2019-06-03 17:00:00", done="2019-06-03 17:00:00")]
    df = run(rows, tmp_path).set_index("sessionID")
    assert not df.loc["ok", "flag_infeasible_at_max_kw"]
    assert df.loc["bad", "flag_infeasible_at_max_kw"]


def test_duplicates(tmp_path):
    rows = [session("s1"), session("s1"), session("s2", station="B"),
            session("s3", station="B")]  # s2/s3: same station + same connection time
    df = run(rows, tmp_path)
    assert df["flag_dup_session_id"].tolist() == [True, True, False, False]
    # s1 copies also share station A + connection time, so all four are flagged.
    assert df["flag_dup_station_time"].tolist() == [True, True, True, True]


def test_station_overlap(tmp_path):
    rows = [session("s1", conn="2019-06-03 15:00:00", disc="2019-06-03 20:00:00"),
            session("s2", conn="2019-06-03 19:00:00", disc="2019-06-03 23:00:00",
                    done="2019-06-03 21:00:00"),
            session("s3", conn="2019-06-04 15:00:00", disc="2019-06-04 20:00:00",
                    done="2019-06-04 16:00:00")]
    df = run(rows, tmp_path).set_index("sessionID")
    assert df["flag_station_overlap"].to_dict() == {"s1": False, "s2": True, "s3": False}


def test_done_after_disconnect_tolerance(tmp_path):
    rows = [session("small", done="2019-06-03 23:01:00"),  # within 2 min tolerance
            session("big", station="B", done="2019-06-03 23:30:00")]
    df = run(rows, tmp_path).set_index("sessionID")
    assert not df.loc["small", "flag_done_after_disconnect"]
    assert df.loc["big", "flag_done_after_disconnect"]


def test_missing_done_is_soft(tmp_path):
    df = run([session(done=None)], tmp_path)
    assert df.loc[0, "flag_missing_done"] and df.loc[0, "ok"]


def test_daily_counts_include_zero_days(tmp_path):
    rows = [session("a", conn="2019-06-03 15:00:00"),
            session("b", station="B", conn="2019-06-05 15:00:00", disc="2019-06-05 20:00:00",
                    done="2019-06-05 16:00:00")]
    c = clean.daily_counts(run(rows, tmp_path))
    assert c.tolist() == [1, 0, 1]


def test_zero_runs():
    days = pd.date_range("2020-01-01", periods=12).date
    c = pd.Series([1] + [0] * 8 + [1, 0, 0], index=days)
    assert clean.zero_runs(c, min_len=7) == [(days[1], days[8], 8)]


def test_flatten_keeps_first_and_last_user_input():
    item = {"sessionID": "x", "userInputs": [
        {"modifiedAt": "Mon, 10 Feb 2020 21:03:35 GMT", "kWhRequested": 72.0,
         "requestedDeparture": "Tue, 11 Feb 2020 00:52:47 GMT"},
        {"modifiedAt": "Mon, 10 Feb 2020 20:55:44 GMT", "kWhRequested": 20.0,
         "requestedDeparture": "Mon, 10 Feb 2020 21:53:47 GMT"},
    ]}
    row = flatten(item)
    assert row["kWhRequested"] == 72.0 and row["kWhRequested_first"] == 20.0
    assert row["n_user_inputs"] == 2


def test_to_frame_parses_api_times_as_utc():
    df = to_frame([{"sessionID": "x", "connectionTime": "Wed, 25 Apr 2018 11:08:04 GMT",
                    "disconnectTime": None, "doneChargingTime": None, "userInputs": None}])
    assert str(df.loc[0, "connectionTime"].tz) == "UTC"
    assert df.loc[0, "connectionTime"].hour == 11


def test_cluster_id_normalized(tmp_path):
    rows = [session("a", clusterID="39"), session("b", station="B", clusterID="0039")]
    df = run(rows, tmp_path)
    assert set(df["clusterID"]) == {"0039"}


def test_load_clean_roundtrip(tmp_path, monkeypatch):
    df = run([session(conn="2019-06-04 03:00:00", disc="2019-06-04 05:00:00",
                      done="2019-06-04 04:00:00")], tmp_path)
    monkeypatch.setattr(clean, "DATA", tmp_path)
    df.to_csv(tmp_path / "x_clean.csv", index=False)
    back = clean.load_clean("x")
    assert back.loc[0, "local_date"] == pd.Timestamp("2019-06-03").date()
    assert str(back["connectionTime"].dt.tz) == clean.TZ
