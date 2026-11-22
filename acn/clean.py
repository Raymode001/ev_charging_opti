"""Clean and validate ACN session CSVs produced by acn.fetch.

Nothing is dropped here: every row gets boolean flag_* columns plus `ok`
(no hard-error flag). Deciding what to exclude is a modeling choice left to you.

Usage:
    python -m acn.clean --site jpl            # writes data/jpl_clean.csv, prints report
    python -m acn.clean --site all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TZ = "America/Los_Angeles"
TIME_COLS = ["connectionTime", "disconnectTime", "doneChargingTime", "requestedDeparture",
             "requestedDeparture_first", "modifiedAt", "modifiedAt_first"]

# Cited assumption, not a measured fact: ACN Level-2 EVSEs ~32 A at 208/240 V ≈ 6.6-7.7 kW.
# Used only to flag physically implausible sessions; override with --max-kw.
DEFAULT_MAX_KW = 7.0
DONE_AFTER_DISCONNECT_TOL = pd.Timedelta(minutes=2)

# Hard errors make `ok` False; soft flags are informational.
HARD_FLAGS = ["flag_dup_session_id", "flag_dup_station_time", "flag_nonpositive_dwell",
              "flag_nonpositive_energy", "flag_infeasible_at_max_kw"]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"siteID": str, "clusterID": str, "stationID": str,
                                  "spaceID": str, "userID": str})
    # API returns Caltech's cluster as both '39' (Oct 2018) and '0039'.
    df["clusterID"] = df["clusterID"].str.zfill(4)
    for c in TIME_COLS:
        if c in df:
            # Parse as UTC first: offsets are fixed in the file, so DST is never ambiguous.
            df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(TZ)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hours = lambda td: td.dt.total_seconds() / 3600
    df["dwell_h"] = hours(df["disconnectTime"] - df["connectionTime"])
    df["charging_h"] = hours(df["doneChargingTime"] - df["connectionTime"])
    df["idle_h"] = hours(df["disconnectTime"] - df["doneChargingTime"])
    # Average power over the whole stay = lower bound on the peak rate the car needed.
    df["avg_kw_dwell"] = df["kWhDelivered"] / df["dwell_h"].where(df["dwell_h"] > 0)
    df["avg_kw_charging"] = df["kWhDelivered"] / df["charging_h"].where(df["charging_h"] > 0)
    df["local_date"] = df["connectionTime"].dt.date
    df["weekday"] = df["connectionTime"].dt.day_name()
    df["is_weekend"] = df["connectionTime"].dt.dayofweek >= 5
    df["arrival_hour"] = df["connectionTime"].dt.hour + df["connectionTime"].dt.minute / 60
    df["app_user"] = df.get("kWhRequested", pd.Series(np.nan, index=df.index)).notna()
    if "requestedDeparture" in df:
        df["departure_error_h"] = hours(df["disconnectTime"] - df["requestedDeparture"])
    if "requestedDeparture_first" in df:
        df["departure_error_first_h"] = hours(df["disconnectTime"] - df["requestedDeparture_first"])
    return df


def station_overlaps(df: pd.DataFrame) -> pd.Series:
    """True where a session overlaps the previous one on the same station (one car per EVSE)."""
    d = df.sort_values(["stationID", "connectionTime"])
    prev_end = d.groupby("stationID")["disconnectTime"].shift()
    return (d["connectionTime"] < prev_end).reindex(df.index, fill_value=False)


def dst_dates(years: range) -> set:
    """Local dates on which a US DST transition occurs (23 h or 25 h days)."""
    days = pd.date_range(f"{years.start}-01-01", f"{years.stop}-01-01", freq="D", tz=TZ)
    lengths = (days[1:] - days[:-1]).total_seconds() / 3600
    return {d.date() for d, h in zip(days[:-1], lengths) if h != 24}


def add_flags(df: pd.DataFrame, max_kw: float = DEFAULT_MAX_KW) -> pd.DataFrame:
    df = df.copy()
    df["flag_dup_session_id"] = df["sessionID"].duplicated(keep=False)
    df["flag_dup_station_time"] = df.duplicated(["stationID", "connectionTime"], keep=False)
    df["flag_station_overlap"] = station_overlaps(df)
    df["flag_nonpositive_dwell"] = ~(df["dwell_h"] > 0)
    df["flag_nonpositive_energy"] = ~(df["kWhDelivered"] > 0)
    df["flag_tiny_energy"] = df["kWhDelivered"].between(0, 0.5, inclusive="right")
    df["flag_missing_done"] = df["doneChargingTime"].isna()
    df["flag_done_before_connect"] = df["doneChargingTime"] < df["connectionTime"]
    df["flag_done_after_disconnect"] = (
        df["doneChargingTime"] > df["disconnectTime"] + DONE_AFTER_DISCONNECT_TOL)
    # Energy cannot exceed dwell * max rate; violators are data issues (or max_kw is wrong).
    df["flag_infeasible_at_max_kw"] = df["kWhDelivered"] > df["dwell_h"].clip(lower=0) * max_kw + 1e-9
    df["flag_overnight"] = df["disconnectTime"].dt.date != df["connectionTime"].dt.date
    df["flag_long_dwell_24h"] = df["dwell_h"] > 24
    years = range(df["connectionTime"].dt.year.min(), df["connectionTime"].dt.year.max() + 1)
    dst = dst_dates(years)
    df["flag_dst_day"] = df["local_date"].isin(dst) | df["disconnectTime"].dt.date.isin(dst)
    if "requestedDeparture" in df:
        df["flag_request_before_connect"] = df["requestedDeparture"] < df["connectionTime"]
    if "kWhRequested" in df:
        df["flag_request_nonpositive"] = df["kWhRequested"] <= 0
    df["ok"] = ~df[HARD_FLAGS].any(axis=1)
    return df


def load_clean(site: str) -> pd.DataFrame:
    """Read data/<site>_clean.csv back with local-time datetimes and real date objects.

    A plain pd.read_csv returns times and local_date as strings, which silently
    breaks date matching (e.g. reindexing on a date range yields all zeros).
    """
    df = load_raw(DATA / f"{site}_clean.csv")
    df["local_date"] = df["connectionTime"].dt.date
    return df


def clean(path: Path, max_kw: float = DEFAULT_MAX_KW) -> pd.DataFrame:
    return add_flags(add_derived(load_raw(path)), max_kw)


def daily_counts(df: pd.DataFrame) -> pd.Series:
    """Sessions per local calendar day, including zero-session days in the covered range."""
    days = pd.date_range(df["local_date"].min(), df["local_date"].max(), freq="D").date
    return df.groupby("local_date").size().reindex(days, fill_value=0)


def zero_runs(counts: pd.Series, min_len: int = 7) -> list[tuple]:
    """Runs of >= min_len consecutive zero-session days (outages / missing data)."""
    runs, start = [], None
    for day, n in list(counts.items()) + [(None, 1)]:
        if n == 0 and start is None:
            start, length = day, 0
        if n == 0:
            length += 1
            end = day
        elif start is not None:
            if length >= min_len:
                runs.append((start, end, length))
            start = None
    return runs


def report(df: pd.DataFrame, site: str) -> None:
    pct = lambda x: f"{100 * x:.1f}%"
    print(f"\n===== {site} =====")
    print(f"sessions: {len(df)} | local dates {df['local_date'].min()} -> {df['local_date'].max()}")
    print(f"unique stationID: {df['stationID'].nunique()} | spaceID: {df['spaceID'].nunique()} "
          f"| clusterID: {sorted(df['clusterID'].dropna().unique())}")
    print(f"app users (kWhRequested present): {pct(df['app_user'].mean())}")

    c = daily_counts(df)
    wk = pd.Series([pd.Timestamp(d).dayofweek < 5 for d in c.index], index=c.index)
    print(f"sessions/day  weekday: mean {c[wk].mean():.1f}, median {c[wk].median():.0f}, max {c.max()} | "
          f"weekend: mean {c[~wk].mean():.1f}")
    print(f"zero-session weekdays: {(c[wk] == 0).sum()} of {wk.sum()}")
    for a, b, n in zero_runs(c):
        print(f"  gap: {a} -> {b} ({n} days with no sessions)")
    monthly = df.groupby(df["connectionTime"].dt.strftime("%Y-%m")).size()
    print("sessions/month:", " ".join(f"{m}:{n}" for m, n in monthly.items()))

    cols = ["kWhDelivered", "dwell_h", "charging_h", "idle_h", "avg_kw_dwell", "avg_kw_charging"]
    print(df[cols].describe(percentiles=[0.01, 0.5, 0.99]).round(2).to_string())
    if df["app_user"].any():
        a = df[df["app_user"]]
        print(f"app users: departure error (actual - requested, h) median "
              f"{a['departure_error_h'].median():.2f}, IQR "
              f"[{a['departure_error_h'].quantile(.25):.2f}, {a['departure_error_h'].quantile(.75):.2f}]; "
              f"delivered/requested kWh median {(a['kWhDelivered'] / a['kWhRequested']).median():.2f}")

    flags = [c for c in df if c.startswith("flag_")]
    print("flags (count):")
    for f in flags:
        n = int(df[f].sum())
        if n:
            print(f"  {'HARD ' if f in HARD_FLAGS else '     '}{f}: {n} ({pct(n / len(df))})")
    print(f"ok (no hard flag): {int(df['ok'].sum())} / {len(df)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", choices=("caltech", "jpl", "office001", "all"), required=True)
    p.add_argument("--max-kw", type=float, default=DEFAULT_MAX_KW)
    a = p.parse_args()
    for site in ("caltech", "jpl", "office001") if a.site == "all" else (a.site,):
        df = clean(DATA / f"{site}_sessions.csv", a.max_kw)
        df.to_csv(DATA / f"{site}_clean.csv", index=False)
        report(df, site)


if __name__ == "__main__":
    main()
