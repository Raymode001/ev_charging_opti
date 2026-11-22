"""Turn cleaned sessions into slotted scheduling instances.

An Instance covers one planning horizon (one operational day, or one week) split
into fixed-length slots on the true elapsed-time axis, so DST days have 92 or 100
15-minute slots, not 96.

avail[i, t] in [0, 1] is the fraction of slot t during which session i is plugged
in ("fractional availability"): no rounding of arrival/departure, so slotting can
never make a session infeasible.

Sessions not fully inside the horizon are dropped and counted. Energy targets are
kWhDelivered, clipped to what p_max could physically deliver over the dwell
(clips are counted; at 6.656 kW they are metering noise on JPL).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from acn.clean import TZ

# acnportal models every ACN Level-2 EVSE at 32 A, 208 V (acnsim/models/evse.py).
P_MAX_KW = 32 * 208 / 1000  # 6.656
DCFC_STATIONS = {"2-39-81-4550"}  # Caltech DC fast charger, present from 2020-11-18


@dataclass
class Instance:
    site: str
    slot_starts: pd.DatetimeIndex  # tz-aware local time, len T
    dt_h: float
    session_ids: np.ndarray  # (n,)
    energy: np.ndarray  # (n,) kWh target after clipping
    energy_raw: np.ndarray  # (n,) kWhDelivered before clipping
    avail: np.ndarray  # (n, T) fraction of each slot plugged in
    p_max: np.ndarray  # (n,) kW max per car (a scalar is broadcast in __post_init__)
    day_index: np.ndarray  # (T,) operational-day number of each slot (0 for a 1-day horizon)
    info: dict = field(default_factory=dict)  # drop/clip counts

    def __post_init__(self) -> None:
        self.p_max = np.broadcast_to(np.asarray(self.p_max, float), (len(self.session_ids),)).copy()

    @property
    def n(self) -> int:
        return len(self.session_ids)

    @property
    def T(self) -> int:
        return len(self.slot_starts)

    @property
    def n_days(self) -> int:
        return int(self.day_index.max()) + 1 if self.T else 0

    def max_power(self) -> np.ndarray:
        """(n, T) upper bound on average power per slot."""
        return self.avail * self.p_max[:, None]


def usable_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Rows fit for scheduling: no hard flag, not the DC fast charger, positive energy."""
    keep = df["ok"] & ~df["stationID"].isin(DCFC_STATIONS) & (df["kWhDelivered"] > 0)
    return df.loc[keep]


def day_starts(first: str | pd.Timestamp, last: str | pd.Timestamp, boundary_hour: int = 3) -> pd.DatetimeIndex:
    """Local operational-day starts at boundary_hour, for calendar dates first..last inclusive."""
    dates = pd.date_range(pd.Timestamp(first).date(), pd.Timestamp(last).date(), freq="D")
    # Localize each wall-clock time separately so DST shifts the elapsed length, not the boundary.
    return pd.DatetimeIndex([(d + pd.Timedelta(hours=boundary_hour)).tz_localize(TZ) for d in dates])


def ns(x: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Epoch nanoseconds; pandas 3 may store us/s resolution, so normalize explicitly."""
    idx = pd.DatetimeIndex(x)
    return idx.as_unit("ns").asi8


def availability(arrive: np.ndarray, depart: np.ndarray, slot_starts: pd.DatetimeIndex,
                 dt: pd.Timedelta) -> np.ndarray:
    """Fraction of each slot covered by [arrive, depart). Inputs: epoch ns (see ns())."""
    step = int(dt / pd.Timedelta(nanoseconds=1))
    s0 = ns(slot_starts)[None, :]
    overlap = np.minimum(depart[:, None], s0 + step) - np.maximum(arrive[:, None], s0)
    return np.clip(overlap, 0, None) / step


def build_instance(df: pd.DataFrame, site: str, start: pd.Timestamp, end: pd.Timestamp,
                   day_bounds: pd.DatetimeIndex | None = None, slot_minutes: int = 15,
                   p_max: float = P_MAX_KW, rate_col: str | None = None) -> Instance:
    """Instance for sessions fully inside [start, end). df: output of usable_sessions.

    rate_col: optional df column with a per-session max rate (kW); else every car gets p_max.
    """
    dt = pd.Timedelta(minutes=slot_minutes)
    slots = pd.date_range(start, end, freq=dt, inclusive="left")
    if (end - start) % dt:
        raise ValueError("horizon length must be a whole number of slots")

    arr, dep = df["connectionTime"], df["disconnectTime"]
    inside = (arr >= start) & (dep <= end)
    touching = (arr < end) & (dep > start)
    # Count a crossing session once: in the horizon where it arrives.
    dropped = (arr >= start) & (arr < end) & (dep > end)
    sub = df.loc[inside]

    avail = availability(ns(sub["connectionTime"]), ns(sub["disconnectTime"]), slots, dt)
    dt_h = slot_minutes / 60
    e_raw = sub["kWhDelivered"].to_numpy(float)
    rates = sub[rate_col].to_numpy(float) if rate_col else np.full(len(sub), p_max)
    e_cap = avail.sum(axis=1) * rates * dt_h
    energy = np.minimum(e_raw, e_cap)

    bounds = day_bounds if day_bounds is not None else pd.DatetimeIndex([start])
    day_index = np.searchsorted(ns(bounds), ns(slots), side="right") - 1

    info = {
        "n_touching": int(touching.sum()),
        "n_dropped_boundary": int(dropped.sum()),
        "n_clipped": int((e_raw > e_cap + 1e-9).sum()),
        "kwh_clipped": float((e_raw - energy).sum()),
    }
    return Instance(site, slots, dt_h, sub["sessionID"].to_numpy(), energy, e_raw, avail,
                    rates, day_index, info)


def daily_instances(df: pd.DataFrame, site: str, first: str, last: str, boundary_hour: int = 3,
                    **kw) -> Iterator[Instance]:
    """One instance per operational day [d boundary_hour, d+1 boundary_hour)."""
    starts = day_starts(first, pd.Timestamp(last) + pd.Timedelta(days=1), boundary_hour)
    for a, b in itertools.pairwise(starts):
        yield build_instance(df, site, a, b, **kw)


def weekly_instances(df: pd.DataFrame, site: str, first: str, last: str, boundary_hour: int = 3,
                     **kw) -> Iterator[Instance]:
    """One instance per Monday-to-Monday week; slots carry their operational-day index."""
    first_monday = pd.Timestamp(first) + pd.Timedelta(days=(-pd.Timestamp(first).dayofweek) % 7)
    starts = day_starts(first_monday, pd.Timestamp(last) + pd.Timedelta(days=1), boundary_hour)
    for k in range(0, len(starts) - 7, 7):
        week = starts[k:k + 8]
        yield build_instance(df, site, week[0], week[-1], day_bounds=week[:-1], **kw)
