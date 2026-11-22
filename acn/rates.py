"""Per-car maximum charging rate estimated from observed charging.

The EVSE allows 6.656 kW, but many cars accept less: their on-board charger is the
bottleneck (JPL per-user rates are bimodal, ~3.3 kW and ~6 kW). We estimate each car's
max rate from what it actually drew:

  1. its user's 90th-percentile session rate (kWhDelivered / charging hours), if the
     user has >= MIN_USER_SESSIONS sessions with a rate. A high quantile, because ACN's
     own scheduler sometimes throttled cars, so a single session underestimates capability;
  2. else the session's own observed rate;
  3. else the median of the estimates above (missing doneChargingTime, no user history);
  4. never below kWhDelivered / dwell (the car demonstrably received that), never above the EVSE limit.

Caveat: the estimate uses each user's sessions across the whole window, i.e. it looks
ahead in time. Fine for offline sensitivity analysis; an online controller would use
only past sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from acn.slots import P_MAX_KW

RATE_COL = "p_obs_kw"
QUANTILE = 0.9
MIN_USER_SESSIONS = 3


def estimate_car_rates(df: pd.DataFrame, p_cap: float = P_MAX_KW) -> pd.DataFrame:
    """Return df with RATE_COL (kW) and rate_source ('user', 'session', 'median') added."""
    df = df.copy()
    sess = df["avg_kw_charging"].where(df["charging_h"] > 0).clip(upper=p_cap)
    g = sess.groupby(df["userID"])
    user_q = g.transform(lambda s: s.quantile(QUANTILE) if s.notna().sum() >= MIN_USER_SESSIONS else np.nan)

    est = user_q.copy()
    source = pd.Series(np.where(est.notna(), "user", None), index=df.index, dtype=object)
    use_sess = est.isna() & sess.notna()
    est[use_sess] = sess[use_sess]
    source[use_sess] = "session"
    fill = est.median()
    source[est.isna()] = "median"
    est = est.fillna(fill)

    floor = df["kWhDelivered"] / df["dwell_h"]
    df[RATE_COL] = np.clip(np.maximum(est, floor), None, p_cap)
    df["rate_source"] = source
    return df
