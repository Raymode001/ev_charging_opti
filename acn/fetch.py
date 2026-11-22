"""Download ACN-Data session metadata (no time series) for one site.

Raw API pages are cached per calendar month (UTC) under data/raw/<site>/YYYY-MM.json
so interrupted runs resume. Then all months are flattened into data/<site>_sessions.csv
with times kept as UTC ISO strings; local-time conversion happens in acn.clean.

Usage:
    python -m acn.fetch --site jpl                 # full site history
    python -m acn.fetch --site caltech --start 2019-01-01 --end 2019-07-01

Token: API_TOKEN in the environment or in .env at the project root.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from email.utils import format_datetime
from itertools import pairwise
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

BASE = "https://ev.caltech.edu/api/v1/"
SITES = ("caltech", "jpl", "office001")
PAGE_SIZE = 100  # server caps max_results at 100
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
API_TIME_FMT = "%a, %d %b %Y %H:%M:%S GMT"

SESSION_FIELDS = [
    "sessionID", "siteID", "clusterID", "stationID", "spaceID", "userID",
    "connectionTime", "disconnectTime", "doneChargingTime", "kWhDelivered", "timezone",
]
INPUT_FIELDS = [
    "kWhRequested", "milesRequested", "WhPerMile", "minutesAvailable",
    "requestedDeparture", "modifiedAt", "paymentRequired",
]
TIME_FIELDS = ["connectionTime", "disconnectTime", "doneChargingTime",
               "requestedDeparture", "modifiedAt", "requestedDeparture_first", "modifiedAt_first"]


def load_token() -> str:
    if tok := os.environ.get("API_TOKEN"):
        return tok
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, val = line.partition("=")
            if key.strip() == "API_TOKEN" and val.strip():
                return val.strip().strip('"').strip("'")
    raise SystemExit("API_TOKEN not set (environment or .env).")


def make_session() -> requests.Session:
    s = requests.Session()
    s.auth = (load_token(), "")  # token as username, blank password
    return s


def get_json(s: requests.Session, url: str, params: dict | None = None, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = s.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            msg = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            msg = repr(e)
        print(f"  {msg}; retry {attempt + 1}/{retries}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def rfc1123(ts: pd.Timestamp) -> str:
    return format_datetime(ts.tz_convert("UTC").to_pydatetime(), usegmt=True)


def fetch_range(s: requests.Session, site: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """All sessions with start <= connectionTime < end, oldest first."""
    where = f'connectionTime>="{rfc1123(start)}" and connectionTime<"{rfc1123(end)}"'
    page = get_json(s, urljoin(BASE, f"sessions/{site}"),
                    {"where": where, "sort": "connectionTime", "max_results": PAGE_SIZE})
    expected = page["_meta"]["total"]
    items: list[dict] = []
    while True:
        items.extend(page.get("_items", []))
        nxt = page.get("_links", {}).get("next")
        if not nxt:
            break
        page = get_json(s, urljoin(BASE, nxt["href"]))
        time.sleep(0.1)
    if len(items) != expected:
        raise RuntimeError(f"{site} {start:%Y-%m}: got {len(items)}, server says {expected}")
    return items


def site_bounds(s: requests.Session, site: str) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """First/last connectionTime and total count. Note: '-field' sorts DESCENDING."""
    url = urljoin(BASE, f"sessions/{site}")
    first = get_json(s, url, {"sort": "connectionTime", "max_results": 1})
    last = get_json(s, url, {"sort": "-connectionTime", "max_results": 1})
    parse = lambda p: pd.to_datetime(p["_items"][0]["connectionTime"], format=API_TIME_FMT, utc=True)
    return parse(first), parse(last), first["_meta"]["total"]


def flatten(item: dict) -> dict:
    row = {k: item.get(k) for k in SESSION_FIELDS}
    inputs = item.get("userInputs") or []
    row["n_user_inputs"] = len(inputs)
    if inputs:
        # Drivers edit requests mid-session. Last = final stated need; first = what a
        # scheduler would know at plug-in (matters for the uncertainty extension).
        ordered = sorted(inputs, key=lambda u: u.get("modifiedAt") and
                         pd.to_datetime(u["modifiedAt"], format=API_TIME_FMT, utc=True).value or 0)
        for k in INPUT_FIELDS:
            row[k] = ordered[-1].get(k)
            row[f"{k}_first"] = ordered[0].get(k)
    return row


def to_frame(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([flatten(i) for i in items])
    for c in TIME_FIELDS:
        if c in df:
            df[c] = pd.to_datetime(df[c], format=API_TIME_FMT, utc=True)
    return df


def download_site(site: str, start: str | None = None, end: str | None = None) -> Path:
    s = make_session()
    first, last, total = site_bounds(s, site)
    print(f"{site}: server reports {total} sessions, {first} -> {last} (UTC)")
    lo = pd.Timestamp(start, tz="UTC") if start else first.normalize().replace(day=1)
    hi = pd.Timestamp(end, tz="UTC") if end else last + pd.Timedelta(seconds=1)

    raw_dir = DATA / "raw" / site
    raw_dir.mkdir(parents=True, exist_ok=True)
    edges = list(pd.date_range(lo.normalize().replace(day=1), hi, freq="MS", tz="UTC"))
    edges = [max(e, lo) for e in edges] + [hi]
    edges = sorted(set(edges))

    items: list[dict] = []
    for a, b in pairwise(edges):
        f = raw_dir / f"{a:%Y-%m}{'' if a.day == 1 and b.day == 1 else f'_{a:%d}-{b:%d}'}.json"
        if f.exists():
            month = json.loads(f.read_text())
        else:
            month = fetch_range(s, site, a, b)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(month))
            tmp.replace(f)  # atomic: a crash never leaves a half-written cache file
        print(f"  {a:%Y-%m-%d}..{b:%Y-%m-%d}: {len(month)}")
        items.extend(month)

    df = to_frame(items)
    out = DATA / f"{site}_sessions.csv" if not (start or end) else \
        DATA / f"{site}_sessions_{lo:%Y%m%d}_{hi:%Y%m%d}.csv"
    df.to_csv(out, index=False)
    print(f"{site}: wrote {len(df)} sessions -> {out.relative_to(ROOT)}")
    if not (start or end) and len(df) != total:
        print(f"  WARNING: {len(df)} rows != server total {total}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", choices=SITES + ("all",), required=True)
    p.add_argument("--start", help="UTC date, inclusive (default: first session)")
    p.add_argument("--end", help="UTC date, exclusive (default: after last session)")
    a = p.parse_args()
    for site in SITES if a.site == "all" else (a.site,):
        download_site(site, a.start, a.end)


if __name__ == "__main__":
    main()
