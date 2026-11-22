# Project Guide: Workplace EV Charging Scheduling

How the codebase works, what the data looks like, how we modelled the problem, and which decisions we made and why.
Last updated 2026-11-22. Every number below was produced by the code in this repo. Related work: [LITERATURE.md](LITERATURE.md).

---

## 1. The problem in one paragraph

A workplace charging site has many cars plugged in at once. If every car charges at full speed the moment it plugs in ("uncontrolled"), the site's total power spikes in the morning. We split each day into 15-minute slots and choose how fast each car charges in each slot. The goal is to **minimize the site's peak power** while every car still gets its energy before it leaves. We compare against two baselines, uncontrolled charging and equal sharing, using real sessions from ACN-Data.

---

## 2. Quick start

```bash
pip install -r requirements.txt                    # or: uv pip install -r requirements.txt
echo "API_TOKEN=<your ACN-Data token>" > .env      # free token from ev.caltech.edu/dataset; .env is gitignored

python -m acn.fetch --site all                     # download all sessions (~5 min, resumable)
python -m acn.clean --site all                     # clean, flag, print validation report
python -m acn.run_core --site jpl --start 2019-01-01 --end 2020-02-29     # core experiment
python -m pytest -q                                # 51 tests
jupyter lab notebooks/core_results.ipynb           # figures comparing LP vs baselines
python -m acn.sensitivity                          # 9 sensitivity scenarios (~2 min)
python -m acn.stochastic --rates observed          # two-stage value curves (~1 min); also --rates uniform
python -m acn.run_online                           # offline vs online, departure uncertainty (~10 min)
```

---

## 3. Repository layout

```
ev_charging_opti/
├── acn/                    # all project code (a Python package; run modules with python -m)
│   ├── fetch.py            # download sessions from the ACN-Data API
│   ├── clean.py            # timezone conversion, derived columns, data-quality flags, report
│   ├── slots.py            # turn sessions into 15-min scheduling instances
│   ├── baselines.py        # uncontrolled + equal sharing
│   ├── models.py           # peak-minimizing LP (PuLP + HiGHS)
│   ├── metrics.py          # peaks, unmet energy, feasibility checks
│   ├── run_core.py         # core experiment runner -> results/*.csv
│   ├── analysis.py         # rebuild per-day schedules, equal-sharing min cap + cap sweep
│   ├── rates.py            # per-car max rate estimated from observed charging
│   ├── sensitivity.py      # sensitivity scenarios -> results/sensitivity/*.csv
│   ├── stochastic.py       # two-stage stochastic model (cap, then daily recourse)
│   ├── online.py           # rolling-horizon (MPC) controller under departure uncertainty
│   ├── run_online.py       # offline vs online experiment -> results/online/*.csv
│   └── plots.py            # all result figures, one consistent style
├── notebooks/
│   ├── core_results.ipynb  # visual comparison of LP vs baselines (executed, figures inline)
│   ├── sensitivity.ipynb   # rates, charger max, minimum rate, Caltech vs JPL, day types
│   ├── two_stage.ipynb     # two-stage stochastic cap model: SAA, EVPI, VSS, out-of-sample
│   └── online_uncertainty.ipynb  # departure uncertainty: offline vs online, frontier
├── tests/
│   ├── test_clean.py       # parsing, timezones/DST, flags, duplicates, gaps
│   ├── test_scheduling.py  # slotting, baselines, LP on hand-checkable cases
│   ├── test_extensions.py  # per-car rates, rate estimation, two-stage vs extensive form
│   └── test_analysis_online.py  # analysis helpers, plot smoke tests, online controller
├── data/
│   ├── raw/<site>/YYYY-MM.json   # raw API pages (cache; gitignored)
│   ├── <site>_sessions.csv       # flattened sessions, times in UTC
│   └── <site>_clean.csv          # + local times, derived columns, flag_* columns
├── results/                # experiment outputs, one row per day
│   └── figures/            # PNGs written by the notebook
├── dataset_EDA.ipynb       # original team notebook (has a sort bug, see section 4.4; superseded by acn/fetch.py)
├── requirements.txt
└── .env                    # API token (never commit)
```

---

## 4. Data

### 4.1 Source
- **ACN-Data**, the Adaptive Charging Network at Caltech: <https://ev.caltech.edu/dataset>.
- We pull session metadata through the REST API, `https://ev.caltech.edu/api/v1/sessions/<site>`, authenticated with the token as the HTTP username.
- We do **not** use the per-session time-series endpoint (`/ts`) because it is slow.
- The GitHub repo `tongxin-li/ACN-Data-Static` has time series only. Its session file is empty, so it is not used.

### 4.2 Coverage (verified against the API's own totals)

| Site | Sessions | Local dates | Unique stationIDs | Notes |
|---|---|---|---|---|
| caltech | 31,424 | 2018-04-25 → 2021-09-13 | 55 | 54 Level-2 chargers + 1 DC fast charger (`2-39-81-4550`, only from 2020-11-18) |
| jpl | 33,638 | 2018-09-05 → 2021-09-13 | 52 | Employees only, very regular weekdays |
| office001 | 1,683 | 2019-03-25 → 2021-09-13 | 8 | ~2.6 sessions/weekday; too small to be interesting |

### 4.3 Fields we use

| Field | Meaning |
|---|---|
| `connectionTime`, `disconnectTime` | Plug-in and unplug (UTC in the API) |
| `doneChargingTime` | When energy stopped flowing (missing for ~6%) |
| `kWhDelivered` | Energy actually delivered |
| `stationID`, `spaceID`, `siteID`, `clusterID` | Hardware and location identifiers |
| `userID` | Present for app users (~97% at JPL); allows per-driver history |
| `userInputs` (app users only) | `kWhRequested`, `requestedDeparture`, `minutesAvailable`, `milesRequested`, `WhPerMile`, `modifiedAt` |

Drivers often **edit their request during the session**; one session has 7 edits. We keep both versions:
- `kWhRequested`, `requestedDeparture`, etc. hold the **last** edit, the final stated need.
- `*_first` columns hold the **first** input, which is what a scheduler would know at plug-in. This matters for the uncertainty extension.

### 4.4 Data quirks found and how we handle them

| Issue | Evidence | Handling |
|---|---|---|
| `sort=-connectionTime` is **descending** | Returns 2021-09 first | `fetch.py` sorts ascending. The original notebook assumed the opposite, so it downloaded one day (51 sessions) and 11 empty months. |
| API page size capped at 100 | `max_results=1000` returns 100 | `fetch.py` pages at 100 |
| Times are UTC | Plug-ins cluster 13:00–16:00 UTC = 6–9 am Pacific | Convert to `America/Los_Angeles` after parsing as UTC, so DST is never ambiguous |
| **Data outage** at Caltech and JPL | No sessions 2020-08-04 → 2020-11-17 (106 days) | Avoid this window |
| COVID collapse | Volume drops from 2020-03 | Use the pre-COVID window (section 6) |
| Caltech regime change in Nov 2018 | ~2,400 → ~900 sessions/month; app usage 6% → 85% | Don't mix 2018 with 2019 without saying so. Likely the start of paid charging, not verified against a source. |
| DC fast charger at Caltech | Station `2-39-81-4550`: median 30.5 kW, 44 min dwell | Excluded (`slots.DCFC_STATIONS`); irrelevant pre-COVID anyway |
| `clusterID` both `'39'` and `'0039'` | All `'39'` rows are Oct 2018 | Normalized with `zfill(4)` |
| `doneChargingTime` before `connectionTime` (27) or after `disconnectTime` (2) | Clock or logging noise | Soft flags |
| Sessions over 24 h | 0.8% at Caltech | Soft flag |
| Duplicate station + time (JPL: 2) | | Hard flag, excluded |
| Requested energy ≫ delivered | Median delivered/requested: 0.60 JPL, 0.53 Caltech, 0.29 office001 | Why we schedule on delivered energy (section 7) |
| Departure estimates are noisy | JPL actual − requested departure: median +0.37 h, IQR [−0.87, +2.01] h | Input to the uncertainty extension |

### 4.5 Data-quality flags (`clean.py`)
Nothing is deleted at cleaning time. Each row gets `flag_*` columns plus `ok`, which means "no hard flag".

- **Hard** (excluded from scheduling):
  - `flag_dup_session_id`
  - `flag_dup_station_time`
  - `flag_nonpositive_dwell`
  - `flag_nonpositive_energy`
  - `flag_infeasible_at_max_kw` (energy > dwell × max rate)
- **Soft** (informational):
  - `flag_station_overlap`
  - `flag_tiny_energy`
  - `flag_missing_done`
  - `flag_done_before_connect`
  - `flag_done_after_disconnect`
  - `flag_overnight`
  - `flag_long_dwell_24h`
  - `flag_dst_day`
  - `flag_request_before_connect`
  - `flag_request_nonpositive`

> **Gotcha:** read cleaned data with `acn.clean.load_clean(site)`, not `pd.read_csv`. A plain `read_csv` returns times and dates as strings, which silently breaks date matching. Reindexing on a date range gives all zeros.

---

## 5. Pipeline

```
ACN API ──fetch.py──▶ data/raw/*.json ──▶ <site>_sessions.csv (UTC)
                                              │
                                         clean.py  (local time, derived cols, flags, report)
                                              ▼
                                       <site>_clean.csv
                                              │
                                 slots.usable_sessions  (ok, not DCFC, energy > 0)
                                              │
                         slots.daily_instances / weekly_instances  (15-min Instance objects)
                                              │
                 ┌────────────────────────────┼────────────────────────────┐
         baselines.uncontrolled      models.min_peak (LP)      baselines.equal_share(cap = LP peak)
                 └────────────────────────────┼────────────────────────────┘
                                         metrics.evaluate
                                              ▼
                                   results/core_*.csv (one row per day)
```

### 5.1 `fetch.py`
- Finds each site's first and last session, then downloads month by month using a `where` filter on `connectionTime`.
- Each month's raw JSON is cached atomically, so a crashed run resumes.
- Checks the downloaded count against the server's `_meta.total` and fails loudly on a mismatch.

### 5.2 `clean.py`
- Parses times and converts them to Pacific.
- Adds derived columns:
  - `dwell_h`, `charging_h`, `idle_h`
  - `avg_kw_dwell`, `avg_kw_charging`
  - `local_date`, `weekday`, `is_weekend`, `arrival_hour`, `app_user`
  - `departure_error_h` (and `_first_h`)
- Adds the flags from section 4.5 and prints the validation report.

### 5.3 `slots.py`: the `Instance` object

| Attribute | Shape | Meaning |
|---|---|---|
| `slot_starts` | (T,) | Local start time of each slot |
| `dt_h` | scalar | Slot length in hours (0.25) |
| `session_ids` | (n,) | Sessions in this horizon |
| `energy` | (n,) | Energy target $E_i$ (kWh) |
| `avail` | (n, T) | Fraction of slot $t$ that car $i$ is plugged in, $a_{it}\in[0,1]$ |
| `p_max` | scalar | Per-car max power (6.656 kW) |
| `day_index` | (T,) | Which operational day each slot belongs to |
| `info` | dict | Sessions dropped at the boundary; energy clipped |

### 5.4 `baselines.py`, `models.py`, `metrics.py`
See section 7.

### 5.5 `run_core.py`
For each instance, `run_core.py`:
1. solves the LP;
2. runs uncontrolled charging;
3. runs equal sharing with each day's cap set to **that day's LP optimal peak**;
4. writes one row per day.

Output columns:
- `date`, `weekday`, `n_sessions`, `energy_kwh`
- `peak_uncontrolled_kw`, `peak_lp_kw`, `peak_equal_share_kw`, `peak_reduction`
- `unmet_equal_share_kwh`, `unmet_equal_share_frac`
- `lp_unmet_kwh`, `lp_violations`
- `n_dropped_boundary`, `n_clipped`, `solve_s`

### 5.6 Extension modules
- **`rates.py`** adds a per-session `p_obs_kw` (D15). Pass `rate_col="p_obs_kw"` to `build_instance` / `daily_instances` to give each car its own max rate.
- **`sensitivity.py`** defines the scenarios (`SCENARIOS`) and writes one CSV per scenario to `results/sensitivity/`. Finished scenarios are skipped unless you pass `--force`.
- **`stochastic.py`** provides:
  - `StageTwo` (one day's stage-2 LP in HiGHS, re-solved warm for any cap);
  - `unmet_curves` (writes the tabulated curves to `results/two_stage/`);
  - `solve_saa`, `wait_and_see`, `evpi_vss`, `policy_table`;
  - `extensive_form` (for validation).
- **`analysis.py`, `plots.py`** are notebook helpers: schedule rebuilding, cap search and sweeps, and every figure in one style.

---

## 6. Decisions made

| # | Decision | Choice | Why | Alternatives considered |
|---|---|---|---|---|
| D1 | Data source | ACN API, all three sites, full history | Static repo has no sessions; team CSVs were broken | Website download (has a pre-Oct-2019 timestamp bug) |
| D2 | Main site and window | **JPL, 2019-01-01 → 2020-02-29** (pre-COVID) | Largest and most stable: 65 sessions/weekday, 97% app users | Caltech same window (site comparison); JPL 2020-11 → 2021-09 (robustness check) |
| D3 | Energy target $E_i$ | **`kWhDelivered`**, a hard constraint | Defined for every session; feasible by construction; no penalty weight to tune | `kWhRequested`: ~23% of requests exceed 6.656 kW × dwell and need a penalty. Kept as an extension. |
| D4 | Slot length | 15 min | Matches the project spec and the 15-min demand interval utilities bill on | |
| D5 | Partial slots | **Fractional availability** $a_{it}$ | Exact; rounding can never make a session infeasible | Inner rounding: 879 JPL sessions infeasible, 59 get zero slots. Outer rounding: charges cars before plug-in. |
| D6 | Day boundary | **03:00 local**, not midnight | Fewest crossing sessions at JPL (0.22% vs 1.37% at midnight) | Midnight; rolling windows |
| D7 | Horizon | **Daily** model (drops 44 crossing sessions) **and weekly** model (drops 3) | Weekly keeps overnight sessions; both give the same peaks at JPL | 40-hour rolling windows (see the uncertainty extension) |
| D8 | Objective | Two variants: **sum of daily peaks** and **max over horizon** | The daily peak is the project goal; the max matches billing-period demand charges | |
| D9 | Max rate | **32 A × 208 V = 6.656 kW** for every car | acnportal's EVSE model (`acnsim/models/evse.py`); JPL data: only 3 of 20,291 sessions exceed it, by ~0.05 kWh | 7 kW (old assumption). Per-car rate from data (planned sensitivity). |
| D10 | Minimum rate | Optional **minimum-progress** constraint (LP) | Protects early leavers and stays an LP | Literal per-slot "rate ≥ r_min while charging" = semi-continuous (MIP extension) |
| D11 | DC fast charger | Excluded | ~50 kW, no flexibility, not in acnportal's Caltech network | Model it as fixed load you can't control |
| D12 | Equal-sharing cap | That day's **LP optimal peak** | Both schedules then have the same peak, so the comparison is on unmet energy | A fixed site cap (transformer); a cap sweep (planned) |
| D13 | Solver | PuLP 2.x + **HiGHS** | PuLP 4.x broke the classic API; bundled CBC is x86-only on Apple Silicon | Gurobi (installed locally, but not needed) |
| D14 | Infeasible energy | Clip $E_i$ to $p_{\max}\sum_t a_{it}\Delta t$, and count it | 3 sessions at JPL, all metering noise | Drop them |
| D15 | Per-car max rate (sensitivity) | User's **90th-percentile** observed rate (kWh / charging hours), else the session's own rate, else the median; at least kWh/dwell, at most 6.656 kW | Per-user rates are bimodal (~3.3 / ~6 kW, on-board chargers) and stable (median within-user CV 0.24). A high quantile, because ACN throttled some sessions. Looks ahead in time (fine offline). | Session's own rate (biased low by throttling) |
| D16 | Two-stage scenarios | Historical **weekdays**: train 2019 (260), test Jan–Feb 2020 (43) | Each scenario is a real day; train/test split checks out-of-sample cost | Bootstrapped or synthetic days |
| D17 | Two-stage prices | Demand charge **$20/kW-month** ÷ 21 weekdays; unmet penalty λ **swept** $0.1–100/kWh (base $1) | ≈ SCE TOU-8 facilities-related demand charge ($19.31–23.71/kW; year unverified). λ has no market price. | |
| D18 | Two-stage solution method | Tabulate each day's unmet-vs-cap curve on a 1 kW grid, then 1-D minimize; validated against the extensive-form LP | C is the only first-stage variable and each curve is convex, so this is exact up to grid resolution and makes price sweeps free | Extensive form for all 260 days (large) |

---

## 7. How we modelled it

### 7.1 Notation

| Symbol | Meaning | Code |
|---|---|---|
| $i = 1..n$ | Charging sessions (cars) | `session_ids` |
| $t = 1..T$ | 15-min slots | `slot_starts` |
| $\Delta t$ | Slot length, 0.25 h | `dt_h` |
| $a_{it}\in[0,1]$ | Fraction of slot $t$ during which car $i$ is plugged in | `avail` |
| $E_i$ | Energy target (kWh) = `kWhDelivered`, clipped | `energy` |
| $\bar p$ | Per-car max power, 6.656 kW | `p_max` |
| $d(t)$ | Operational day of slot $t$ | `day_index` |
| $r_{it}\ge 0$ | **Decision:** average power to car $i$ in slot $t$ (kW) | `R[i, t]` |
| $P_d$ | Peak site power on day $d$ | `peaks` |

Because $r_{it}$ is an *average* over the slot, the site load $\sum_i r_{it}$ is the 15-minute average demand. That is the quantity utilities bill on.

### 7.2 Core LP (`models.min_peak`)

$$
\begin{aligned}
\min_{r,\,P}\quad & \sum_{d} P_d \qquad\text{(objective="daily")} \\
\text{s.t.}\quad & 0 \le r_{it} \le a_{it}\,\bar p && \forall i,t \quad\text{(rate limit; 0 when absent)}\\
& \sum_t r_{it}\,\Delta t = E_i && \forall i \quad\text{(deliver energy before departure)}\\
& \sum_i r_{it} \le P_{d(t)} && \forall t \quad\text{(site load within that day's peak)}
\end{aligned}
$$

- **objective="max":** minimize $z + \varepsilon\sum_d P_d$ with $P_d \le z$. This is one peak over the whole horizon; the small $\varepsilon = 10^{-4}$ term stops non-peak days from being left arbitrary.
- **Variables** exist only where $a_{it}>0$. A typical JPL day has about 70 cars and 96 slots and solves in about 0.01 s.
- **Separability:** if no session crosses a day boundary, the weekly sum-of-daily-peaks model splits into 7 independent daily LPs. This is why the daily and weekly results are nearly identical at JPL.

### 7.3 Minimum rate as minimum progress (optional, `--r-min-kw`)

$$
\sum_{s\le t} r_{is}\,\Delta t \;\ge\; \min\Big(E_i,\; r_{\min}\sum_{s\le t} a_{is}\,\Delta t\Big) \qquad \forall i,t
$$

A car that leaves at *any* moment has at least what charging at $r_{\min}$ the whole time would have given it, capped at its need. The right-hand side is data, so this stays linear.

Hardware note from acnportal:
- AeroVironment EVSEs accept 6–32 A, so the minimum is 6 A ≈ 1.25 kW.
- ClipperCreek EVSEs accept only {0, 8, 16, 24, 32} A.

A literal "if charging then $r \ge r_{\min}$" constraint is semi-continuous and needs binary variables. That is the planned MIP extension.

### 7.4 Baselines (`baselines.py`)
- **Uncontrolled:** $r_{it} = \min\big(a_{it}\bar p,\; \text{remaining}_i/\Delta t\big)$ from plug-in until done. Front-loaded, and it always meets $E_i$.
- **Equal sharing with cap $C$:** in each slot, $C$ is split equally among cars that are present *and still need energy*. Each car is capped at $\min(a_{it}\bar p,\ \text{remaining}_i/\Delta t)$. A share a car can't use goes to the others (water-filling). Cars that are done get nothing.
- **Both are causal.** They use only information available at time $t$; neither knows departure times.

### 7.5 Metrics (`metrics.py`)
- **Peaks:** `peak_kw` (max over the horizon) and `daily_peaks` (per day).
- **Unmet energy:** $\sum_i \max(0, E_i - \sum_t r_{it}\Delta t)$, as kWh, as a share of energy, and as the number of sessions affected.
- **Feasibility check:** negative rates; charging above the rate limit or while absent. Should always be 0, and is.

### 7.6 Per-car max rate (sensitivity)
$\bar p$ becomes $p_i$ per car: its user's 90th-percentile observed rate, clamped to $[E_i/\text{dwell}_i,\ 6.656]$. Everything in 7.2–7.4 holds with $\bar p \to p_i$.

### 7.7 Two-stage stochastic cap (`stochastic.py`)

$$
\min_{C\ge 0}\; c\,C + \frac{1}{S}\sum_{s=1}^{S}\lambda\,U_s(C),\qquad
U_s(C)=\min_{r,u\ge 0}\Big\{\textstyle\sum_i u_i:\ r_{it}\le a_{it}p_i,\ \sum_t r_{it}\Delta t+u_i=E_i,\ \sum_i r_{it}\le C\ \forall t\Big\}
$$

- **Stage 1:** the site cap $C$, paid as a demand charge $c$ per kW per weekday.
- **Stage 2:** for each scenario (a historical weekday), charging under the cap, with undelivered energy $u$ penalized at $\lambda$.
- **Why the grid method is exact:** $U_s$ is convex, piecewise-linear and non-increasing in $C$. So tabulating it on a 1 kW grid and minimizing in 1-D is exact up to grid resolution, and is checked against the extensive form.
- **Policies compared:**
  - mean-day plan (EEV): $C$ = mean of daily optimal peaks;
  - median-day plan;
  - worst-case plan: $C$ = max of daily optimal peaks;
  - wait-and-see (WS): each day picks its own cap.
- **Textbook quantities:** EVPI = RP − WS and VSS = EEV − RP.

### 7.8 Online control under departure uncertainty (`online.py`)
Every 15 minutes, the controller sees cars that have arrived, their remaining energy $E_i^{rem}$ and an estimated departure. It then solves

$$
\min\; P + M\sum_i u_i + \varepsilon\sum_{i,s} s\, r_{is}\Delta t\quad\text{s.t.}\quad \sum_{s\ge t} r_{is}\Delta t + u_i = E_i^{rem},\;\; 0\le r_{is}\le \hat a_{is}p_i,\;\; \sum_i r_{is}\le P,\;\; P\ge \text{peak so far}
$$

- **Plan vs reality:** it applies slot $t$'s rates. Cars leave at their actual time, and whatever they haven't received is unmet. A car that stays past its estimate is assumed to leave at the end of the current slot.
- **Estimators:**
  - true departure;
  - driver's **first** stated departure (median dwell by arrival hour if missing);
  - cautious (stated + 25th-percentile error);
  - none (03:00).
- **Optional minimum progress:** a lower bound on each car's current-slot rate so that delivered ≥ min(E, r_min × hours plugged in).
- **What's assumed known:** energy needs are known at plug-in, so only departures are uncertain.

---

## 8. Assumptions and parameters (with sources)

| Parameter | Value | Source / status |
|---|---|---|
| Per-car max power | 6.656 kW (32 A × 208 V) | acnportal `acnsim/models/evse.py`. Simulator default, not a hardware spec sheet. |
| EVSE minimum | 6 A ≈ 1.25 kW (AV); discrete 8/16/24/32 A (CC) | acnportal `evse.py` |
| Caltech transformer | **150 kW**; two 80 A pods | acnportal `caltech_acn.py`. Note: a 2019 Caltech info page says 300 "kWh" across 80+ ports. **These disagree; cite which one you use.** |
| JPL transformers | 45 kW (1st floor) + 150 kW (3rd/4th floors) | acnportal `jpl_acn.py` |
| office001 transformer | 50 kW | acnportal `office001_acn.py` |
| Timezone | America/Los_Angeles | API `timezone` field |
| Per-car max rate (sensitivity) | User's 90th-pct observed rate, ≤ 6.656 kW | `acn/rates.py`, from data; looks ahead in time |
| Demand charge (two-stage) | $20/kW-month (÷ 21 weekdays) | ≈ SCE TOU-8 facilities-related demand charge ($19.31–23.71/kW in the RTP schedule; year not confirmed) |
| Unmet-energy penalty λ | $1/kWh base, swept $0.1–100 | Modelling choice; no market price |
| Literature to cite | Lee, Li, Low, "ACN-Data: Analysis and Applications of an Open EV Charging Dataset" (e-Energy 2019); Lee et al., "Adaptive Charging Networks: A Framework for Smart EV Charging" (IEEE Trans. Smart Grid, 2021) | Not yet checked against the paper text |

---

## 9. Validation

**51 unit tests** (`python -m pytest -q`).

`test_clean.py` covers:
- UTC → local in summer and winter;
- late-evening UTC sessions mapped to the previous local day;
- dwell across the fall-back DST switch (elapsed hours, not wall-clock);
- DST date detection;
- zero and negative dwell, zero energy;
- energy infeasible at max rate;
- duplicates and station overlap;
- done-after-disconnect tolerance;
- zero-session days;
- first and last driver input;
- `clusterID` normalization;
- `load_clean` round trip.

`test_scheduling.py` covers:
- partial-slot fractions summing to dwell;
- DST operational days having 92, 100 or 96 slots;
- boundary sessions dropped and counted;
- weekly instances keeping overnight sessions;
- energy clipping;
- water-filling;
- uncontrolled front-loading;
- equal-sharing redistribution;
- **a hand-solved LP** (optimal peak 2 kW);
- a random LP that meets all energy and is never worse than uncontrolled;
- daily vs max objectives;
- minimum progress.

`test_extensions.py` covers:
- per-car rates broadcast and respected by uncontrolled charging and the LP;
- a minimum rate above a slow car's max staying feasible;
- rate-estimation sources (user / session / median) and bounds;
- the stage-2 LP's unmet energy, checked by hand;
- grid SAA matching the extensive-form LP;
- wait-and-see being a lower bound;
- the optimal cap rising with the penalty λ.

`test_analysis_online.py` covers:
- the minimum equal-sharing cap, checked by hand (including the 0.01 kWh tolerance);
- the cap sweep being monotone;
- occupancy;
- every plot function rendering;
- the online LP with true departures delivering everything, and never beating the offline peak;
- an optimistic departure estimate causing unmet energy, which minimum progress removes;
- the departure estimators.

`test_scheduling.py` also checks that the smoothing tie-break keeps the peak and flattens cars.

**Sanity checks on real data:**
- Downloaded counts equal the API totals.
- Dropped-session counts match independent estimates: 44 daily, 3 weekly.
- The LP has zero violations over 420 days.
- The LP peak is never above the uncontrolled peak.
- Equal sharing's unmet energy never increases as its cap rises, on every day (so the bisection is valid).
- All 303 two-stage unmet-vs-cap curves are non-increasing and convex.
- Every figure was inspected by eye after rendering.

---

## 10. Results so far

**Headline (most defensible) number:** with each car's observed charging rate, the LP cuts JPL's median weekday peak by **58.5%** (95% CI 58.1–59.2%) vs uncontrolled, under perfect information. The 67% below is the core run, where every car is assumed to draw 6.656 kW.

### 10.0 Core model

JPL, 2019-01-01 → 2020-02-29, 297 weekdays with at least 5 sessions. Daily mode, perfect information.

| Metric | Median | 10th–90th pct |
|---|---|---|
| Sessions per day | 72 | 34–78 |
| Uncontrolled peak | 254 kW | 77–281 kW |
| LP peak | 81 kW | 36–94 kW |
| Peak reduction | **67%** | 54–72% |
| Equal-sharing unmet energy at the LP's peak | **8.6%** | 6.6–11.7% |

Weekly mode (both objectives) gives the same peaks. Files: `results/core_jpl_*.csv`.

**Peak needed to deliver 100% of energy** (from `notebooks/core_results.ipynb`; for equal sharing, the smallest cap that delivers everything, found by bisection):

| Method | Median daily peak | Median reduction vs uncontrolled |
|---|---|---|
| Uncontrolled | 254 kW | 0% |
| Equal sharing | 144 kW | 40% |
| Optimized (LP) | 81 kW | 67% |

At the LP's own peak, equal sharing leaves a median 8.6% of energy undelivered; it needs about 1.8× the LP peak to deliver everything on the median day. Unmet energy was non-increasing in the cap on every day, so the bisection is valid.

**Quiet Fridays.** About every other Friday has 25–45 sessions instead of about 70. This fits a 9/80 schedule (alternate Fridays off), which I believe JPL uses but haven't verified. Friday medians therefore mix two kinds of day.

**The LP schedule is not unique.** Only the peak is determined, and a plain solve returns an arbitrary corner solution that looks speckled per car. `min_peak(..., smooth=True)` adds a second LP that keeps every day's peak and minimizes each car's highest instantaneous power. On 2019-09-23 the peak is identical (80.05 kW), and the median per-car maximum power drops from 6.66 to 4.43 kW. The notebooks' heatmaps use it.

**Caltech daily vs weekly.** Daily mode drops 487 overnight sessions at the 03:00 boundary; weekly mode keeps 434 of them. The result doesn't change:

| Mode | Median reduction | Median LP peak |
|---|---|---|
| Daily | 66.8% | 23.6 kW |
| Weekly (both objectives) | 66.8% | 23.7 kW |

The daily-mode bias is negligible.

**Caveats. Read before quoting these numbers:**
1. **Perfect information.** The LP knows each car's actual departure and energy. This is an upper bound on what scheduling can achieve.
2. **Every car is assumed to draw 6.656 kW.** Real cars are often slower because the car's own charger limits them. This inflates the uncontrolled peak: 254 kW is more than JPL's modeled 45 + 150 kW transformers. **Resolved in 10.1:** with observed per-car rates the reduction is 58.5%.
3. **Energy target = delivered energy.** That comes from ACN's own scheduler and from cars filling up, so it is a lower bound on true need.

### 10.1 Sensitivity (`notebooks/sensitivity.ipynb`)

JPL weekdays, 2019-01 → 2020-02. The peak is the one each method needs to deliver 100% of energy. 95% CIs are bootstrap intervals on the median over days.

| Scenario | Uncontrolled | Equal sharing | LP | LP reduction (95% CI) | Equal-sharing reduction |
|---|---|---|---|---|---|
| All cars 6.656 kW (core) | 254 kW | 144 kW | 81 kW | 67.1% [66.6, 67.6] | 40.0% |
| **Observed per-car rates** | **210 kW** | **166 kW** | **84 kW** | **58.5% [58.1, 59.2]** | **14.8%** |
| All cars 5.0 kW | 213 | 210 | 82 | 60.1% | 0.4% |
| All cars 3.3 kW (13.6% of energy clipped) | 152 | 152 | 75 | 48.6% | 0.1% |
| Observed + min 6 A (1.25 kW) | 210 | 166 | 86 | 57.6% | 14.8% |
| Observed + min 8 A (1.66 kW) | 210 | 166 | 87 | 57.0% | 14.8% |
| Observed + min 16 A (3.33 kW) | 210 | 166 | 128 | 38.6% | 14.8% |
| Caltech, all cars 6.656 kW | 73 | 50 | 24 | 66.8% | 26.1% |
| Caltech, observed rates | 66 | 51 | 24 | 62.9% | 18.6% |

- **With realistic per-car rates the LP's reduction is 58.5%, not 67%.** Slow cars can't spike the uncontrolled peak as much, while the LP peak barely moves.
- **Equal sharing collapses as cars get slower.** With little slack, any cap leaves someone short, so it needs almost the uncontrolled peak.
- Minimum rates of 6–8 A cost only 2–3 kW of peak. 16 A costs 44 kW.
- By day type (observed rates), median reduction:

  | Day type | Median reduction | Sessions |
  |---|---|---|
  | Mon–Thu | 59% | 72 |
  | Busy Friday | 58% | |
  | Quiet Friday | 46% | 32 |
  | Weekend | 33% | 6 |

- The app-user share can't be varied at JPL (about 97% everywhere).

### 10.2 Two-stage stochastic cap (`notebooks/two_stage.ipynb`)

Observed per-car rates; $20/kW-month; λ = $1/kWh.

- Optimal cap **C\* = 98 kW**: about the 91st percentile of daily optimal peaks.
- Expected cost **$97.2/weekday in-sample and $98.4 out-of-sample**.
- Alternative policies, out-of-sample:

  | Policy | Cap | Cost per weekday |
  |---|---|---|
  | Mean-day plan | 78 kW | $180.1 |
  | Median-day plan | 84 kW | $142.7 |
  | Worst-case plan | 113 kW | $107.6 |
  | Wait-and-see (perfect information) | | $74.4 |

- **EVPI = $23.1/weekday, VSS = $54.6/weekday.** VSS grows quickly with λ ($206 at λ = 3).
- C\* rises from 64 kW (λ = $0.1) to the worst-day 113 kW (λ ≥ $30). The model behaves like a newsvendor: the cap is a quantile of the daily-peak distribution.
- Validation:
  - every unmet-vs-cap curve is non-increasing and convex (0 violations);
  - the grid optimum is within 0.07 of the extensive-form LP on a 15-day subset, which is within the grid-resolution bound.
- Recourse is **offline within the day** (departures known once the day is revealed), so this models day-to-day uncertainty, not intraday departure uncertainty.

### 10.3 Departure uncertainty (`notebooks/online_uncertainty.ipynb`)

JPL, 297 weekdays, observed per-car rates; medians over days.

| Policy | Peak | Energy not delivered |
|---|---|---|
| Offline LP (perfect information) | 84 kW | 0% |
| Online LP, true departures (unknown arrivals) | 119 kW | 0% |
| Online LP, driver's first stated departure | 98 kW | **10.2%** |
| Online LP, cautious estimate | 112 kW | 5.5% |
| Online LP, driver estimate + 8 A min progress | 105 kW | 3.0% |
| Online LP, driver estimate + 12 A | 130 kW | 0.95% |
| Equal sharing, same 98 kW peak | 98 kW | **3.5%** |
| Equal sharing, 1.2× / 1.4× offline peak | 101 / 118 kW | 2.9% / 1.1% |
| Uncontrolled | 214 kW | 0% |

- **Unknown arrivals cost peak:** +41% over offline.
- **Wrong departures cost energy.** 35% of drivers leave earlier than stated, and 39% of estimates are off by more than 2 h.
- **At JPL, equal sharing matches or beats the online LP across the whole peak-vs-unmet frontier.** The online LP needs a minimum-progress hedge to catch up. With noisy information, a rule that never defers anyone is hard to beat.
- The gap between offline and online is the value of information.

---

## 11. Open questions and next steps

**Done:**
- sensitivity analyses (10.1);
- two-stage stochastic cap model (10.2);
- departure uncertainty, offline vs online (10.3);
- Caltech weekly mode;
- LP smoothing tie-break;
- literature review (`LITERATURE.md`);
- tests for analysis, plots and the online controller.

**Future work (in suggested order):**
1. **Make optimization win online.**
   - Per-driver departure distributions learned from `userID` history.
   - Scenario-based online planning (Puech et al. 2024; Tucker et al. 2022).
   - A fairness / "don't defer" term in the online objective.
   - Then put the online controller in stage 2 of the two-stage model instead of today's offline recourse.
2. **Prices / monthly demand charge.** A monthly horizon where the bill is c × (max over the month) plus time-of-use energy prices; the "max" and "sum of daily peaks" objectives should diverge here.
3. **Energy-need uncertainty:** plan with drivers' requested energy (median about 1.6× delivered) instead of known `kWhDelivered`.
4. **Peak vs unmet-energy trade-off under a fixed site cap** (150 kW acnportal vs 300 kW info page). `StageTwo.unmet(C)` and `analysis.cap_sweep` already compute both sides.
5. **App-user share** at Caltech (85% app users); **MIP minimum rate** with ClipperCreek's discrete {0, 8, 16, 24, 32} A.

**Open questions for the team:**
- Which transformer limit to cite for Caltech: 150 kW (acnportal) or 300 kW (2019 info page)?
- Should the large `data/*.csv` files (8–20 MB each) be committed, or regenerated from the API?
- Fix or retire `dataset_EDA.ipynb`? It still has the descending-sort bug.

---

## 12. Gotchas

- `sort=-field` in the ACN API means **descending**.
- Use `load_clean(site)`, not `read_csv`, for cleaned data.
- Don't `pip install pulp` without the version pin: 4.x breaks the classic API.
- On Apple Silicon, PuLP's bundled CBC fails with "Bad CPU type"; use `pulp.HiGHS()` (needs `highspy`).
- DST days have 92 or 100 slots, not 96. Never assume `T = 96`.
- A session that crosses the day boundary touches two days. Dropped-session counts are attributed to the day the car arrives.
- The API token lives in `.env`. Never paste it into code, notebooks, or commits.
