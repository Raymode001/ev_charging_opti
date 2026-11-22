# ev_charging_opti
ev charging optimization project for class

Full documentation (data, modelling, decisions, results): [PROJECT_GUIDE.md](PROJECT_GUIDE.md) · Related work: [LITERATURE.md](LITERATURE.md)

## Setup
```
pip install -r requirements.txt
echo "API_TOKEN=<your ACN-Data token>" > .env     # never commit; .env is gitignored
```

## Pipeline
| Step | Command | Output |
|---|---|---|
| Download sessions (all history, resumable) | `python -m acn.fetch --site jpl` (or `caltech`, `office001`, `all`) | `data/<site>_sessions.csv`, raw cache in `data/raw/` |
| Clean + validate (flags, report) | `python -m acn.clean --site all` | `data/<site>_clean.csv` |
| Core experiment | `python -m acn.run_core --site jpl --start 2019-01-01 --end 2020-02-29 [--mode weekly --objective daily\|max] [--r-min-kw 1.25]` | `results/core_*.csv` |
| Sensitivity scenarios | `python -m acn.sensitivity` | `results/sensitivity/*.csv` |
| Two-stage value curves | `python -m acn.stochastic --rates observed` | `results/two_stage/*.csv` |
| Departure uncertainty (online vs offline) | `python -m acn.run_online` | `results/online/*.csv` |
| Notebooks | `notebooks/core_results.ipynb`, `sensitivity.ipynb`, `two_stage.ipynb`, `online_uncertainty.ipynb` | `results/figures/*.png` |
| Tests | `python -m pytest -q` | |

The cleaned data is committed in `data/`. To regenerate it (or refresh), run the download and clean steps; the raw API cache `data/raw/` is not committed.

Load cleaned data with `acn.clean.load_clean(site)` (plain `read_csv` returns times and dates as strings).

## Modules
- `acn/fetch.py` – ACN API download (100 sessions/page; note `sort=-field` is **descending**).
- `acn/clean.py` – UTC→America/Los_Angeles, derived columns, `flag_*` columns, `ok` = no hard flag. Nothing dropped.
- `acn/slots.py` – 15-min slot instances (03:00–03:00 operational days or Mon–Mon weeks) with fractional availability; DST days have 92/100 slots.
- `acn/baselines.py` – uncontrolled, equal sharing (water-filling).
- `acn/models.py` – peak-minimizing LP (PuLP + HiGHS), daily or max objective, optional minimum-progress rate.
- `acn/metrics.py` – peaks, unmet energy, feasibility checks.

## Data facts (verified 2026-11-21)
| Site | Sessions | Local dates | stationIDs |
|---|---|---|---|
| caltech | 31,424 | 2018-04-25 → 2021-09-13 | 55 (54 L2 + 1 DCFC `2-39-81-4550`, from 2020-11-18) |
| jpl | 33,638 | 2018-09-05 → 2021-09-13 | 52 |
| office001 | 1,683 | 2019-03-25 → 2021-09-13 | 8 |

No data at Caltech and JPL 2020-08-04 → 2020-11-17. EVSE max 32 A × 208 V = 6.656 kW (acnportal `acnsim/models/evse.py`).
