"""Figures for the core results. One consistent style; one function per figure.

Color follows the method everywhere (never its rank):
  LP (optimized) = blue, uncontrolled = orange, equal sharing = aqua.
Palette: validated categorical slots 1-3 (CVD-safe as a set of three).
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from acn.slots import Instance

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"lp": "#2a78d6", "uncontrolled": "#eb6834", "equal_share": "#1baf7a"}
LABELS = {"lp": "Optimized (LP)", "uncontrolled": "Uncontrolled", "equal_share": "Equal sharing"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })


def _hours(inst: Instance) -> np.ndarray:
    """Slot start as decimal local hour, continuing past 24 for times after midnight."""
    s = inst.slot_starts
    h = s.hour + s.minute / 60
    return np.where(s.date > s[0].date(), h + 24, h)


def _hour_axis(ax) -> None:
    ticks = np.arange(3, 28, 3)
    ax.set_xticks(ticks, [f"{int(t % 24):02d}:00" for t in ticks])
    ax.set_xlim(3, 27)


def day_profile(inst: Instance, schedules: dict, occupancy: np.ndarray, title: str):
    """Site load over one day for each method, with cars-present in a separate panel."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    x = _hours(inst)
    for k in ["uncontrolled", "equal_share", "lp"]:
        load = schedules[k].sum(0)
        ax.step(x, load, where="post", color=COLORS[k], label=LABELS[k],
                lw=2.2 if k == "lp" else 1.8)
        if k != "equal_share":  # capped at the LP peak, so its label would duplicate the LP's
            i = int(load.argmax())
            txt = f"{load[i]:.0f} kW" + (" (LP peak = equal-sharing cap)" if k == "lp" else "")
            ax.annotate(txt, (x[i], load[i]), xytext=(6, 5), textcoords="offset points",
                        color=INK, fontsize=9)
    ax.set_ylabel("Site power (kW, 15-min average)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncols=3)
    ax.set_ylim(bottom=0)
    ax2.fill_between(x, occupancy, step="post", color=INK_2, alpha=0.25, lw=0)
    ax2.step(x, occupancy, where="post", color=INK_2, lw=1.2)
    ax2.set_ylabel("Cars plugged in")
    ax2.set_ylim(bottom=0)
    _hour_axis(ax2)
    ax2.set_xlabel("Local time")
    return fig


def schedule_heatmaps(inst: Instance, schedules: dict, title: str):
    """Per-car charging power (rows = cars sorted by arrival) for uncontrolled vs LP."""
    order = np.argsort(np.argmax(inst.avail > 0, axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, constrained_layout=True)
    x = _hours(inst)
    extent = [x[0], x[-1] + inst.dt_h, inst.n, 0]
    for ax, k in zip(axes, ["uncontrolled", "lp"], strict=True):
        im = ax.imshow(schedules[k][order], aspect="auto", cmap=BLUES, vmin=0, vmax=float(inst.p_max.max()),
                       extent=extent, interpolation="nearest")
        # Outline when each car is plugged in, so "absent" and "present but idle" differ.
        ax.contour(np.linspace(x[0], x[-1] + inst.dt_h, inst.T), np.arange(inst.n) + 0.5,
                   (inst.avail[order] > 0).astype(float), levels=[0.5], colors=INK_2, linewidths=0.5)
        ax.set_title(LABELS[k])
        ax.grid(False)
        _hour_axis(ax)
        ax.set_xlabel("Local time")
    axes[0].set_ylabel("Car (sorted by arrival)")
    fig.colorbar(im, ax=axes, label="Charging power (kW)", shrink=0.8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=INK)
    return fig


def peaks_over_time(res: pd.DataFrame):
    """Daily peak per weekday (dots) with a 20-weekday rolling median (line)."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    r = res.assign(d=pd.to_datetime(res["date"])).sort_values("d")
    for k, col in [("uncontrolled", "peak_uncontrolled_kw"), ("lp", "peak_lp_kw")]:
        ax.scatter(r["d"], r[col], s=9, color=COLORS[k], alpha=0.35, lw=0)
        ax.plot(r["d"], r[col].rolling(20, center=True, min_periods=5).median(), color=COLORS[k],
                lw=2.2, label=f"{LABELS[k]} (dots: days; line: 4-week rolling median)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Daily peak (kW)")
    ax.set_ylim(0, r["peak_uncontrolled_kw"].max() * 1.15)
    ax.set_title("Daily site peak, every weekday")
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    return fig


def peak_distributions(res: pd.DataFrame, min_cap: pd.Series):
    """Distribution of daily peaks when every method delivers all energy."""
    data = {"uncontrolled": res["peak_uncontrolled_kw"].to_numpy(),
            "equal_share": min_cap.to_numpy(),
            "lp": res["peak_lp_kw"].to_numpy()}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(0)
    for j, k in enumerate(["uncontrolled", "equal_share", "lp"]):
        v = data[k]
        ax.scatter(v, j + rng.uniform(-0.18, 0.18, len(v)), s=10, color=COLORS[k], alpha=0.45, lw=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [j, j], color=INK, lw=3, solid_capstyle="round")
        ax.plot(med, j, "o", color=SURFACE, mec=INK, mew=1.8, ms=8)
        ax.annotate(f"median {med:.0f} kW", (med, j), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_yticks(range(3), [LABELS[k] for k in ["uncontrolled", "equal_share", "lp"]])
    ax.set_xlabel("Daily peak needed to deliver 100% of energy (kW)")
    ax.set_xlim(left=0)
    ax.grid(axis="y", visible=False)
    ax.set_title("Peak needed to meet every car's energy, per weekday")
    return fig


def tradeoff(sweep: pd.DataFrame, res: pd.DataFrame):
    """Equal-sharing unmet energy vs its site cap (as a multiple of the LP peak)."""
    g = sweep.groupby("multiplier")["unmet_frac"]
    m = g.median().index.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(m, g.quantile(0.1) * 100, g.quantile(0.9) * 100, color=COLORS["equal_share"],
                    alpha=0.18, lw=0, label="Equal sharing, 10th-90th pct of days")
    ax.plot(m, g.median() * 100, color=COLORS["equal_share"], marker="o", ms=5,
            label=f"{LABELS['equal_share']}, median day")
    ax.plot(1.0, 0, "o", color=COLORS["lp"], ms=10, mec=SURFACE, mew=2, zorder=5,
            label=f"{LABELS['lp']}: 0% unmet at 1.0x")
    unc_mult = (res["peak_uncontrolled_kw"] / res["peak_lp_kw"]).median()
    ax.plot(unc_mult, 0, "s", color=COLORS["uncontrolled"], ms=9, mec=SURFACE, mew=2, zorder=5,
            label=f"{LABELS['uncontrolled']}: 0% unmet at {unc_mult:.1f}x (median)")
    ax.set_xlabel("Site power cap, as a multiple of that day's optimal (LP) peak")
    ax.set_ylabel("Energy not delivered (%)")
    ax.set_ylim(bottom=-0.4)
    ax.set_title("Same cap, different outcome: equal sharing leaves energy undelivered")
    ax.legend(loc="upper right")
    return fig


def reduction_vs_sessions(res: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(res["n_sessions"], res["peak_reduction"] * 100, s=14, color=COLORS["lp"], alpha=0.55, lw=0)
    ax.set_xlabel("Sessions that day")
    ax.set_ylabel("Peak reduction vs uncontrolled (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Busier days gain more from scheduling")
    return fig


def by_weekday(res: pd.DataFrame):
    """Median daily peak per weekday, grouped bars per method."""
    med = res.groupby("weekday")[["peak_uncontrolled_kw", "peak_equal_share_min_kw", "peak_lp_kw"]]\
        .median().reindex(WEEKDAYS)
    keys = ["uncontrolled", "equal_share", "lp"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.26
    x = np.arange(len(WEEKDAYS))
    for j, (k, col) in enumerate(zip(keys, med.columns, strict=True)):
        bars = ax.bar(x + (j - 1) * w, med[col], width=w - 0.02, color=COLORS[k], label=LABELS[k])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(x, WEEKDAYS)
    ax.set_ylabel("Median daily peak (kW)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Median peak by weekday (all methods deliver 100% of energy)", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3)
    return fig


# ---------- sensitivity ----------

def scenario_peaks(summary: pd.DataFrame):
    """Median daily peak (dot) and IQR (bar) per scenario and method.

    summary: one row per (scenario label, method) with columns label, method, q25, q50, q75.
    """
    labels = list(dict.fromkeys(summary["label"]))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(labels) + 1.6))
    offs = {"uncontrolled": -0.22, "equal_share": 0.0, "lp": 0.22}
    for k, off in offs.items():
        s = summary[summary["method"] == k].set_index("label").reindex(labels)
        y = np.arange(len(labels)) + off
        ax.hlines(y, s["q25"], s["q75"], color=COLORS[k], lw=3, alpha=0.55)
        ax.plot(s["q50"], y, "o", color=COLORS[k], ms=7, mec=SURFACE, mew=1.5, label=LABELS[k])
        for yi, v in zip(y, s["q50"], strict=True):
            ax.annotate(f"{v:.0f}", (v, yi), xytext=(7, -3), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Daily peak needed to deliver 100% of energy (kW): median, bar = middle 50% of weekdays")
    ax.grid(axis="y", visible=False)
    ax.set_title("How the peaks move under each assumption", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3)
    return fig


def scenario_reductions(summary: pd.DataFrame):
    """Median peak reduction vs uncontrolled per scenario: LP and equal sharing."""
    labels = list(dict.fromkeys(summary["label"]))
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(labels) + 1.6))
    y = np.arange(len(labels))
    for k, off in [("equal_share", -0.18), ("lp", 0.18)]:
        s = summary.set_index("label").reindex(labels)[f"red_{k}"] * 100
        bars = ax.barh(y + off, s, height=0.34, color=COLORS[k], label=LABELS[k])
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK_2, padding=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Median peak reduction vs uncontrolled (%), weekdays")
    ax.grid(axis="y", visible=False)
    ax.set_title("Peak reduction under each assumption", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    return fig


def rmin_effect(df: pd.DataFrame):
    """LP peak vs minimum charging rate (median and IQR across weekdays)."""
    g = df.groupby("r_min_kw")["peak_lp_kw"]
    x = g.median().index.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(x, g.quantile(0.25), g.quantile(0.75), color=COLORS["lp"], alpha=0.15, lw=0,
                    label="middle 50% of weekdays")
    ax.plot(x, g.median(), color=COLORS["lp"], marker="o", ms=7, label="median weekday")
    for xi, v in zip(x, g.median(), strict=True):
        ax.annotate(f"{v:.0f} kW", (xi, v), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xticks(x, [f"{v:.2f} kW" + ("" if v == 0 else f"\n({v / 0.208:.0f} A)") for v in x])
    ax.set_xlabel("Minimum charging rate (minimum-progress constraint)")
    ax.set_ylabel("Optimal (LP) daily peak (kW)")
    ax.set_ylim(bottom=0)
    ax.set_title("Cost of guaranteeing a minimum charging rate")
    ax.legend(loc="lower right")
    return fig


# ---------- two-stage ----------

def unmet_curve_plot(U: pd.DataFrame, C_star: float | None = None):
    """Mean unmet energy per day vs site cap, with the spread across days."""
    grid = U.columns.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.fill_between(grid, U.quantile(0.1), U.quantile(0.9), color=COLORS["lp"], alpha=0.15, lw=0,
                    label="10th-90th pct of days")
    ax.plot(grid, U.mean(), color=COLORS["lp"], lw=2.2, label="mean over days")
    if C_star is not None:
        ax.axvline(C_star, color=INK_2, lw=1, ls="--")
        ax.annotate(f"two-stage cap\n{C_star:.0f} kW", (C_star, U.mean().max() * 0.45), xytext=(-6, 0),
                    textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.set_xlabel("Site power cap C (kW)")
    ax.set_ylabel("Energy not delivered (kWh per day)")
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(bottom=0)
    ax.set_title("Stage 2: undelivered energy as the cap tightens")
    ax.legend(loc="upper right")
    return fig


def cost_decomposition(U: pd.DataFrame, c: float, lam: float):
    """Expected daily cost f(C) = demand charge + unmet penalty, with its minimum."""
    grid = U.columns.to_numpy(float)
    demand = c * grid
    unmet = lam * U.mean().to_numpy()
    total = demand + unmet
    j = int(total.argmin())
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(grid, demand, color=INK_2, lw=1.4, ls="--", label="demand charge  c·C")
    ax.plot(grid, unmet, color=COLORS["uncontrolled"], lw=1.6, label="unmet-energy penalty  λ·E[U(C)]")
    ax.plot(grid, total, color=COLORS["lp"], lw=2.4, label="total expected cost")
    ax.plot(grid[j], total[j], "o", color=COLORS["lp"], ms=9, mec=SURFACE, mew=2)
    ax.annotate(f"C* = {grid[j]:.0f} kW, \\${total[j]:.2f}/day", (grid[j], total[j]), xytext=(-10, 12),
                textcoords="offset points", ha="right", fontsize=9)
    ax.set_ylim(0, total[j] * 3)
    ax.set_xlim(0, grid[-1])
    ax.set_xlabel("Site power cap C (kW)")
    ax.set_ylabel("Expected cost per weekday ($)")
    # Escape "$": matplotlib treats paired dollar signs as math mode.
    ax.set_title(f"Stage 1: choosing the cap (c = \\${c:.2f}/kW-day, λ = \\${lam:g}/kWh)")
    ax.legend(loc="upper right")
    return fig


def cap_vs_penalty(caps: pd.Series, peaks: pd.Series):
    """Optimal two-stage cap as the unmet-energy penalty grows, vs simple planning rules."""
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(caps.index, caps.values, color=COLORS["lp"], marker="o", ms=6, label="two-stage optimal cap C*")
    # Labels sit where the C* line is far away: left for high references, right for the mean.
    for v, txt, at_right in [(peaks.mean(), "mean-day peak", True), (peaks.quantile(0.9), "90th-pct day", False),
                             (peaks.max(), "worst day (robust)", False)]:
        ax.axhline(v, color=INK_2, lw=0.9, ls=":")
        x = caps.index[-1] if at_right else caps.index[0]
        ax.annotate(f"{txt}: {v:.0f} kW", (x, v), xytext=(0, -12 if at_right else 3), textcoords="offset points",
                    ha="right" if at_right else "left", fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("Penalty per kWh not delivered, λ ($/kWh, log scale)")
    ax.set_ylabel("Cap (kW)")
    ax.set_ylim(0, peaks.max() * 1.12)
    ax.set_title("A higher penalty pushes the cap from 'typical day' toward 'worst day'")
    ax.legend(loc="lower right")
    return fig


def policy_costs(tbl: pd.DataFrame):
    """Expected cost per weekday of each cap policy, train vs test."""
    order = ["wait-and-see (perfect info)", "two-stage (SAA)", "mean-day plan (EV)",
             "median-day plan", "worst-case plan (robust)"]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    y = np.arange(len(order))
    for split, off, alpha in [("train", -0.18, 0.55), ("test", 0.18, 1.0)]:
        s = tbl[tbl["split"] == split].set_index("policy").reindex(order)["expected_cost"]
        bars = ax.barh(y + off, s, height=0.34, color=COLORS["lp"], alpha=alpha,
                       label="in-sample (2019)" if split == "train" else "out-of-sample (Jan-Feb 2020)")
        ax.bar_label(bars, fmt="$%.1f", fontsize=8, color=INK_2, padding=3)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("Expected cost per weekday ($): demand charge + unmet-energy penalty")
    ax.grid(axis="y", visible=False)
    ax.set_title("Which cap should the site commit to?", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    return fig


# ---------- online / departure uncertainty ----------

ONLINE_COLORS = {"offline": COLORS["lp"], "mpc_driver": "#6250d6", "mpc_none": "#e87ba4",
                 "equal_share": COLORS["equal_share"], "uncontrolled": COLORS["uncontrolled"],
                 "mpc_true": "#008300", "mpc_conservative": "#eda100"}


def online_frontier(summary: pd.DataFrame):
    """Median peak vs median unmet energy for every policy; families joined as curves.

    summary: index = policy, columns peak_kw, unmet_pct (medians over weekdays).
    """
    fig, ax = plt.subplots(figsize=(10, 5.6))

    def curve(prefixes, key, label):
        s = summary.loc[[p for p in prefixes if p in summary.index]].sort_values("peak_kw")
        ax.plot(s["peak_kw"], s["unmet_pct"], "-o", color=ONLINE_COLORS[key], ms=6, lw=1.8, label=label)
        return s

    eq = curve([p for p in summary.index if p.startswith("equal_share_x")], "equal_share",
               "Equal sharing, cap = 1.0x to 2.0x offline peak (no departure info)")
    for p, r in eq.iterrows():
        ax.annotate(p.replace("equal_share_", ""), (r["peak_kw"], r["unmet_pct"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=INK_2)
    for est, lab in [("driver", "Online LP, driver's stated departure"), ("none", "Online LP, no departure info")]:
        fam = curve([f"mpc_{est}"] + [f"mpc_{est}_rmin{a}A" for a in (8, 12, 16)], f"mpc_{est}",
                    f"{lab} (+ min rate 0/8/12/16 A)")
        for p, r in fam.iterrows():
            tag = p.split("_rmin")[1] if "_rmin" in p else "0A"
            ax.annotate(tag, (r["peak_kw"], r["unmet_pct"]), xytext=(4, -10), textcoords="offset points",
                        fontsize=8, color=INK_2)
    singles = [("offline_lp", "offline", "s", "Offline LP (perfect information)"),
               ("mpc_true", "mpc_true", "D", "Online LP, true departures"),
               ("mpc_conservative", "mpc_conservative", "D", "Online LP, cautious departure estimate"),
               ("uncontrolled", "uncontrolled", "s", "Uncontrolled")]
    for p, key, mk, lab in singles:
        if p in summary.index:
            r = summary.loc[p]
            ax.plot(r["peak_kw"], r["unmet_pct"], mk, color=ONLINE_COLORS[key], ms=9, mec=SURFACE, mew=1.5,
                    label=lab, zorder=5)
    ax.set_xlabel("Median daily peak (kW)")
    ax.set_ylabel("Median energy not delivered (%)")
    ax.set_xlim(left=0)
    # Keep the informative region readable; mark points beyond the cap instead of stretching the axis.
    ymax = 20.0
    ax.set_ylim(-0.5, ymax)
    for p, r in summary[summary["unmet_pct"] > ymax].iterrows():
        ax.annotate(f"{p} (off scale):\n{r['peak_kw']:.0f} kW, {r['unmet_pct']:.0f}% unmet",
                    (r["peak_kw"], ymax), xytext=(3, ymax * 0.8), textcoords="data", ha="left",
                    fontsize=8, color=INK_2, arrowprops={"arrowstyle": "->", "color": INK_2, "lw": 0.8})
    ax.set_title("Peak vs undelivered energy when departures are uncertain (lower-left is better)")
    ax.legend(loc="upper right", fontsize=8.5)
    return fig


def departure_error_hist(err_h: pd.Series):
    """Distribution of (actual - first stated) departure, hours."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    e = err_h.clip(-8, 8)
    ax.hist(e, bins=np.arange(-8, 8.25, 0.25), color=COLORS["lp"], alpha=0.85, edgecolor=SURFACE, lw=0.4)
    ax.axvline(0, color=INK, lw=1)
    top = ax.get_ylim()[1]
    # Labels at staggered heights and sides so close quantiles don't collide.
    for q, txt, y, side in [(0.25, "25th pct", 0.95, "right"), (0.5, "median", 0.78, "left"),
                            (0.75, "75th pct", 0.95, "left")]:
        v = err_h.quantile(q)
        ax.axvline(v, color=INK_2, lw=0.9, ls="--")
        ax.annotate(f"{txt}\n{v:+.1f} h", (v, top * y), xytext=(-4 if side == "right" else 4, 0),
                    textcoords="offset points", ha=side, fontsize=8, color=INK_2, va="top")
    ax.set_xlabel("Actual minus stated departure (hours; negative = left earlier than stated; clipped to ±8)")
    ax.set_ylabel("Sessions")
    ax.set_title("Drivers' stated departure times are rough")
    return fig
