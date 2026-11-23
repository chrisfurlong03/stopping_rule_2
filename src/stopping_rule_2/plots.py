"""Static verification figures (PNG) + CSV table twins.

Palette: reference instance from the dataviz skill (light mode). Categorical
slots 1-3 (blue, orange, aqua) are validated all-pairs; the wet-spell baseline
is drawn in secondary ink so it never takes a series hue.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from . import config as C  # noqa: E402
from . import evaluate as E  # noqa: E402
from . import features as F  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"climo": "#2a78d6", "climo+obs": "#eb6834", "climo+obs+aifs": "#1baf7a"}
LABEL = {"climo": "Climatology", "climo+obs": "Climatology + obs",
         "climo+obs+aifs": "Climatology + obs + AIFS", E.BASELINE: "Wet-spell baseline"}
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "bss", ["#a3302f", "#e34948", NEUTRAL, "#2a78d6", "#104281"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _out(name):
    d = C.OUTPUT_DIR / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK2)


def reliability_figure(pre: pd.DataFrame, groups):
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.4, 7.2), height_ratios=[3, 1], sharex=True)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
    tables = []
    for g in groups:
        r = E.reliability(pre["y"].to_numpy(), pre[f"p_{g}"].to_numpy())
        r.insert(0, "group", g)
        tables.append(r)
        ax.plot(r.mean_p, r.obs_freq, color=SERIES[g], marker="o", ms=8,
                mec=SURFACE, mew=2, label=LABEL[g], zorder=3)
        axh.step(np.r_[r.bin_lo, 1.0], np.r_[r.n, r.n.iloc[-1]], where="post",
                 color=SERIES[g], lw=2)
    ax.set_ylabel("Observed frequency of onset")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability: wet-spell days up to true onset")
    ax.legend(loc="upper left")
    axh.set_ylabel("Cases per bin")
    axh.set_xlabel("Forecast probability that the day is onset")
    fig.tight_layout()
    fig.savefig(_out("reliability.png"), dpi=150)
    plt.close(fig)
    pd.concat(tables).to_csv(_out("reliability.csv"), index=False)


def roc_figure(pre: pd.DataFrame, groups, metrics):
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    for g in groups:
        fpr, tpr, _ = roc_curve(pre["y"], pre[f"p_{g}"])
        auc = metrics["probabilistic_pre_onset"][g]["AUC"]
        ax.plot(fpr, tpr, color=SERIES[g], label=f"{LABEL[g]} (AUC {auc:.2f})")
    # Wet-spell baseline is the single point POFD = POD = 1.
    ax.plot([1], [1], marker="o", ms=8, color=INK2, mec=SURFACE, mew=2, clip_on=False)
    ax.annotate("Wet-spell baseline (always yes)", (1, 1), xytext=(-10, -14),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("False-alarm rate (POFD)")
    ax.set_ylabel("Hit rate (POD)")
    ax.set_title("ROC: wet-spell days up to true onset")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(_out("roc.png"), dpi=150)
    plt.close(fig)


OUTCOME_COLORS = {  # adjacent pairlist (segmented bar) -> slots in fixed order
    "False onset (early)": "#e34948",
    "Exact day": "#2a78d6",
    "Late": "#eda100",
    "Not declared": "#898781",
}


def stopping_outcomes(oof, groups) -> pd.DataFrame:
    rows = []
    for k in [E.BASELINE, *groups]:
        sr = E.simulate_stopping_rule(oof, None if k == E.BASELINE else k)
        dec = sr["declared"].notna()
        e = sr["error"]
        rows.append({
            "rule": LABEL[k],
            "False onset (early)": (dec & (sr["declared_ok"] == 0)).mean(),
            "Exact day": (e == 0).mean(),
            "Late": (e > 0).mean(),
            "Not declared": (~dec).mean(),
        })
    return pd.DataFrame(rows).set_index("rule")


def stopping_figure(oof, groups):
    tab = stopping_outcomes(oof, groups)
    tab.to_csv(_out("stopping_rule_outcomes.csv"))
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ypos = np.arange(len(tab))[::-1]
    h = 0.42
    for yi, (rule, r) in zip(ypos, tab.iterrows()):
        left = 0.0
        for k, v in r.items():
            if v <= 0:
                continue
            ax.barh(yi, v - 0.004, left=left, height=h, color=OUTCOME_COLORS[k], lw=0)
            if v >= 0.045:
                txt = "#0b0b0b" if k in ("Late",) else "#ffffff"
                ax.text(left + 0.008, yi, f"{v:.0%}", va="center", ha="left",
                        fontsize=9, color=txt)
            left += v
    ax.set_yticks(ypos, tab.index)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of woreda-years (2000-2024)")
    ax.set_title("Stopping-rule outcome vs true onset")
    handles = [matplotlib.patches.Patch(color=c, label=k) for k, c in OUTCOME_COLORS.items()]
    ax.legend(handles=handles, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.28))
    fig.tight_layout()
    fig.savefig(_out("stopping_rule_outcomes.png"), dpi=150)
    plt.close(fig)


def error_figure(oof, group="climo+obs+aifs"):
    """Distribution of non-zero onset-date errors (exact-day share in caption)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bins = np.arange(-125, 126, 5)
    rec, exact = {}, {}
    for k, color in [(E.BASELINE, INK2), (group, SERIES[group])]:
        sr = E.simulate_stopping_rule(oof, None if k == E.BASELINE else k)
        e = sr["error"].dropna()
        exact[k] = (e == 0).sum() / len(sr)
        cnt, _ = np.histogram(e[e != 0], bins=bins)
        frac = cnt / len(sr)
        ax.step(np.r_[bins[:-1], bins[-1]], np.r_[frac, frac[-1]], where="post",
                color=color, label=LABEL[k])
        rec[LABEL[k]] = frac
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlim(-125, 125)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Declared onset minus true onset (days, 5-day bins; exact-day cases omitted)")
    ax.set_ylabel("Share of woreda-years")
    ax.set_title("When the stopping rule misses, by how much")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(_out("onset_error_hist.png"), dpi=150)
    plt.close(fig)
    pd.DataFrame(rec, index=pd.Index(bins[:-1], name="error_bin_start_days")).to_csv(
        _out("onset_error_hist.csv"))


def bss_maps(oof, pcodes):
    from . import chirps as Ch

    full = E.per_woreda_bss(oof, "climo+obs+aifs", pcodes)
    noaifs = E.per_woreda_bss(oof, "climo+obs", pcodes)
    tab = full.merge(noaifs[["pcode", "BSS_vs_wet_spell"]], on="pcode", suffixes=("", "_no_aifs"))
    tab["AIFS_gain"] = tab["BSS_vs_wet_spell"] - tab["BSS_vs_wet_spell_no_aifs"]
    tab.to_csv(_out("bss_by_woreda.csv"), index=False)
    w = Ch.load_woredas().merge(tab, on="pcode", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    panels = [
        ("BSS_vs_wet_spell", "BSS vs wet-spell baseline (full model)", 1.0),
        ("AIFS_gain", "BSS gain from adding AIFS", 0.4),
    ]
    for ax, (col, title, lim) in zip(axes, panels):
        v = w[col]
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        w[v.isna()].plot(ax=ax, color=GRID, edgecolor=SURFACE, linewidth=0.3)
        w[v.notna()].plot(ax=ax, column=col, cmap=DIVERGING, norm=norm,
                          edgecolor=SURFACE, linewidth=0.3)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
        cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, extend="both")
        cb.outline.set_visible(False)
        cb.ax.tick_params(color=MUTED, labelcolor=INK2)
        ax.set_title(title)
        ax.set_axis_off()
    fig.text(0.01, 0.02, "Gray: not modelled (woreda absent from the AIFS file, or no May-Aug wet-spell "
             "candidates). Leave-one-year-out predictions, wet-spell days up to true onset.", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(_out("bss_maps.png"), dpi=150)
    plt.close(fig)


def coefficient_figure(group="climo+obs+aifs"):
    path = C.OUTPUT_DIR / f"coefficients_{group.replace('+', '_')}.csv"
    if not path.exists():
        return
    cf = pd.read_csv(path, index_col=0)
    s = cf.agg(["mean", "min", "max"]).T.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * len(s) + 1.2))
    y = np.arange(len(s))
    ax.hlines(y, s["min"], s["max"], color="#86b6ef", lw=2)
    ax.plot(s["mean"], y, "o", ms=8, color=SERIES["climo"], mec=SURFACE, mew=2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, s.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Standardized log-odds coefficient (dot: mean; bar: range over CV folds)")
    ax.set_title("Feature effects on P(onset), full model")
    fig.tight_layout()
    fig.savefig(_out("coefficients.png"), dpi=150)
    plt.close(fig)


def after_onset_table(oof, group="climo+obs+aifs", max_days=60, width=3) -> pd.DataFrame:
    """Observed onset frequency vs mean forecast, by days relative to true onset."""
    d = oof[oof["true_onset"] >= 0].copy()
    d["k"] = d["t"] - d["true_onset"]
    d = d[(d["k"] >= -max_days) & (d["k"] <= max_days)]
    d["bin"] = (np.floor(d["k"] / width) * width).astype(int)
    d.loc[d["k"] == 0, "bin"] = 0
    d.loc[(d["k"] > 0) & (d["k"] < width), "bin"] = 1
    d["below"] = (d[f"p_{group}"] < d[f"thr_{group}"]).astype(float)
    return d.groupby("bin").agg(n=("y", "size"), observed=("y", "mean"),
                                forecast=(f"p_{group}", "mean"),
                                share_below_threshold=("below", "mean"))


def after_onset_figure(oof, group="climo+obs+aifs"):
    tab = after_onset_table(oof, group)
    tab.to_csv(_out("forecast_vs_days_from_onset.csv"))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = tab.index.to_numpy()
    ax.axvline(0, color=AXIS, lw=1)
    ax.plot(x, tab["observed"], color=INK2, marker="o", ms=4, label="Observed share meeting onset criteria")
    ax.plot(x, tab["forecast"], color=SERIES[group], marker="o", ms=4,
            label="Mean forecast probability (full model)")
    thr = float(oof[f"thr_{group}"].median())
    ax.axhline(thr, color=AXIS, lw=1)
    ax.annotate(f"Median declaration threshold {thr:.2f}", (x.min(), thr), xytext=(2, 4),
                textcoords="offset points", fontsize=9, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Candidate day minus true onset (days; 0 = true onset)")
    ax.set_ylabel("Share / probability")
    ax.set_title("Forecasts stay cautious just after the true onset")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(_out("forecast_vs_days_from_onset.png"), dpi=150)
    plt.close(fig)


def make_all(oof: pd.DataFrame, groups=tuple(F.GROUPS)):
    import json

    with open(C.OUTPUT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    pcodes = F.load_base()["pcodes"]
    pre = oof[oof["pre_onset"]]
    reliability_figure(pre, groups)
    roc_figure(pre, groups, metrics)
    stopping_figure(oof, groups)
    error_figure(oof)
    after_onset_figure(oof)
    bss_maps(oof, pcodes)
    coefficient_figure()
    print(f"figures -> {C.OUTPUT_DIR / 'figures'}")
