"""Verification: probabilistic, deterministic and stopping-rule metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from . import features as F

BASELINE = "wet_spell"  # p = 1 for every wet-spell day


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def contingency(y, yes):
    y, yes = np.asarray(y, bool), np.asarray(yes, bool)
    a = int((yes & y).sum())  # hits
    b = int((yes & ~y).sum())  # false alarms
    c = int((~yes & y).sum())  # misses
    d = int((~yes & ~y).sum())  # correct negatives
    pod = a / max(a + c, 1)
    pofd = b / max(b + d, 1)
    far = b / max(a + b, 1)
    return {"hits": a, "false_alarms": b, "misses": c, "correct_neg": d,
            "POD": pod, "POFD": pofd, "FAR": far, "PSS": pod - pofd}


def prob_scores(df: pd.DataFrame, group: str) -> dict:
    y = df["y"].to_numpy()
    p = df[f"p_{group}"].to_numpy()
    bs = brier(y, p)
    bs_ws = brier(y, np.ones_like(p))
    bs_clim = brier(y, df["base_rate_train"].to_numpy())
    out = {
        "n": len(y),
        "base_rate": float(y.mean()),
        "BS": bs,
        "BS_wet_spell_baseline": bs_ws,
        "BS_climatology": bs_clim,
        "BSS_vs_wet_spell": 1 - bs / bs_ws,
        "BSS_vs_climatology": 1 - bs / bs_clim,
        "AUC": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan,
    }
    yes = p >= df[f"thr_{group}"].to_numpy()
    out.update(contingency(y, yes))
    # The wet-spell baseline always says yes: POD = POFD = 1, PSS = 0.
    out["PSS_wet_spell_baseline"] = contingency(y, np.ones_like(yes))["PSS"]
    return out


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                         "mean_p": float(p[m].mean()), "obs_freq": float(y[m].mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Stopping rule
# --------------------------------------------------------------------------
def simulate_stopping_rule(df: pd.DataFrame, group: str | None) -> pd.DataFrame:
    """Per woreda-year: declare at the first wet-spell day with p >= threshold.

    group=None -> baseline: declare at the first wet-spell day.
    """
    d = df.sort_values(["year", "ip", "t"])
    if group is None:
        decl = d
    else:
        decl = d[d[f"p_{group}"] >= d[f"thr_{group}"]]
    first = decl.groupby(["year", "ip"], sort=False).first()
    allwy = d.groupby(["year", "ip"], sort=False)["true_onset"].first()
    res = pd.DataFrame({"true_onset": allwy})
    res["declared"] = first["t"].reindex(res.index)
    res["declared_ok"] = first["y"].reindex(res.index)
    res["true_onset"] = res["true_onset"].where(res["true_onset"] >= 0)
    res["error"] = res["declared"] - res["true_onset"]
    return res.reset_index()


def stopping_scores(sr: pd.DataFrame) -> dict:
    declared = sr["declared"].notna()
    both = declared & sr["true_onset"].notna()
    e = sr.loc[both, "error"]
    return {
        "n_woreda_years": len(sr),
        "declared_rate": float(declared.mean()),
        "false_onset_rate": float((sr.loc[declared, "declared_ok"] == 0).mean()),
        "MAE_days": float(e.abs().mean()),
        "bias_days": float(e.mean()),
        "median_error_days": float(e.median()),
        "within_7d": float((e.abs() <= 7).mean()),
        "early_rate": float((e < 0).mean()),
        "late_rate": float((e > 0).mean()),
        "exact_rate": float((e == 0).mean()),
    }


# --------------------------------------------------------------------------
# Bootstrap by year
# --------------------------------------------------------------------------
def bootstrap(oof: pd.DataFrame, groups, n=C.N_BOOTSTRAP, seed=C.RANDOM_SEED) -> pd.DataFrame:
    """Year-block bootstrap CIs for key scores (pre-onset set + stopping rule)."""
    rng = np.random.default_rng(seed)
    years = np.sort(oof["year"].unique())
    pre = oof[oof["pre_onset"]]
    by_year_pre = {y: g for y, g in pre.groupby("year")}
    sr = {g: simulate_stopping_rule(oof, g) for g in groups}
    sr[BASELINE] = simulate_stopping_rule(oof, None)
    sr_by_year = {k: {y: g for y, g in v.groupby("year")} for k, v in sr.items()}
    recs = []
    for _ in range(n):
        ys = rng.choice(years, size=len(years), replace=True)
        P = pd.concat([by_year_pre[y] for y in ys])
        rec = {}
        yv = P["y"].to_numpy()
        bs_ws = brier(yv, 1.0)
        for g in groups:
            rec[f"BSS_vs_wet_spell[{g}]"] = 1 - brier(yv, P[f"p_{g}"]) / bs_ws
        for k in sr:
            S = pd.concat([sr_by_year[k][y] for y in ys])
            s = stopping_scores(S)
            rec[f"false_onset_rate[{k}]"] = s["false_onset_rate"]
            rec[f"MAE_days[{k}]"] = s["MAE_days"]
        recs.append(rec)
    B = pd.DataFrame(recs)
    return B.quantile([0.025, 0.5, 0.975]).T.rename(columns={0.025: "lo", 0.5: "median", 0.975: "hi"})


def per_woreda_bss(oof: pd.DataFrame, group: str, pcodes) -> pd.DataFrame:
    pre = oof[oof["pre_onset"]]
    rows = []
    for ip, g in pre.groupby("ip"):
        y = g["y"].to_numpy()
        bs_ws = brier(y, 1.0)
        rows.append({
            "pcode": pcodes[ip], "n": len(y), "base_rate": y.mean(),
            "BSS_vs_wet_spell": 1 - brier(y, g[f"p_{group}"]) / bs_ws if bs_ws > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def evaluate_all(oof: pd.DataFrame, groups=tuple(F.GROUPS)) -> dict:
    pre = oof[oof["pre_onset"]]
    out = {"probabilistic_pre_onset": {}, "probabilistic_all_wet_days": {}, "stopping_rule": {}}
    for g in groups:
        out["probabilistic_pre_onset"][g] = prob_scores(pre, g)
        out["probabilistic_all_wet_days"][g] = prob_scores(oof, g)
        out["stopping_rule"][g] = stopping_scores(simulate_stopping_rule(oof, g))
    out["stopping_rule"][BASELINE] = stopping_scores(simulate_stopping_rule(oof, None))
    return out
