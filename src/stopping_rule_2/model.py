"""Calibrated logistic regression with leave-one-year-out cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import features as F

CAL_METHODS = ("sigmoid", "isotonic")
THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.01), 2)


def base_estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def calibrated(method: str, X, y, groups, n_splits=C.INNER_FOLDS):
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    return CalibratedClassifierCV(base_estimator(), method=method, cv=splits).fit(X, y)


def pss_curve(y, p, thresholds=THRESHOLDS):
    y = np.asarray(y, bool)
    pos, neg = y.sum(), (~y).sum()
    out = []
    for t in thresholds:
        yes = p >= t
        pod = (yes & y).sum() / max(pos, 1)
        pofd = (yes & ~y).sum() / max(neg, 1)
        out.append(pod - pofd)
    return np.asarray(out)


def best_threshold(y, p):
    return float(THRESHOLDS[int(np.argmax(pss_curve(y, p)))])


def fit_fold_group(X_tr, y_tr, g_tr):
    """Choose calibration method + threshold by inner CV; fit final model."""
    outer = list(GroupKFold(n_splits=C.INNER_FOLDS).split(X_tr, y_tr, g_tr))
    best = None
    for method in CAL_METHODS:
        oof = np.zeros(len(y_tr))
        for tr, va in outer:
            m = calibrated(method, X_tr[tr], y_tr[tr], g_tr[tr], n_splits=C.INNER_FOLDS - 1)
            oof[va] = m.predict_proba(X_tr[va])[:, 1]
        bs = np.mean((oof - y_tr) ** 2)
        if best is None or bs < best[1]:
            best = (method, bs, oof)
    method, _, oof = best
    thr = best_threshold(y_tr, oof)
    model = calibrated(method, X_tr, y_tr, g_tr)
    return model, method, thr


def coefficients(model, cols) -> pd.Series:
    """Mean standardized LR coefficients across the calibration sub-models."""
    coefs = [c.estimator[-1].coef_[0] for c in model.calibrated_classifiers_]
    return pd.Series(np.mean(coefs, axis=0), index=cols)


def run_fold(base, clim, test_year, groups=F.GROUPS):
    rows = base["rows"]
    X = F.fold_design(rows, clim, base["ac"], base["obs_cnt"], base["obs_tot"], test_year)
    train = (rows["year"] != test_year).to_numpy() & rows["pre_onset"].to_numpy()
    test = (rows["year"] == test_year).to_numpy()
    y = rows["y"].to_numpy()
    g = rows["year"].to_numpy()
    out = rows.loc[test, ["year", "t", "ip", "y", "pre_onset", "true_onset"]].copy()
    out["clim_p_success"] = X.loc[test, "clim_p_success"].to_numpy()
    out["base_rate_train"] = y[train].mean()
    info, coefs = [], {}
    for name, cols in groups.items():
        Xa = X[cols].to_numpy()
        model, method, thr = fit_fold_group(Xa[train], y[train], g[train])
        out[f"p_{name}"] = model.predict_proba(Xa[test])[:, 1]
        out[f"thr_{name}"] = thr
        info.append({"test_year": test_year, "group": name, "method": method, "threshold": thr})
        coefs[name] = coefficients(model, cols)
    return out, info, coefs


def cross_validate(base, years=C.MODEL_YEARS, n_jobs=-1):
    clim = F.Climatology(base["ch"])
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_fold)(base, clim, y) for y in years
    )
    oof = pd.concat([r[0] for r in res], ignore_index=True)
    info = pd.DataFrame([i for r in res for i in r[1]])
    coefs = {
        g: pd.concat([r[2][g] for r in res], axis=1, keys=years).T for g in F.GROUPS
    }
    return oof, info, coefs


def fit_final(base, group="climo+obs+aifs"):
    """Operational model trained on all model years (climatology from all years)."""
    clim = F.Climatology(base["ch"])
    rows = base["rows"]
    X = F.fold_design(rows, clim, base["ac"], base["obs_cnt"], base["obs_tot"], test_year=-1)
    cols = F.GROUPS[group]
    tr = rows["pre_onset"].to_numpy()
    model, method, thr = fit_fold_group(
        X.loc[tr, cols].to_numpy(), rows.loc[tr, "y"].to_numpy(), rows.loc[tr, "year"].to_numpy()
    )
    return model, method, thr, cols
