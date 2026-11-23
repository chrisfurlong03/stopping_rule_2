"""Feature construction.

Two stages:

1. ``build_base`` (fold-independent): per-year CHIRPS "cubes" (year, t, pcode)
   of labels / wet spells / antecedent rainfall, AIFS ensemble diagnostics for
   every candidate day at a grid of dry-day thresholds, and the long table of
   candidate rows (wet-spell days in the season for model years).
2. ``fold_design`` (per CV fold): climatology-based features and anomalies
   computed leaving out both the held-out test year and the row's own year,
   plus selection of each woreda's AIFS dry-day threshold on training years.

Indexing: ``t`` is the candidate onset day as an offset from May 1. The
extended range T_EXT = -CLIM_HALF_WINDOW .. SEASON_LEN-1+CLIM_HALF_WINDOW is
kept in the CHIRPS cubes so climatologies can be smoothed along the season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import aifs as A
from . import chirps as Ch
from . import config as C
from . import onset as O

H = C.CLIM_HALF_WINDOW
T_EXT = np.arange(-H, C.SEASON_LEN + H)  # offsets kept in cubes
T_SEASON = np.arange(C.SEASON_LEN)
ANOM_HALF_WINDOW = 5
APR1_OFFSET = -30

CLIMO = ["clim_logit_success", "clim_logit_dry7", "clim_log_rain21", "doy_minus_clim_onset"]
OBS = ["rain14_z", "rain30_z", "std_z", "dry14"]
AIFS = [
    "ens_logit_dry7",
    "ens_mean_maxdry",
    "ens_mean_drydays",
    "ens_log_rain21",
    "ens_log_wk1",
    "ens_log_wk2",
    "ens_log_wk3",
    "fcst_age",
]
GROUPS = {
    "climo": CLIMO,
    "climo+obs": CLIMO + OBS,
    "climo+obs+aifs": CLIMO + OBS + AIFS,
}


def _logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _shift_window_sum(r, n, lag):
    """out[i] = sum(r[i+lag : i+lag+n]) (lag may be negative); NaN off-edge."""
    ws = O._window_sum(r, n)
    out = np.full_like(ws, np.nan)
    T = r.shape[0]
    src = np.arange(T) + lag
    ok = (src >= 0) & (src < T)
    out[ok] = ws[src[ok]]
    return out


# --------------------------------------------------------------------------
# Stage 1: fold-independent arrays
# --------------------------------------------------------------------------
def chirps_cubes(df: pd.DataFrame, pcodes, years=C.CHIRPS_YEARS) -> dict:
    """Per-year CHIRPS-derived arrays on T_EXT: shape (year, t, pcode)."""
    R = Ch.season_cube(df, years, pcodes)  # (year, day, pcode), day 0 = Mar 1
    i_ext = T_EXT - C.ARRAY_START_OFFSET
    i_apr1 = APR1_OFFSET - C.ARRAY_START_OFFSET
    keys = ["wet", "label", "dry7", "rain21", "rain14", "rain30", "dry14", "std"]
    out = {k: [] for k in keys}
    for r in R:
        dry = (r < C.DRY_DAY_MM).astype(float)
        cs = np.cumsum(r, axis=0)
        std = np.full(r.shape, np.nan)
        # season-to-date: Apr 1 .. t-1
        std[i_apr1 + 1 :] = cs[i_apr1:-1] - (cs[i_apr1 - 1] if i_apr1 > 0 else 0)
        std[i_apr1] = 0.0
        a = {
            "wet": O.wet_spell(r).astype(float),
            "label": O.onset_label(r),
            "dry7": O.dry_spell_ahead(r),
            "rain21": _shift_window_sum(r, C.CHECK_DAYS, C.WET_SPELL_DAYS),
            "rain14": _shift_window_sum(r, 14, -14),
            "rain30": _shift_window_sum(r, 30, -30),
            "dry14": _shift_window_sum(dry, 14, -14),
            "std": std,
        }
        for k in keys:
            out[k].append(a[k][i_ext])
    cubes = {k: np.stack(v) for k, v in out.items()}
    for k, v in cubes.items():
        if np.isnan(v).any():
            raise ValueError(f"NaN in CHIRPS cube {k}")
    # True onset (first t in season meeting criteria); -1 if none.
    s = slice(H, H + C.SEASON_LEN)
    cubes["onset"] = O.first_true(cubes["label"][:, s] == 1, axis=1)
    cubes["years"] = np.asarray(years)
    return cubes


def aifs_cubes(pcodes, years=C.MODEL_YEARS, thresholds=C.AIFS_DRY_THRESHOLDS) -> dict:
    """AIFS ensemble diagnostics for every (year, t in season, pcode).

    Threshold-dependent arrays have a trailing threshold axis. Also returns
    per-year dry-day counts (pcode, thr) for threshold matching.
    """
    thr = np.asarray(thresholds)
    nY, nT, nP, nK = len(years), C.SEASON_LEN, len(pcodes), len(thr)
    frac7 = np.zeros((nY, nT, nP, nK), "float32")
    maxdry = np.zeros((nY, nT, nP, nK), "float32")
    drydays = np.zeros((nY, nT, nP, nK), "float32")
    rain = {k: np.zeros((nY, nT, nP), "float32") for k in ["rain21", "wk1", "wk2", "wk3"]}
    age = np.zeros((nY, nT), "int16")
    dry_count = np.zeros((nY, nP, nK))
    dry_total = np.zeros(nY)
    for iy, y in enumerate(years):
        a = A.load_year(y, pcodes)
        may1 = pd.Timestamp(y, *C.SEASON_START)
        for t in range(nT):
            D = may1 + pd.Timedelta(days=t + C.DECISION_LAG)
            w, k = a.window(D)  # (member, 21, pcode)
            age[iy, t] = k
            wt = np.moveaxis(w, 1, 0)[..., None] < thr  # (21, member, pcode, K)
            mdr = O.max_dry_run(wt)  # (member, pcode, K)
            frac7[iy, t] = (mdr >= C.DRY_SPELL_DAYS).mean(axis=0)
            maxdry[iy, t] = mdr.mean(axis=0)
            drydays[iy, t] = wt.sum(axis=0).mean(axis=0)
            em = w.mean(axis=0)  # (21, pcode)
            rain["rain21"][iy, t] = em.sum(axis=0)
            rain["wk1"][iy, t] = em[0:7].sum(axis=0)
            rain["wk2"][iy, t] = em[7:14].sum(axis=0)
            rain["wk3"][iy, t] = em[14:21].sum(axis=0)
        # dry-day frequency over each init's leads 0..20 (all members)
        v = a.values[:, :, : C.CHECK_DAYS, :]  # (init, member, lead, pcode)
        dry_count[iy] = (v[..., None] < thr).sum(axis=(0, 1, 2))
        dry_total[iy] = v.shape[0] * v.shape[1] * v.shape[2]
    return {
        "frac7": frac7, "maxdry": maxdry, "drydays": drydays, "age": age,
        "dry_count": dry_count, "dry_total": dry_total,
        "thresholds": thr, "years": np.asarray(years), **rain,
    }


def obs_dry_freq(df: pd.DataFrame, pcodes, years=C.MODEL_YEARS) -> tuple[np.ndarray, np.ndarray]:
    """CHIRPS (<1 mm) dry-day counts on the AIFS verification days: (year, pcode)."""
    obs = df[list(pcodes)].to_numpy(dtype=float)
    cnt = np.zeros((len(years), len(pcodes)))
    tot = np.zeros(len(years))
    for iy, y in enumerate(years):
        a = A.load_year(y, pcodes)
        for init in a.inits:
            j = df.index.get_loc(init + pd.Timedelta(days=C.AIFS_LEAD_OFFSET))
            cnt[iy] += (obs[j : j + C.CHECK_DAYS] < C.DRY_DAY_MM).sum(axis=0)
            tot[iy] += C.CHECK_DAYS
    return cnt, tot


def candidate_rows(ch: dict, years=C.MODEL_YEARS) -> pd.DataFrame:
    """Long table of wet-spell days t in season for model years."""
    yi = np.searchsorted(ch["years"], years)
    wet = ch["wet"][yi][:, H : H + C.SEASON_LEN] == 1  # (year, t, pcode)
    iy, t, ip = np.nonzero(wet)
    lab = ch["label"][yi][:, H : H + C.SEASON_LEN]
    onset = ch["onset"][yi][iy, ip]
    return pd.DataFrame({
        # candidates the stopping rule faces: on/before the true onset
        "pre_onset": (t <= onset) | (onset < 0),
        "true_onset": onset,
        "year": np.asarray(years)[iy],
        "iy_model": iy,  # index into MODEL_YEARS / aifs cubes
        "iy_chirps": yi[iy],  # index into CHIRPS cube years
        "t": t,
        "ip": ip,
        "y": lab[iy, t, ip].astype(int),
    })


# --------------------------------------------------------------------------
# Stage 2: fold-dependent design matrix
# --------------------------------------------------------------------------
def _box(x, h, axis=1):
    """Moving sum over +/-h along axis (output trimmed by h on each side)."""
    cs = np.cumsum(x, axis=axis)
    cs = np.concatenate([np.zeros_like(np.take(cs, [0], axis=axis)), cs], axis=axis)
    n = x.shape[axis]
    hi = np.take(cs, np.arange(2 * h + 1, n + 1), axis=axis)
    lo = np.take(cs, np.arange(0, n - 2 * h), axis=axis)
    return hi - lo


class Climatology:
    """Leave-years-out climatologies via sums of per-year contributions."""

    def __init__(self, ch: dict):
        self.ch = ch
        wet, lab = ch["wet"], ch["label"]
        # per-year smoothed sums over +/-H, restricted to season t (year, t, p)
        self.n_wet = _box(wet, H)
        self.n_succ = _box(lab * wet, H)
        self.s_dry7 = _box(ch["dry7"], H)
        self.s_rain21 = _box(ch["rain21"], H)
        self.n_days = 2 * H + 1
        # anomaly stats on a narrower window
        h2 = ANOM_HALF_WINDOW
        trim = H - h2
        self.anom = {}
        for k in ["rain14", "rain30", "std"]:
            x = ch[k]
            s1 = _box(x, h2)[:, trim : trim + C.SEASON_LEN]
            s2 = _box(x * x, h2)[:, trim : trim + C.SEASON_LEN]
            self.anom[k] = (s1, s2)
        self.n_anom = 2 * h2 + 1
        on = ch["onset"].astype(float)
        on[on < 0] = C.SEASON_LEN  # no onset in window -> treat as end of season
        self.onset = on  # (year, p)

    def for_excluding(self, exclude: list[int]) -> dict:
        """Climatology fields (t, p) using all CHIRPS years except `exclude`."""
        keep = ~np.isin(self.ch["years"], exclude)
        nyr = keep.sum()
        n_wet = self.n_wet[keep].sum(0)
        out = {
            "p_success": (self.n_succ[keep].sum(0) + 0.5) / (n_wet + 1.0),
            "p_dry7": self.s_dry7[keep].sum(0) / (nyr * self.n_days),
            "rain21": self.s_rain21[keep].sum(0) / (nyr * self.n_days),
            "onset": np.median(self.onset[keep], axis=0),
        }
        n = nyr * self.n_anom
        for k, (s1, s2) in self.anom.items():
            m = s1[keep].sum(0) / n
            v = np.maximum(s2[keep].sum(0) / n - m * m, 1e-6)
            out[f"{k}_mean"], out[f"{k}_sd"] = m, np.sqrt(v)
        return out


def choose_aifs_thresholds(ac: dict, obs_cnt, obs_tot, train_mask) -> np.ndarray:
    """Per-pcode index into thresholds matching CHIRPS dry-day frequency."""
    f_obs = obs_cnt[train_mask].sum(0) / obs_tot[train_mask].sum()  # (p,)
    f_fc = ac["dry_count"][train_mask].sum(0) / ac["dry_total"][train_mask].sum()  # (p, K)
    return np.argmin(np.abs(f_fc - f_obs[:, None]), axis=1)


def fold_design(rows, clim: Climatology, ac: dict, obs_cnt, obs_tot, test_year: int) -> pd.DataFrame:
    """Full feature matrix for all rows, for the CV fold holding out test_year."""
    ch = clim.ch
    t, ip = rows["t"].to_numpy(), rows["ip"].to_numpy()
    iyc = rows["iy_chirps"].to_numpy()
    year = rows["year"].to_numpy()
    cols = CLIMO + ["rain14_z", "rain30_z", "std_z", "clim_p_success"]
    F = {c: np.full(len(rows), np.nan) for c in cols}
    for y in np.unique(year):
        m = year == y
        cl = clim.for_excluding([test_year, y])
        tt, pp, yy = t[m], ip[m], iyc[m]
        F["clim_logit_success"][m] = _logit(cl["p_success"][tt, pp])
        F["clim_logit_dry7"][m] = _logit(cl["p_dry7"][tt, pp])
        F["clim_log_rain21"][m] = np.log1p(cl["rain21"][tt, pp])
        F["doy_minus_clim_onset"][m] = tt - cl["onset"][pp]
        for k, name in [("rain14", "rain14_z"), ("rain30", "rain30_z"), ("std", "std_z")]:
            x = ch[k][yy, tt + H, pp]
            F[name][m] = (x - cl[f"{k}_mean"][tt, pp]) / cl[f"{k}_sd"][tt, pp]
        F["clim_p_success"][m] = cl["p_success"][tt, pp]
    X = pd.DataFrame(F, index=rows.index)
    X["dry14"] = ch["dry14"][iyc, t + H, ip]

    train_mask = ac["years"] != test_year
    kthr = choose_aifs_thresholds(ac, obs_cnt, obs_tot, train_mask)[ip]
    iym = rows["iy_model"].to_numpy()
    n_mem = 25
    f7 = ac["frac7"][iym, t, ip, kthr]
    X["ens_logit_dry7"] = _logit((f7 * n_mem + 0.5) / (n_mem + 1))
    X["ens_mean_maxdry"] = ac["maxdry"][iym, t, ip, kthr]
    X["ens_mean_drydays"] = ac["drydays"][iym, t, ip, kthr]
    X["ens_frac_dry7"] = f7
    X["ens_log_rain21"] = np.log1p(ac["rain21"][iym, t, ip])
    for w in ["wk1", "wk2", "wk3"]:
        X[f"ens_log_{w}"] = np.log1p(ac[w][iym, t, ip])
    X["fcst_age"] = ac["age"][iym, t]
    X["aifs_thr"] = ac["thresholds"][kthr]
    return X


# --------------------------------------------------------------------------
# Orchestration / caching
# --------------------------------------------------------------------------
def common_pcodes(df: pd.DataFrame) -> list[str]:
    return [p for p in df.columns if p in set(A.pcodes())]


def build_base(path=C.PROCESSED_DIR / "base.npz"):
    df = Ch.load_woreda_series()
    pcodes = common_pcodes(df)
    ch = chirps_cubes(df, pcodes)
    ac = aifs_cubes(pcodes)
    obs_cnt, obs_tot = obs_dry_freq(df, pcodes)
    np.savez_compressed(
        path,
        pcodes=np.asarray(pcodes),
        obs_cnt=obs_cnt,
        obs_tot=obs_tot,
        **{f"ch_{k}": v for k, v in ch.items()},
        **{f"ac_{k}": v for k, v in ac.items()},
    )
    return load_base(path)


def load_base(path=C.PROCESSED_DIR / "base.npz"):
    z = np.load(path)
    ch = {k[3:]: z[k] for k in z.files if k.startswith("ch_")}
    ac = {k[3:]: z[k] for k in z.files if k.startswith("ac_")}
    return {
        "pcodes": z["pcodes"].astype(str),
        "ch": ch,
        "ac": ac,
        "obs_cnt": z["obs_cnt"],
        "obs_tot": z["obs_tot"],
        "rows": candidate_rows(ch),
    }
