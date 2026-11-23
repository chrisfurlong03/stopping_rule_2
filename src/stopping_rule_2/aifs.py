"""AIFS ENS reforecast loading (already woreda-aggregated)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class AifsYear:
    year: int
    inits: pd.DatetimeIndex  # (n_init,)
    leads: np.ndarray  # (n_lead,)
    pcodes: np.ndarray  # (n_pcode,)
    values: np.ndarray  # (n_init, n_member, n_lead, n_pcode) mm/day

    def latest_init(self, day: pd.Timestamp) -> tuple[int, int]:
        """Index of the latest init <= day, and forecast age in days."""
        i = int(np.searchsorted(self.inits.values, np.datetime64(day), side="right")) - 1
        if i < 0:
            raise ValueError(f"no AIFS init on or before {day}")
        return i, int((day - self.inits[i]).days)

    def window(self, day: pd.Timestamp, n: int = C.CHECK_DAYS) -> tuple[np.ndarray, int]:
        """Member forecasts for calendar days day..day+n-1: (member, n, pcode), age."""
        i, age = self.latest_init(day)
        l0 = age - C.AIFS_LEAD_OFFSET
        if l0 < 0 or l0 + n > len(self.leads):
            raise ValueError(f"{day}: leads {l0}..{l0 + n - 1} unavailable")
        return self.values[i, :, l0 : l0 + n, :], age


def pcodes(year: int = C.MODEL_YEARS[0]) -> np.ndarray:
    with np.load(C.AIFS_DIR / C.AIFS_PATTERN.format(year=year)) as z:
        return z["pcodes"].astype(str)


def load_year(year: int, keep_pcodes=None) -> AifsYear:
    with np.load(C.AIFS_DIR / C.AIFS_PATTERN.format(year=year)) as z:
        pc = z["pcodes"].astype(str)
        vals = z["values"]
        inits = pd.DatetimeIndex(z["init"])
        leads = z["lead"]
    if keep_pcodes is not None:
        idx = pd.Index(pc).get_indexer(list(keep_pcodes))
        if (idx < 0).any():
            raise KeyError("requested pcodes missing from AIFS")
        vals, pc = vals[..., idx], pc[idx]
    order = np.argsort(inits.values)
    return AifsYear(year, inits[order], leads, pc, np.ascontiguousarray(vals[order]))


def check_lead_alignment(chirps: pd.DataFrame, years=C.MODEL_YEARS, offsets=(-1, 0, 1, 2)):
    """Correlate AIFS ens-mean at each lead with CHIRPS on init+lead+offset.

    Returns DataFrame (lead x offset) of correlations pooled over woredas/inits,
    plus a table of dry-day frequencies (AIFS by threshold vs CHIRPS <1 mm).
    """
    common = [p for p in pcodes(years[0]) if p in chirps.columns]
    obs = chirps[common].to_numpy(dtype="float32", copy=True)
    day_index = chirps.index
    leads = range(8)
    fc = {(L, off): [] for L in leads for off in offsets}
    ob = {(L, off): [] for L in leads for off in offsets}
    for y in years:
        a = load_year(y, common)
        em = a.values.mean(axis=1)  # (init, lead, pcode)
        for i, init in enumerate(a.inits):
            for L in leads:
                for off in offsets:
                    j = day_index.get_loc(init + pd.Timedelta(days=L + off))
                    fc[(L, off)].append(em[i, L].copy())
                    ob[(L, off)].append(obs[j].copy())
    out = {
        k: np.corrcoef(np.concatenate(fc[k]), np.concatenate(ob[k]))[0, 1] for k in fc
    }
    corr = pd.Series(out).unstack()
    corr.index.name, corr.columns.name = "lead", "offset"
    return corr
