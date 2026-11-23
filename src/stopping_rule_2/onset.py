"""Onset definition: wet spells, dry spells and onset labels.

All functions operate on daily rainfall arrays whose *first* axis is time; any
trailing axes (woredas, ensemble members, ...) are broadcast. Days whose
look-ahead window runs past the end of the array are NaN-safe: they are
reported as False for wet/onset and as NaN-aware for counts.
"""

from __future__ import annotations

import numpy as np

from . import config as C


def _window_sum(r: np.ndarray, n: int) -> np.ndarray:
    """s[t] = sum(r[t:t+n]); NaN where the window runs past the end."""
    r = np.asarray(r, dtype=float)
    cs = np.concatenate([np.zeros((1,) + r.shape[1:]), np.cumsum(r, axis=0)], axis=0)
    out = np.full(r.shape, np.nan)
    if r.shape[0] >= n:
        out[: r.shape[0] - n + 1] = cs[n:] - cs[:-n]
    return out


def wet_spell(r: np.ndarray, n: int = C.WET_SPELL_DAYS, mm: float = C.WET_SPELL_MM) -> np.ndarray:
    """True on day t if r[t:t+n] totals at least `mm`."""
    with np.errstate(invalid="ignore"):
        return _window_sum(r, n) >= mm


def max_dry_run(dry: np.ndarray) -> np.ndarray:
    """Longest run of consecutive True along axis 0."""
    dry = np.asarray(dry, dtype=bool)
    run = np.zeros(dry.shape[1:], dtype=int)
    best = np.zeros(dry.shape[1:], dtype=int)
    for d in dry:
        run = np.where(d, run + 1, 0)
        best = np.maximum(best, run)
    return best


def dry_run_ending(dry: np.ndarray) -> np.ndarray:
    """e[t] = length of the run of dry days ending at (and including) t."""
    dry = np.asarray(dry, dtype=bool)
    out = np.zeros(dry.shape, dtype=int)
    run = np.zeros(dry.shape[1:], dtype=int)
    for i, d in enumerate(dry):
        run = np.where(d, run + 1, 0)
        out[i] = run
    return out


def has_dry_spell(
    r: np.ndarray,
    thr: float = C.DRY_DAY_MM,
    run: int = C.DRY_SPELL_DAYS,
) -> np.ndarray:
    """True if `r` (time on axis 0) contains `run` consecutive days < thr."""
    return max_dry_run(np.asarray(r) < thr) >= run


def dry_spell_ahead(
    r: np.ndarray,
    start_lag: int = C.WET_SPELL_DAYS,
    n: int = C.CHECK_DAYS,
    thr: float = C.DRY_DAY_MM,
    run: int = C.DRY_SPELL_DAYS,
) -> np.ndarray:
    """For each t: does r[t+start_lag : t+start_lag+n] contain a dry spell?

    Returned as float (1/0) with NaN where the window is incomplete.
    """
    r = np.asarray(r, dtype=float)
    dry = (r < thr).astype(float)
    # A dry spell exists in a window iff some length-`run` sub-window is all dry.
    all_dry = _window_sum(dry, run) >= run  # all_dry[s]: r[s:s+run] all dry
    # Window [a, a+n) contains spell starting at s in [a, a+n-run].
    span = n - run + 1
    any_spell = _window_sum(all_dry.astype(float), span) > 0
    out = np.full(r.shape, np.nan)
    T = r.shape[0]
    last = T - (start_lag + n)  # last t with a complete window
    if last >= 0:
        out[: last + 1] = any_spell[start_lag : start_lag + last + 1]
    return out


def onset_label(r: np.ndarray) -> np.ndarray:
    """Onset criteria met on day t: wet spell t..t+2 and no dry spell in t+3..t+23.

    Float array: 1/0, NaN where the look-ahead is incomplete.
    """
    wet = _window_sum(r, C.WET_SPELL_DAYS)
    dry = dry_spell_ahead(r)
    out = np.where(wet >= C.WET_SPELL_MM, 1.0 - dry, 0.0)
    return np.where(np.isnan(dry) | np.isnan(wet), np.nan, out)


def first_true(mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """Index of the first True along axis 0, or -1 if none."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.argmax(mask, axis=axis)
    return np.where(mask.any(axis=axis), idx, -1)
