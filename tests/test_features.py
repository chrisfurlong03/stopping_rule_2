import numpy as np

from stopping_rule_2 import config as C
from stopping_rule_2 import features as F


def synthetic_cubes(n_years=6, n_p=3, seed=0):
    rng = np.random.default_rng(seed)
    T = len(F.T_EXT)
    wet = (rng.random((n_years, T, n_p)) < 0.4).astype(float)
    label = wet * (rng.random((n_years, T, n_p)) < 0.6)
    ch = {
        "wet": wet,
        "label": label,
        "dry7": (rng.random((n_years, T, n_p)) < 0.3).astype(float),
        "rain21": rng.gamma(2, 50, (n_years, T, n_p)),
        "rain14": rng.gamma(2, 20, (n_years, T, n_p)),
        "rain30": rng.gamma(2, 40, (n_years, T, n_p)),
        "dry14": rng.integers(0, 14, (n_years, T, n_p)).astype(float),
        "std": rng.gamma(2, 80, (n_years, T, n_p)),
        "onset": rng.integers(-1, C.SEASON_LEN, (n_years, n_p)),
        "years": np.arange(2000, 2000 + n_years),
    }
    return ch


def test_climatology_excludes_years_exactly():
    ch = synthetic_cubes()
    clim = F.Climatology(ch)
    excl = [2001, 2004]
    cl = clim.for_excluding(excl)
    keep = ~np.isin(ch["years"], excl)
    H = F.H
    for t in [0, 17, C.SEASON_LEN - 1]:
        sl = slice(t, t + 2 * H + 1)  # ext indices t-H..t+H
        n_wet = ch["wet"][keep, sl].sum(axis=(0, 1))
        n_succ = ch["label"][keep, sl].sum(axis=(0, 1))
        np.testing.assert_allclose(cl["p_success"][t], (n_succ + 0.5) / (n_wet + 1.0))
        np.testing.assert_allclose(cl["p_dry7"][t], ch["dry7"][keep, sl].mean(axis=(0, 1)))
        h2 = F.ANOM_HALF_WINDOW
        s2 = slice(t + H - h2, t + H + h2 + 1)
        x = ch["rain14"][keep, s2]
        np.testing.assert_allclose(cl["rain14_mean"][t], x.mean(axis=(0, 1)))
        np.testing.assert_allclose(cl["rain14_sd"][t], x.std(axis=(0, 1)), rtol=1e-6)
    on = ch["onset"][keep].astype(float)
    on[on < 0] = C.SEASON_LEN
    np.testing.assert_allclose(cl["onset"], np.median(on, axis=0))


def test_threshold_matching_picks_closest_frequency():
    ac = {
        "dry_count": np.array([[[10, 30, 50]], [[10, 30, 50]]], float),  # (year, p, K)
        "dry_total": np.array([100.0, 100.0]),
    }
    obs_cnt = np.array([[32.0], [28.0]])
    obs_tot = np.array([100.0, 100.0])
    k = F.choose_aifs_thresholds(ac, obs_cnt, obs_tot, np.array([True, True]))
    assert k.tolist() == [1]
