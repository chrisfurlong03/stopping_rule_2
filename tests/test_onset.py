import numpy as np

from stopping_rule_2 import onset as O


def series(n=40, base=5.0):
    return np.full(n, base)


def test_wet_spell_threshold():
    r = np.zeros(10)
    r[2:5] = [10, 5, 5]  # exactly 20
    assert O.wet_spell(r)[2]
    r[4] = 4.9  # 19.9
    assert not O.wet_spell(r)[2]


def test_wet_spell_incomplete_window_false():
    r = np.full(4, 30.0)
    w = O.wet_spell(r)
    assert w[0] and w[1] and not w[2] and not w[3]


def test_six_day_dry_run_passes_seven_fails():
    r = series()
    r[0:3] = 10  # wet spell t=0, window days 3..23
    r[5:11] = 0  # 6 dry days
    assert O.onset_label(r)[0] == 1
    r[11] = 0.99  # 7th dry day (<1 mm)
    assert O.onset_label(r)[0] == 0


def test_dry_day_threshold_is_strict():
    r = series()
    r[0:3] = 10
    r[5:12] = 1.0  # exactly 1 mm is not dry
    assert O.onset_label(r)[0] == 1


def test_dry_spell_at_window_edges():
    r = series()
    r[0:3] = 10
    r[3:10] = 0  # starts at first day of window
    assert O.onset_label(r)[0] == 0
    r = series()
    r[0:3] = 10
    r[17:24] = 0  # ends at last day of window (t+23)
    assert O.onset_label(r)[0] == 0
    r = series()
    r[0:3] = 10
    r[18:25] = 0  # last day falls outside window -> only 6 dry inside
    assert O.onset_label(r)[0] == 1


def test_incomplete_lookahead_is_nan():
    r = series(30)
    lab = O.onset_label(r)
    assert not np.isnan(lab[30 - 24])
    assert np.isnan(lab[30 - 23])


def test_label_zero_when_not_wet():
    r = series(40, base=2.0)  # 3-day total 6 mm
    assert np.all(O.onset_label(r)[:17] == 0)


def test_broadcast_over_columns():
    r = np.stack([series(), series()], axis=1)
    r[0:3] = 10
    r[5:12, 1] = 0
    lab = O.onset_label(r)
    assert lab[0, 0] == 1 and lab[0, 1] == 0


def test_first_true():
    m = np.array([[0, 0], [1, 0], [1, 0]], dtype=bool)
    assert list(O.first_true(m)) == [1, -1]


def test_max_dry_run():
    d = np.array([1, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert O.max_dry_run(d) == 3
    assert list(O.dry_run_ending(d)) == [1, 2, 0, 1, 2, 3, 0]
