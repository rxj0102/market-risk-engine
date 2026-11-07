"""Tests for the backtesting package."""

import numpy as np
import pandas as pd
import pytest

from backtesting import BaselTrafficLight, ChristoffersenTest, KupiecTest
from utils.statistics import z_score


def _series(pnl, var):
    idx = pd.bdate_range("2015-01-01", periods=len(pnl))
    return pd.Series(var, index=idx), pd.Series(pnl, index=idx)


def test_exception_count():
    pnl = np.array([-1.0, -3.0, 0.5, -2.5, 1.0])
    var = np.full(5, 2.0)
    var_s, pnl_s = _series(pnl, var)
    n, T, dates = KupiecTest().exception_count(var_s, pnl_s)
    assert n == 2  # -3.0 and -2.5 breach -2.0
    assert T == 5


def test_kupiec_accepts_correct_rate():
    """Exception rate equal to (1-conf) should not be rejected."""
    rng = np.random.default_rng(0)
    T = 2000
    pnl = rng.standard_normal(T)
    var = np.full(T, z_score(0.99))  # exact 99% Normal VaR
    var_s, pnl_s = _series(pnl, var)
    stat, pval, reject = KupiecTest().test(var_s, pnl_s, 0.99)
    assert not reject and pval > 0.05


def test_kupiec_rejects_double_rate():
    """Roughly double the expected exceptions should be rejected."""
    rng = np.random.default_rng(1)
    T = 2000
    pnl = rng.standard_normal(T)
    # VaR set so the breach rate is ~2%: quantile at 98%.
    var = np.full(T, z_score(0.98))
    var_s, pnl_s = _series(pnl, var)
    stat, pval, reject = KupiecTest().test(var_s, pnl_s, 0.99)
    assert reject and pval < 0.05


def test_traffic_light_zones():
    tl = BaselTrafficLight()
    assert tl.classify(0, 250)[0] == "green"
    assert tl.classify(7, 250)[0] == "yellow"
    assert tl.classify(12, 250)[0] == "red"
    assert tl.classify(12, 250)[1] == pytest.approx(4.0)


def test_christoffersen_detects_clustering():
    """Clustered exceptions yield a low independence p-value."""
    # 20 exceptions all bunched at the start -> strong dependence.
    e = np.zeros(500, dtype=int)
    e[10:30] = 1
    stat, pval, trans = ChristoffersenTest().independence_test(pd.Series(e))
    assert pval < 0.05


def test_traffic_light_reproduces_basel_table_for_250_days():
    tl = BaselTrafficLight()
    for n in range(0, 5):
        assert tl.classify(n, 250) == ("green", 3.0)
    expected = {5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}
    for n, mult in expected.items():
        zone, m = tl.classify(n, 250)
        assert zone == "yellow" and m == pytest.approx(mult)
    for n in (10, 11, 20):
        assert tl.classify(n, 250) == ("red", 4.0)


def test_traffic_light_report_is_consistent_for_other_windows():
    tl = BaselTrafficLight()
    rng = np.random.default_rng(0)
    for T in (125, 500, 1000):
        for n_exc in (0, 3, 6, 12, 25):
            pnl = np.zeros(T)
            pnl[:n_exc] = -2.0
            rep = tl.backtest_report(pd.Series(np.ones(T)), pd.Series(pnl))
            assert rep["capital_multiplier"] == pytest.approx(3.0 + rep["capital_add_on"])
            if rep["traffic_light"] == "green":
                assert rep["capital_add_on"] == 0.0
            if rep["traffic_light"] == "red":
                assert rep["capital_add_on"] == 1.0
