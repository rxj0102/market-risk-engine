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
