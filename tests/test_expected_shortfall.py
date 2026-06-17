"""Tests for Expected Shortfall estimators."""

import numpy as np
import pandas as pd
import pytest

from core import Portfolio
from data import equity_portfolio
from expected_shortfall import HistoricalES, ParametricES
from utils.statistics import normal_pdf_at_quantile, z_score


@pytest.fixture(scope="module")
def portfolio():
    return equity_portfolio()


def test_es_ge_var(portfolio):
    """ES is at least as large as VaR at the same confidence."""
    res = HistoricalES().compute(portfolio, confidence=0.99)
    assert res.es >= res.var


def test_normal_es_var_ratio(portfolio):
    """Parametric-Normal ES/VaR ratio equals phi(z)/((1-conf) z)."""
    res = ParametricES().compute(portfolio, confidence=0.99, distribution="normal")
    z = z_score(0.99)
    expected_ratio = normal_pdf_at_quantile(0.99) / ((1 - 0.99) * z)
    assert res.es_var_ratio == pytest.approx(expected_ratio, rel=1e-3)


def test_historical_es_matches_manual(portfolio):
    """Historical ES equals the mean of losses beyond VaR."""
    res = HistoricalES().compute(portfolio, confidence=0.99,
                                 lookback=len(portfolio.returns))
    pnl = portfolio.pnl().to_numpy()
    var = res.var
    tail = pnl[pnl < -var]
    manual_es = -tail.mean()
    assert res.es == pytest.approx(manual_es, rel=1e-6)


def test_subadditivity(portfolio):
    """Parametric-Normal ES is subadditive: ES(A+B) <= ES(A)+ES(B)."""
    pe = ParametricES()
    a = Portfolio(np.array([1.0]), [portfolio.asset_names[0]],
                  portfolio.returns.iloc[:, [0]], portfolio.notional)
    b = Portfolio(np.array([1.0]), [portfolio.asset_names[4]],
                  portfolio.returns.iloc[:, [4]], portfolio.notional)
    ab = Portfolio(np.array([0.5, 0.5]), [portfolio.asset_names[0], portfolio.asset_names[4]],
                   portfolio.returns.iloc[:, [0, 4]], portfolio.notional)
    es_a = pe.compute(a, 0.99, "normal").es
    es_b = pe.compute(b, 0.99, "normal").es
    es_ab = pe.compute(ab, 0.99, "normal").es
    assert es_ab <= 0.5 * es_a + 0.5 * es_b + 1e-6


def test_cornish_fisher_vs_normal_negative_skew():
    """Cornish-Fisher ES exceeds Normal ES for a negatively-skewed portfolio."""
    rng = np.random.default_rng(0)
    # Build a strongly left-skewed return series.
    base = rng.standard_normal(2000) * 0.01
    jumps = -np.abs(rng.standard_normal(2000)) * 0.03 * (rng.random(2000) < 0.05)
    rets = base + jumps
    df = pd.DataFrame({"X": rets}, index=pd.bdate_range("2015-01-01", periods=2000))
    p = Portfolio(np.array([1.0]), ["X"], df, 1_000_000.0)
    pe = ParametricES()
    cf = pe.cornish_fisher_es(p, 0.99).es
    normal = pe.compute(p, 0.99, "normal").es
    assert cf > normal
