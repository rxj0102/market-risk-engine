"""Tests for parametric (variance-covariance) VaR."""

import numpy as np
import pandas as pd
import pytest

from core import Portfolio
from data import equity_portfolio
from utils.statistics import z_score
from var import ParametricVaR


@pytest.fixture(scope="module")
def portfolio():
    return equity_portfolio()


def _single_asset(portfolio):
    w = np.array([1.0])
    return Portfolio(
        weights=w,
        asset_names=[portfolio.asset_names[0]],
        returns=portfolio.returns.iloc[:, [0]],
        notional=portfolio.notional,
    )


def test_single_asset_formula(portfolio):
    """VaR(99%) = z * sigma * notional for a single asset."""
    p = _single_asset(portfolio)
    res = ParametricVaR().compute(p, confidence=0.99, cov_method="sample")
    sigma = p.returns.iloc[:, 0].std()
    expected = z_score(0.99) * sigma * p.notional
    assert res.var == pytest.approx(expected, rel=1e-3)


def test_component_var_additivity(portfolio):
    """Sum of analytical component VaRs equals total parametric VaR."""
    pv = ParametricVaR()
    comp = pv.component_var_analytical(portfolio, confidence=0.99)
    total = pv.compute(portfolio, confidence=0.99, cov_method="sample").var
    assert np.sum(comp) == pytest.approx(total, rel=1e-6)


def test_diversification(portfolio):
    """Portfolio VaR is less than the sum of standalone asset VaRs."""
    pv = ParametricVaR()
    port_var = pv.compute(portfolio, confidence=0.99, cov_method="sample").var
    standalone = 0.0
    for i, name in enumerate(portfolio.asset_names):
        sigma = portfolio.returns.iloc[:, i].std()
        standalone += z_score(0.99) * sigma * portfolio.weights[i] * portfolio.notional
    assert port_var < standalone


def test_student_t_heavier_than_normal(portfolio):
    pv = ParametricVaR()
    normal = pv.compute(portfolio, confidence=0.99).var
    tvar = pv.compute_student_t(portfolio, confidence=0.99).var
    assert tvar > normal


def test_ewma_differs_from_sample(portfolio):
    pv = ParametricVaR()
    sample = pv.compute(portfolio, confidence=0.99, cov_method="sample").var
    ewma = pv.compute(portfolio, confidence=0.99, cov_method="ewma").var
    assert ewma > 0 and not np.isclose(ewma, sample)
