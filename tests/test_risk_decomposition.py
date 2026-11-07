"""Tests for the risk_decomposition package."""

import numpy as np
import pytest

from data import equity_portfolio
from core import Portfolio
from risk_decomposition import ComponentVaR, MarginalVaR, FactorRiskDecomposition
from models import CovarianceEstimator
from utils.statistics import z_score


@pytest.fixture(scope="module")
def portfolio():
    return equity_portfolio()


def test_component_var_sums_to_total(portfolio):
    """Component VaR must add up to the total parametric VaR (additivity)."""
    cv = ComponentVaR()
    df = cv.compute(portfolio, confidence=0.99, cov_method="sample")
    comp_col = [c for c in df.columns if "ComponentVaR" in c and "$" in c][0]
    total = df[comp_col].sum()

    cov = CovarianceEstimator().sample_covariance(portfolio.returns).to_numpy()
    w = portfolio.weights
    sigma_p = np.sqrt(w @ cov @ w)
    expected = z_score(0.99) * sigma_p * portfolio.notional
    assert total == pytest.approx(expected, rel=1e-6)


def test_component_pct_sums_to_one(portfolio):
    cv = ComponentVaR()
    df = cv.compute(portfolio, confidence=0.99, cov_method="sample")
    pct_col = [c for c in df.columns if "ComponentVaR" in c and "%" in c][0]
    assert df[pct_col].sum() == pytest.approx(1.0, abs=1e-6)


def test_marginal_var_equal_weight_approximately_equal(portfolio):
    """For an equal-weight portfolio marginal VaRs are of comparable size."""
    mv = MarginalVaR().marginal_var(portfolio, confidence=0.99, cov_method="sample")
    assert np.all(np.isfinite(mv))
    # Not identical (assets differ) but same order of magnitude / all positive-ish spread.
    assert mv.std() / np.abs(mv.mean()) < 2.0


def test_fully_concentrated_component_equals_total():
    """100% in one asset: its component VaR equals the total VaR."""
    base = equity_portfolio()
    w = np.zeros(base.n_assets)
    w[0] = 1.0
    p = Portfolio(weights=w, asset_names=base.asset_names, returns=base.returns, notional=base.notional)
    cv = ComponentVaR()
    df = cv.compute(p, confidence=0.99, cov_method="sample")
    comp_col = [c for c in df.columns if "ComponentVaR" in c and "$" in c][0]
    total = df[comp_col].sum()
    assert df[comp_col].iloc[0] == pytest.approx(total, rel=1e-6)


def test_diversification_benefit_single_vs_multi():
    """Single asset => no diversification; multi-asset => positive benefit."""
    base = equity_portfolio()
    w = np.zeros(base.n_assets)
    w[0] = 1.0
    single = Portfolio(weights=w, asset_names=base.asset_names, returns=base.returns, notional=base.notional)
    cv = ComponentVaR()
    assert cv.diversification_ratio(single) == pytest.approx(0.0, abs=1e-6)
    assert cv.diversification_ratio(base) > 0.0


def test_pca_eigenvalues_sum_to_trace(portfolio):
    """Sum of eigenvalues equals trace of covariance; variance explained sums to 1."""
    fac = FactorRiskDecomposition()
    out = fac.pca_decomposition(portfolio, n_factors=5)
    cov = CovarianceEstimator().sample_covariance(portfolio.returns).to_numpy()
    assert np.sum(out["eigenvalues"]) == pytest.approx(np.trace(cov), rel=1e-6)
    assert np.sum(out["variance_explained"]) == pytest.approx(1.0, abs=1e-6)
