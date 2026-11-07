"""Tests for historical-simulation VaR."""

import numpy as np
import pytest

from data import equity_portfolio
from var import HistoricalVaR


@pytest.fixture(scope="module")
def portfolio():
    return equity_portfolio()


def test_var_non_negative(portfolio):
    res = HistoricalVaR().compute(portfolio, confidence=0.99)
    assert res.var > 0
    assert res.var_pct > 0


def test_exception_rate_in_sample(portfolio):
    """In-sample breach frequency should be close to (1-confidence)."""
    hv = HistoricalVaR()
    res = hv.compute(portfolio, confidence=0.95, lookback_days=len(portfolio.returns))
    pnl = portfolio.pnl().to_numpy()
    breaches = np.mean(pnl < -res.var)
    assert breaches == pytest.approx(0.05, abs=0.02)


def test_sqrt_time_scaling(portfolio):
    """10-day sqrt-scaled VaR ~= 1-day VaR * sqrt(10) within tolerance."""
    hv = HistoricalVaR()
    v1 = hv.compute(portfolio, confidence=0.99, horizon_days=1).var
    v10 = hv.compute(portfolio, confidence=0.99, horizon_days=10).var
    assert v10 == pytest.approx(v1 * np.sqrt(10), rel=0.10)


def test_age_weighted_runs(portfolio):
    res = HistoricalVaR().compute_age_weighted(portfolio, confidence=0.99, lambda_=0.98)
    assert res.var > 0
    assert "weight" in res.method.lower() or "age" in res.method.lower()


def test_vol_scaled_responds_to_current_vol(portfolio):
    """Vol-scaled VaR should differ from plain historical (vol regime aware)."""
    hv = HistoricalVaR()
    plain = hv.compute(portfolio, confidence=0.99).var
    scaled = hv.compute_vol_scaled(portfolio, confidence=0.99).var
    assert scaled > 0
    assert not np.isclose(scaled, plain)


def test_multi_horizon_table(portfolio):
    df = HistoricalVaR().compute_multi_horizon(portfolio, confidence=0.99, horizons=[1, 5, 10])
    assert len(df) >= 3
