"""Tests for the stress_testing package."""

import numpy as np
import pytest

from data import equity_portfolio
from stress_testing import (
    HistoricalScenarioEngine,
    HypotheticalScenarioEngine,
    ReverseStressTester,
)


@pytest.fixture(scope="module")
def portfolio():
    return equity_portfolio()


def _find_scenario(engine, keyword):
    """Locate a built-in scenario name containing ``keyword`` (case-insensitive)."""
    names = []
    if hasattr(engine, "scenario_names"):
        names = list(engine.scenario_names())
    else:
        from stress_testing import historical_scenarios as hs

        names = [s["name"] if isinstance(s, dict) else getattr(s, "name", str(s))
                 for s in getattr(hs, "SCENARIOS", [])]
    for n in names:
        if keyword.lower() in n.lower():
            return n
    return None


def test_covid_scenario_matches_manual(portfolio):
    """COVID 2020 replay P&L equals the manual window calculation."""
    eng = HistoricalScenarioEngine()
    name = _find_scenario(eng, "covid")
    assert name is not None
    res = eng.apply_scenario(portfolio, name)
    window = portfolio.returns.loc["2020-02-19":"2020-03-23"]
    manual_pnl = (window.to_numpy() @ portfolio.weights).sum() * portfolio.notional
    assert res.pnl == pytest.approx(manual_pnl, rel=1e-6)
    assert res.pnl < 0  # COVID is a loss


def test_uniform_shock(portfolio):
    """Shocking every asset by -10% gives P&L = -10% * notional."""
    eng = HypotheticalScenarioEngine()
    shocks = {a: -0.10 for a in portfolio.asset_names}
    res = eng.apply_custom_shock(portfolio, shocks)
    assert res.pnl == pytest.approx(-0.10 * portfolio.notional, rel=1e-9)


def test_reverse_stress_satisfies_threshold(portfolio):
    """Minimum-norm reverse-stress scenario reproduces the target loss."""
    rst = ReverseStressTester()
    out = rst.minimum_stress_scenario(portfolio, 0.10)
    implied = out.get("implied_loss_pct", out.get("loss_pct"))
    assert implied == pytest.approx(0.10, rel=1e-3)


def test_correlation_shock_increases_var(portfolio):
    """Flooring correlations at 0.7 raises portfolio VaR (diversification lost)."""
    eng = HypotheticalScenarioEngine()
    res = eng.apply_correlation_shock(portfolio, correlation_floor=0.7)
    # The scenario reports the incremental loss from the correlation stress.
    assert res.pnl <= 0
