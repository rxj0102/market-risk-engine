"""Tests for Monte Carlo VaR."""

import numpy as np
import pytest

from data import equity_portfolio
from var import MonteCarloVaR, ParametricVaR


@pytest.fixture(scope="module")
def portfolio():
    return equity_portfolio()


def test_mc_converges_to_parametric(portfolio):
    """MC-Normal VaR converges to parametric Normal VaR for large n_paths."""
    mc = MonteCarloVaR().compute(portfolio, confidence=0.99, n_paths=200_000,
                                 distribution="normal", seed=7).var
    para = ParametricVaR().compute(portfolio, confidence=0.99).var
    assert mc == pytest.approx(para, rel=0.05)


def test_antithetic_runs_and_agrees(portfolio):
    """Antithetic estimator should be close to the standard MC estimate."""
    mc = MonteCarloVaR()
    standard = mc.compute(portfolio, confidence=0.99, n_paths=100_000, seed=1).var
    vr = mc.compute_with_vr(portfolio, confidence=0.99, n_paths=100_000, method="antithetic").var
    assert vr == pytest.approx(standard, rel=0.07)


def test_path_dependent_short_horizon(portfolio):
    """Path-dependent 1-day VaR is consistent with single-period MC."""
    mc = MonteCarloVaR()
    single = mc.compute(portfolio, confidence=0.99, horizon_days=1, n_paths=50_000, seed=3).var
    path = mc.compute_path_dependent(portfolio, confidence=0.99, horizon_days=1,
                                     n_paths=50_000, n_steps=1).var
    assert path == pytest.approx(single, rel=0.15)


def test_tail_sensitivity_t_vs_normal(portfolio):
    """At 99.9% the Student-t MC VaR materially exceeds Normal (fat tails)."""
    mc = MonteCarloVaR()
    normal = mc.compute(portfolio, confidence=0.999, n_paths=200_000,
                        distribution="normal", seed=11).var
    student = mc.compute(portfolio, confidence=0.999, n_paths=200_000,
                        distribution="student_t", seed=11).var
    assert student / normal > 1.2
