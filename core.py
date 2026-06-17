"""Core shared data structures and constants for the market risk engine.

This module defines the dataclasses and named constants used across every
sub-package (``var``, ``expected_shortfall``, ``backtesting``, ``stress_testing``,
``risk_decomposition``, ``capital``, ``models``, ``diagnostics``).

References
----------
Basel Committee on Banking Supervision (BCBS).
    "Minimum capital requirements for market risk." (FRTB), Jan 2019.
J.P. Morgan/Reuters. "RiskMetrics Technical Document." 4th ed., 1996.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Named constants (see CODE QUALITY section of the project specification)
# ---------------------------------------------------------------------------
RISKMETRICS_LAMBDA: float = 0.94
"""RiskMetrics EWMA decay factor for daily covariance estimation."""

BASEL_CONFIDENCE: float = 0.99
"""Basel II/III VaR confidence level."""

FRTB_CONFIDENCE: float = 0.975
"""Basel IV (FRTB) Expected Shortfall confidence level."""

TRADING_DAYS: int = 252
"""Number of trading days in a year (annualization factor)."""

BASE_CAPITAL_MULTIPLIER: float = 3.0
"""Base Basel market-risk capital multiplier (mc)."""

FELLER_THRESHOLD: float = 2.0
"""Minimum degrees of freedom for a Student-t distribution with finite variance."""

AGE_WEIGHT_LAMBDA: float = 0.98
"""Default decay factor for age-weighted (BRW) historical simulation."""

DEFAULT_SEED: int = 42
"""Default RNG seed for reproducible Monte Carlo simulation."""


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
@dataclass
class Portfolio:
    """A market portfolio with weights and a historical return panel.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, ideally summing to 1.
    asset_names : list[str]
        Names of the assets, aligned with ``weights`` and ``returns`` columns.
    returns : pd.DataFrame
        Historical simple return series indexed by date (dates x assets).
    notional : float
        Total portfolio value in ``currency`` units (USD by default).
    currency : str
        Reporting currency. Defaults to ``'USD'``.

    Notes
    -----
    Portfolio P&L on day ``t`` is ``sum_i(w_i * r_{i,t}) * notional``.
    """

    weights: np.ndarray
    asset_names: list[str]
    returns: pd.DataFrame
    notional: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1-D array")
        if len(self.weights) != len(self.asset_names):
            raise ValueError("weights and asset_names must have equal length")
        if self.returns.shape[1] != len(self.asset_names):
            raise ValueError("returns columns must match number of assets")

    # -- Derived quantities -------------------------------------------------
    @property
    def n_assets(self) -> int:
        """Number of assets in the portfolio."""
        return len(self.asset_names)

    def pnl(self) -> pd.Series:
        """Historical portfolio P&L in currency units.

        Returns
        -------
        pd.Series
            ``sum_i(w_i r_{i,t}) * notional`` indexed by date.
        """
        port_ret = self.returns.to_numpy() @ self.weights
        return pd.Series(port_ret * self.notional, index=self.returns.index, name="pnl")

    def return_series(self) -> pd.Series:
        """Historical portfolio simple-return series (no notional scaling)."""
        port_ret = self.returns.to_numpy() @ self.weights
        return pd.Series(port_ret, index=self.returns.index, name="portfolio_return")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class VaRResult:
    """Container for a Value-at-Risk estimate.

    Attributes
    ----------
    var : float
        VaR expressed as a positive dollar loss.
    var_pct : float
        VaR as a fraction of portfolio notional.
    confidence : float
        Confidence level, e.g. ``0.99``.
    horizon_days : int
        Risk horizon in trading days.
    method : str
        Name of the estimation method.
    n_scenarios : int
        Number of scenarios / observations used.
    elapsed_seconds : float
        Wall-clock computation time.
    """

    var: float
    var_pct: float
    confidence: float
    horizon_days: int
    method: str
    n_scenarios: int
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"VaR[{self.method}] {self.confidence:.1%} {self.horizon_days}d = "
            f"${self.var:,.0f} ({self.var_pct:.2%})"
        )


@dataclass
class ESResult:
    """Container for an Expected Shortfall (CVaR) estimate.

    Attributes
    ----------
    es : float
        Expected Shortfall as a positive dollar loss.
    es_pct : float
        ES as a fraction of portfolio notional.
    var : float
        The corresponding VaR (dollar loss) at the same confidence.
    confidence : float
        Confidence level, e.g. ``0.99``.
    method : str
        Name of the estimation method.
    tail_scenarios : np.ndarray
        Loss scenarios beyond VaR used to compute ES.
    horizon_days : int
        Risk horizon in trading days.
    """

    es: float
    es_pct: float
    var: float
    confidence: float
    method: str
    tail_scenarios: np.ndarray = field(default_factory=lambda: np.array([]))
    horizon_days: int = 1

    @property
    def es_var_ratio(self) -> float:
        """Ratio of ES to VaR (>= 1 by construction)."""
        return self.es / self.var if self.var else float("nan")

    def __str__(self) -> str:
        return (
            f"ES[{self.method}] {self.confidence:.1%} = ${self.es:,.0f} "
            f"({self.es_pct:.2%}), ES/VaR={self.es_var_ratio:.3f}"
        )


@dataclass
class BacktestResult:
    """Container for VaR backtesting diagnostics.

    Attributes
    ----------
    exceptions : int
        Number of VaR breaches observed.
    expected_exceptions : float
        Expected number of breaches under the null ``(1-conf) * days``.
    exception_rate : float
        Observed breach frequency.
    kupiec_stat, kupiec_pvalue : float
        Kupiec POF likelihood-ratio statistic and p-value.
    christoffersen_stat, christoffersen_pvalue : float
        Christoffersen conditional-coverage statistic and p-value.
    traffic_light : str
        Basel zone: ``'green'``, ``'yellow'`` or ``'red'``.
    days_tested : int
        Number of out-of-sample test days.
    capital_multiplier : float
        Basel capital multiplier implied by the traffic-light zone.
    """

    exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christoffersen_stat: float
    christoffersen_pvalue: float
    traffic_light: str
    days_tested: int
    capital_multiplier: float = BASE_CAPITAL_MULTIPLIER

    def __str__(self) -> str:
        return (
            f"Backtest[{self.traffic_light.upper()}] {self.exceptions}/"
            f"{self.days_tested} exceptions (exp {self.expected_exceptions:.1f}), "
            f"Kupiec p={self.kupiec_pvalue:.3f}"
        )


@dataclass
class ScenarioResult:
    """Container for a stress-test scenario outcome.

    Attributes
    ----------
    name : str
        Scenario label.
    pnl : float
        Total scenario P&L in dollars (negative = loss).
    pnl_pct : float
        Scenario P&L as a fraction of notional.
    max_1d_loss : float
        Largest single-day dollar loss within the scenario window.
    max_drawdown : float
        Maximum peak-to-trough drawdown over the window (fraction).
    total_loss : float
        Total period loss in dollars (positive number).
    asset_pnl : np.ndarray
        Per-asset P&L contributions in dollars.
    description : str
        Human-readable scenario description.
    recovery_days : int
        Trading days to recover prior peak (``-1`` if not recovered).
    """

    name: str
    pnl: float
    pnl_pct: float
    max_1d_loss: float = 0.0
    max_drawdown: float = 0.0
    total_loss: float = 0.0
    asset_pnl: np.ndarray = field(default_factory=lambda: np.array([]))
    description: str = ""
    recovery_days: int = -1

    def __str__(self) -> str:
        return f"Scenario[{self.name}] P&L = ${self.pnl:,.0f} ({self.pnl_pct:.2%})"
