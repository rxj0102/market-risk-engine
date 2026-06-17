"""Hypothetical (forward-looking) stress testing.

Applies analyst-defined instantaneous shocks to the portfolio's assets.
Unlike :mod:`historical_scenarios`, these scenarios are not tied to realised
history and express expert views ("what if equities fall 30%?").

References
----------
Basel Committee on Banking Supervision. "Principles for sound stress testing
    practices and supervision." May 2009.
Kupiec, P. (1998). "Stress testing in a value at risk framework."
    *Journal of Derivatives* 6(1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import BASEL_CONFIDENCE, Portfolio, ScenarioResult
from models import CovarianceEstimator
from utils.statistics import z_score

# ---------------------------------------------------------------------------
# Pre-built parallel-shock library (asset -> shock as a simple return)
# ---------------------------------------------------------------------------
_BROAD_EQUITY = ("SPY", "QQQ", "IWM", "VNQ", "XLF", "EEM")

HYPOTHETICAL_SCENARIOS: dict[str, dict[str, float]] = {
    "Equity Crash": {
        "SPY": -0.30, "QQQ": -0.30, "IWM": -0.30, "VNQ": -0.30,
        "XLF": -0.30, "EEM": -0.30, "HYG": -0.15, "USO": -0.20,
        "GLD": 0.05, "TLT": 0.08,
    },
    "Rate Spike": {
        "TLT": -0.15, "HYG": -0.06, "XLF": 0.03,
    },
    "Dollar Surge": {
        "EEM": -0.20, "USO": -0.10, "GLD": -0.08,
    },
    "Oil Spike": {
        "USO": 0.50, "XLF": 0.02,
        "SPY": -0.03, "QQQ": -0.03, "IWM": -0.03, "VNQ": -0.03, "EEM": -0.03,
    },
    "Credit Event": {
        "HYG": -0.15, "TLT": 0.06,
        "SPY": -0.20, "QQQ": -0.20, "IWM": -0.20, "VNQ": -0.20,
        "XLF": -0.20, "EEM": -0.20,
    },
    "Stagflation": {
        "SPY": -0.20, "QQQ": -0.20, "IWM": -0.20, "VNQ": -0.20,
        "XLF": -0.20, "EEM": -0.20, "TLT": -0.10, "GLD": 0.12, "USO": 0.15,
    },
    "Tech Rout": {
        "QQQ": -0.40, "SPY": -0.10, "IWM": -0.08, "XLF": -0.05,
        "VNQ": -0.05, "EEM": -0.06,
    },
    "Bank Stress": {
        "XLF": -0.40, "HYG": -0.10, "SPY": -0.12,
    },
    "EM Crisis": {
        "EEM": -0.35, "USO": -0.12, "SPY": -0.08,
    },
}


class HypotheticalScenarioEngine:
    """Apply forward-looking parallel shocks to a portfolio.

    Carries a library of named macro scenarios
    (:data:`HYPOTHETICAL_SCENARIOS`) and supports arbitrary custom shocks and
    a correlation-breakdown stress on parametric VaR.
    """

    def __init__(self, scenarios: dict[str, dict[str, float]] | None = None) -> None:
        self.scenarios = (
            dict(scenarios) if scenarios is not None else dict(HYPOTHETICAL_SCENARIOS)
        )

    # ------------------------------------------------------------- core shock
    def apply_custom_shock(
        self, portfolio: Portfolio, shocks_dict: dict[str, float], name: str = "Custom Shock"
    ) -> ScenarioResult:
        """Apply an arbitrary asset-level shock vector.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to stress.
        shocks_dict : dict
            Mapping ``asset -> shock`` (simple return).  Assets absent from
            the mapping receive a zero shock.
        name : str
            Scenario label.

        Returns
        -------
        ScenarioResult
            P&L is ``sum_i w_i * shock_i * notional`` over portfolio assets.

        Notes
        -----
        Because weights sum to one, shocking *all* assets by the same ``x``
        yields portfolio P&L of exactly ``x * notional``.
        """
        shock_vec = np.array(
            [float(shocks_dict.get(a, 0.0)) for a in portfolio.asset_names], dtype=float
        )
        asset_pnl = portfolio.weights * shock_vec * portfolio.notional
        total = float(asset_pnl.sum())
        return ScenarioResult(
            name=name,
            pnl=total,
            pnl_pct=total / portfolio.notional,
            max_1d_loss=float(-total) if total < 0 else 0.0,
            max_drawdown=abs(total) / portfolio.notional if total < 0 else 0.0,
            total_loss=float(-total) if total < 0 else 0.0,
            asset_pnl=asset_pnl,
            description=f"Custom instantaneous shock applied to {len(shocks_dict)} asset(s).",
            recovery_days=-1,
        )

    def apply_scenario(self, portfolio: Portfolio, scenario_name: str) -> ScenarioResult:
        """Apply a named library scenario to ``portfolio``.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to stress.
        scenario_name : str
            Name of a scenario in :data:`HYPOTHETICAL_SCENARIOS`.

        Returns
        -------
        ScenarioResult
        """
        if scenario_name not in self.scenarios:
            raise KeyError(f"Unknown hypothetical scenario {scenario_name!r}")
        res = self.apply_custom_shock(
            portfolio, self.scenarios[scenario_name], name=scenario_name
        )
        res.description = f"Hypothetical macro scenario: {scenario_name}."
        return res

    # --------------------------------------------------- correlation breakdown
    def apply_correlation_shock(
        self,
        portfolio: Portfolio,
        correlation_floor: float = 0.7,
        confidence: float = BASEL_CONFIDENCE,
    ) -> ScenarioResult:
        """Stress parametric VaR by flooring all pairwise correlations.

        Builds a stressed covariance in which every off-diagonal correlation
        is raised to at least ``correlation_floor`` (variances unchanged),
        modelling a crisis-driven diversification breakdown.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to stress.
        correlation_floor : float
            Minimum pairwise correlation to impose (e.g. ``0.7``).
        confidence : float
            VaR confidence level (default Basel 99%).

        Returns
        -------
        ScenarioResult
            ``pnl`` is the (negative) increase in 99% parametric VaR:
            ``-(stressed_VaR - base_VaR)``.
        """
        estimator = CovarianceEstimator()
        base_cov = estimator.sample_covariance(portfolio.returns).to_numpy()

        std = np.sqrt(np.diag(base_cov))
        outer = np.outer(std, std)
        corr = base_cov / outer
        np.fill_diagonal(corr, 1.0)

        # Floor off-diagonal correlations, preserving the diagonal.
        stressed_corr = np.maximum(corr, correlation_floor)
        np.fill_diagonal(stressed_corr, 1.0)
        stressed_cov = stressed_corr * outer

        z = z_score(confidence)
        w = portfolio.weights
        base_vol = float(np.sqrt(w @ base_cov @ w))
        stressed_vol = float(np.sqrt(w @ stressed_cov @ w))
        base_var = z * base_vol * portfolio.notional
        stressed_var = z * stressed_vol * portfolio.notional
        delta = stressed_var - base_var

        return ScenarioResult(
            name=f"Correlation Floor {correlation_floor:.0%}",
            pnl=-delta,
            pnl_pct=-delta / portfolio.notional,
            max_1d_loss=stressed_var,
            max_drawdown=0.0,
            total_loss=delta if delta > 0 else 0.0,
            asset_pnl=np.array([]),
            description=(
                f"Diversification breakdown: all correlations floored to "
                f">= {correlation_floor:.2f}. Base 99% VaR=${base_var:,.0f}, "
                f"stressed=${stressed_var:,.0f}, increase=${delta:,.0f}."
            ),
            recovery_days=-1,
        )

    # ----------------------------------------------------- run_all_hypothetical
    def run_all_hypothetical(self, portfolio: Portfolio) -> pd.DataFrame:
        """Apply every named library scenario and tabulate the P&L.

        Returns
        -------
        pd.DataFrame
            Columns ``Scenario``, ``P&L($)``, ``P&L(%)``.
        """
        rows: list[dict[str, object]] = []
        for sc in self.scenarios:
            res = self.apply_scenario(portfolio, sc)
            rows.append(
                {"Scenario": sc, "P&L($)": res.pnl, "P&L(%)": res.pnl_pct}
            )
        return pd.DataFrame(rows)

    # ---------------------------------------------------- scenario_builder
    @staticmethod
    def scenario_builder_interface(asset_names: list[str]) -> dict[str, float]:
        """Return a zero-initialised shock template for the given assets.

        Parameters
        ----------
        asset_names : list[str]
            Assets to include in the template.

        Returns
        -------
        dict
            ``{asset: 0.0}`` for each asset, ready to edit and pass to
            :meth:`apply_custom_shock`.
        """
        return {a: 0.0 for a in asset_names}
