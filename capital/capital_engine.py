"""Unified capital aggregator.

Combines economic capital, Basel III and Basel IV (FRTB) regulatory capital,
risk-adjusted performance and the diversification benefit into a single report.
"""

from __future__ import annotations

import pandas as pd

from core import FRTB_CONFIDENCE, Portfolio

from capital.economic_capital import EconomicCapitalEngine
from capital.rapm import RiskAdjustedPerformance
from capital.regulatory_capital import BaselCapitalEngine


class CapitalEngine:
    """One-stop aggregator across the capital sub-package.

    Parameters
    ----------
    cov_method : str, default 'sample'
        Covariance estimator propagated to the component engines.
    """

    def __init__(self, cov_method: str = "sample") -> None:
        self._ec = EconomicCapitalEngine(cov_method=cov_method)
        self._reg = BaselCapitalEngine(cov_method=cov_method)
        self._rapm = RiskAdjustedPerformance(cov_method=cov_method)

    def full_report(
        self,
        portfolio: Portfolio,
        expected_return: float = 0.08,
        confidence: float = 0.999,
    ) -> dict:
        """Aggregate capital report.

        Parameters
        ----------
        portfolio : Portfolio
        expected_return : float, default 0.08
            Expected annual return as a fraction of notional.
        confidence : float, default 0.999
            Economic-capital confidence level.

        Returns
        -------
        dict
            Nested report with ``economic_capital``, ``regulatory`` (Basel III
            and IV), ``rapm`` and ``diversification_benefit`` sections.
        """
        ec = self._ec.economic_capital(portfolio, confidence=confidence)
        expected_pnl = expected_return * portfolio.notional

        basel3 = self._reg.var_capital_charge(portfolio, confidence=0.99)
        frtb = self._reg.frtb_capital(portfolio, confidence=FRTB_CONFIDENCE)
        frtb_base = self._reg.frtb_es_capital(portfolio, confidence=FRTB_CONFIDENCE)

        sharpes = self._rapm.all_sharpe_ratios(portfolio, expected_return)
        raroc = self._rapm.raroc(portfolio, expected_pnl, ec)

        return {
            "economic_capital": {
                "confidence": confidence,
                "ec": ec,
                "ec_by_asset": self._ec.ec_by_asset(portfolio, confidence=confidence),
            },
            "regulatory": {
                "basel_iii": basel3,
                "frtb_base_es": frtb_base,
                "frtb_lh_scaled": frtb["total_charge"],
                "comparison": self._reg.capital_comparison(portfolio),
            },
            "rapm": {
                "expected_pnl": expected_pnl,
                "raroc": raroc,
                **sharpes,
            },
            "diversification_benefit": self._ec.diversification_benefit(
                portfolio, confidence=confidence
            ),
        }

    def summary_table(self, portfolio: Portfolio) -> pd.DataFrame:
        """Compact one-row-per-metric capital summary.

        Parameters
        ----------
        portfolio : Portfolio

        Returns
        -------
        pd.DataFrame
            Columns ``Metric`` and ``Value``.
        """
        report = self.full_report(portfolio)
        ec = report["economic_capital"]["ec"]
        basel3 = report["regulatory"]["basel_iii"]["total"]
        frtb_base = report["regulatory"]["frtb_base_es"]
        frtb_lh = report["regulatory"]["frtb_lh_scaled"]
        rows = [
            ("Economic Capital (99.9%, 1Y)", ec),
            ("Basel III MRC (VaR + SVaR)", basel3),
            ("FRTB ES (97.5%, base)", frtb_base),
            ("FRTB ES (LH-scaled)", frtb_lh),
            ("Diversification Benefit", report["diversification_benefit"]),
            ("RAROC", report["rapm"]["raroc"]),
            ("Sharpe", report["rapm"]["sharpe"]),
            ("VaR-Sharpe", report["rapm"]["var_sharpe"]),
            ("ES-Sharpe", report["rapm"]["es_sharpe"]),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])
