"""Economic capital estimation.

Economic capital (EC) is the unexpected-loss buffer a bank holds against its
own internal risk appetite, conventionally a one-year Value-at-Risk at a very
high confidence level (here 99.9%, consistent with an investment-grade target
rating). EC is contrasted with *regulatory* capital (Basel III/IV), which is
governed by external rules and lives in :mod:`capital.regulatory_capital`.

The estimators here build on the locked :class:`var.ParametricVaR` engine and
apply square-root-of-time scaling to move from the engine's native one-day
horizon to the one-year (``TRADING_DAYS``) economic-capital horizon.

References
----------
Basel Committee on Banking Supervision (BCBS).
    "An Explanatory Note on the Basel II IRB Risk Weight Functions." 2005.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import TRADING_DAYS, Portfolio
from var import ParametricVaR


class EconomicCapitalEngine:
    """Internal economic-capital (EC) estimator.

    EC is defined as a one-year parametric Value-at-Risk at a high confidence
    level (default 99.9%). All one-day quantities produced by the underlying
    :class:`var.ParametricVaR` engine are scaled to the annual horizon by
    ``sqrt(TRADING_DAYS)``.

    Parameters
    ----------
    cov_method : str, default 'sample'
        Covariance estimator passed through to :class:`var.ParametricVaR`.
    """

    def __init__(self, cov_method: str = "sample") -> None:
        self._cov_method = cov_method
        self._pv = ParametricVaR()

    # ------------------------------------------------------------------ helpers
    def _annual_scale(self, horizon_days: int) -> float:
        """Square-root-of-time scaling factor from 1 day to ``horizon_days``."""
        return float(np.sqrt(horizon_days))

    # ------------------------------------------------------------------ EC
    def economic_capital(
        self,
        portfolio: Portfolio,
        confidence: float = 0.999,
        horizon_days: int = TRADING_DAYS,
    ) -> float:
        """One-year economic capital ``EC = VaR(confidence, horizon)``.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.999
            Economic-capital confidence level.
        horizon_days : int, default ``TRADING_DAYS`` (252)
            Capital horizon in trading days.

        Returns
        -------
        float
            Economic capital as a positive dollar loss. Computed as the one-day
            parametric VaR scaled by ``sqrt(horizon_days)``.
        """
        daily = self._pv.compute(
            portfolio, confidence=confidence, horizon_days=1, cov_method=self._cov_method
        )
        return float(daily.var * self._annual_scale(horizon_days))

    # ------------------------------------------------------------------ by asset
    def ec_by_asset(
        self,
        portfolio: Portfolio,
        confidence: float = 0.999,
        horizon_days: int = TRADING_DAYS,
    ) -> pd.DataFrame:
        """Component economic capital per asset (Euler allocation).

        The analytical one-day component VaR (which sums to total VaR) is scaled
        to the annual EC horizon, so the component ECs sum to the portfolio EC.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.999
        horizon_days : int, default ``TRADING_DAYS``

        Returns
        -------
        pd.DataFrame
            Columns ``Asset``, ``EC($)`` and ``EC(%)`` (share of total EC).
        """
        comp_daily = self._pv.component_var_analytical(
            portfolio, confidence=confidence, cov_method=self._cov_method
        )
        comp_ec = comp_daily * self._annual_scale(horizon_days)
        total = comp_ec.sum()
        return pd.DataFrame(
            {
                "Asset": portfolio.asset_names,
                "EC($)": comp_ec,
                "EC(%)": comp_ec / total if total else np.full(len(comp_ec), np.nan),
            }
        )

    # ------------------------------------------------------------------ RAROC
    def raroc(
        self,
        portfolio: Portfolio,
        expected_return: float,
        confidence: float = 0.999,
    ) -> float:
        """Risk-adjusted return on capital ``RAROC = E[P&L] / EC``.

        Parameters
        ----------
        portfolio : Portfolio
        expected_return : float
            Expected annual P&L in dollars.
        confidence : float, default 0.999

        Returns
        -------
        float
            RAROC. ``nan`` if economic capital is zero.
        """
        ec = self.economic_capital(portfolio, confidence=confidence)
        return float(expected_return / ec) if ec else float("nan")

    # ------------------------------------------------------------ risk contrib
    def risk_contribution_to_ec(
        self, portfolio: Portfolio, confidence: float = 0.999
    ) -> pd.DataFrame:
        """Percentage contribution of each asset to total economic capital.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.999

        Returns
        -------
        pd.DataFrame
            Columns ``Asset`` and ``RiskContribution(%)`` summing to 1.
        """
        ec = self.ec_by_asset(portfolio, confidence=confidence)
        return pd.DataFrame(
            {"Asset": ec["Asset"], "RiskContribution(%)": ec["EC(%)"]}
        )

    # ------------------------------------------------------ diversification
    def diversification_benefit(
        self, portfolio: Portfolio, confidence: float = 0.999
    ) -> float:
        """Diversification benefit of the portfolio.

        Defined as ``(undiversified - diversified) / undiversified`` where the
        *diversified* EC is the portfolio EC and the *undiversified* EC is the
        sum of each asset's standalone EC. Each standalone EC is the asset's own
        VaR at its full position weight ``w_i * notional`` (i.e. as if held
        alone), so it ignores cross-asset correlation.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.999

        Returns
        -------
        float
            A number in ``(0, 1)``; higher means more diversified.
        """
        cov = self._pv.covariance_matrix(portfolio, method=self._cov_method).to_numpy()
        vols = np.sqrt(np.diag(cov))  # daily per-asset volatilities
        from utils.statistics import z_score

        z = z_score(confidence)
        scale = self._annual_scale(TRADING_DAYS)
        # Standalone EC for each asset held at its own dollar exposure.
        exposures = np.abs(portfolio.weights) * portfolio.notional
        standalone = z * vols * exposures * scale
        undiversified = float(standalone.sum())
        diversified = self.economic_capital(portfolio, confidence=confidence)
        if undiversified <= 0:
            return float("nan")
        return float((undiversified - diversified) / undiversified)
