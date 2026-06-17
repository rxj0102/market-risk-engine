"""Correlation-driven risk analytics.

Tools for isolating the portion of portfolio risk that arises from
cross-asset correlation (as opposed to standalone volatilities), tracking it
through time, and stress-testing the correlation structure.

References
----------
Engle, R. "Dynamic Conditional Correlation." Journal of Business & Economic
Statistics, 2002.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import Portfolio
from models import CovarianceEstimator
from utils.statistics import z_score


class CorrelationRiskAnalyzer:
    """Decompose and stress the correlation contribution to portfolio risk.

    Parameters
    ----------
    cov_estimator : CovarianceEstimator, optional
        Covariance estimator. A fresh instance is created if omitted.
    """

    def __init__(self, cov_estimator: CovarianceEstimator | None = None) -> None:
        self._cov = cov_estimator if cov_estimator is not None else CovarianceEstimator()

    # ------------------------------------------------------------------ utils
    def _covariance(self, portfolio: Portfolio, cov_method: str) -> np.ndarray:
        cov_df = self._cov.estimate(portfolio.returns, method=cov_method)
        cov_df = cov_df.reindex(index=portfolio.asset_names, columns=portfolio.asset_names)
        return cov_df.to_numpy()

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(std, correlation_matrix)`` from a covariance matrix."""
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(std > 0, 1.0 / std, 0.0)
        corr = cov * np.outer(inv, inv)
        return std, corr

    # ----------------------------------------------------------------- public
    def correlation_contribution(
        self, portfolio: Portfolio, cov_method: str = "sample"
    ) -> float:
        """Fraction of portfolio variance attributable to correlations.

        ``1 - sigma_uncorr^2 / sigma_p^2`` where
        ``sigma_uncorr^2 = w^T D^2 w`` (``D`` = diagonal of standard
        deviations) and ``sigma_p^2 = w^T Sigma w``.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        cov_method : str, default 'sample'
            Covariance estimation method.

        Returns
        -------
        float
            Value in ``(0, 1)`` for a typically positively-correlated
            portfolio; ``0`` when assets are uncorrelated.
        """
        w = portfolio.weights
        cov = self._covariance(portfolio, cov_method)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        sigma_p2 = float(w @ cov @ w)
        sigma_uncorr2 = float(np.sum((w * std) ** 2))
        if sigma_p2 == 0.0:
            return 0.0
        return 1.0 - sigma_uncorr2 / sigma_p2

    def rolling_correlation_risk(
        self, portfolio: Portfolio, window: int = 60
    ) -> pd.Series:
        """Rolling correlation contribution over a trailing window.

        For each window the sample covariance is formed and the correlation
        contribution (see :meth:`correlation_contribution`) is recorded.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        window : int, default 60
            Trailing window length in observations.

        Returns
        -------
        pd.Series
            Correlation contribution indexed by the window's end date.
        """
        w = portfolio.weights
        ret = portfolio.returns[portfolio.asset_names].to_numpy()
        dates = portfolio.returns.index
        n = ret.shape[0]
        out_vals: list[float] = []
        out_idx: list = []
        for end in range(window, n + 1):
            block = ret[end - window : end]
            cov = np.cov(block, rowvar=False)
            std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            sigma_p2 = float(w @ cov @ w)
            sigma_uncorr2 = float(np.sum((w * std) ** 2))
            val = 0.0 if sigma_p2 == 0.0 else 1.0 - sigma_uncorr2 / sigma_p2
            out_vals.append(val)
            out_idx.append(dates[end - 1])
        return pd.Series(out_vals, index=out_idx, name="correlation_risk")

    def correlation_stress_impact(
        self,
        portfolio: Portfolio,
        confidence: float,
        stressed_corr_matrix: np.ndarray,
        cov_method: str = "sample",
    ) -> dict:
        """VaR impact of replacing correlations with a stressed matrix.

        Standalone volatilities are held fixed; only the correlation structure
        is replaced. The stressed covariance is ``D C* D`` with ``D`` the
        diagonal of standard deviations and ``C*`` the stressed correlation
        matrix.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        confidence : float
            VaR confidence level.
        stressed_corr_matrix : np.ndarray
            ``(n, n)`` stressed correlation matrix.
        cov_method : str, default 'sample'
            Covariance estimation method for the base case.

        Returns
        -------
        dict
            ``{'base_var', 'stressed_var', 'var_change', 'var_change_pct'}``
            with VaR values in positive dollars.
        """
        w = portfolio.weights
        cov = self._covariance(portfolio, cov_method)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        z = z_score(confidence)
        notional = portfolio.notional

        c_star = np.asarray(stressed_corr_matrix, dtype=float)
        stressed_cov = c_star * np.outer(std, std)

        base_var = z * np.sqrt(max(float(w @ cov @ w), 0.0)) * notional
        stressed_var = z * np.sqrt(max(float(w @ stressed_cov @ w), 0.0)) * notional
        change = stressed_var - base_var
        return {
            "base_var": base_var,
            "stressed_var": stressed_var,
            "var_change": change,
            "var_change_pct": change / base_var if base_var != 0 else float("nan"),
        }

    def average_correlation(
        self, portfolio: Portfolio, cov_method: str = "sample"
    ) -> float:
        """Weighted average of off-diagonal pairwise correlations.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        cov_method : str, default 'sample'
            Covariance estimation method.

        Returns
        -------
        float
            ``(w^T C w - w^T w) / ((sum w)^2 - w^T w)``, the weight-weighted
            mean off-diagonal correlation. Returns ``nan`` for a single asset.
        """
        w = portfolio.weights
        if portfolio.n_assets <= 1:
            return float("nan")
        cov = self._covariance(portfolio, cov_method)
        _, corr = self._cov_to_corr(cov)
        numerator = float(w @ corr @ w) - float(w @ w)
        denominator = float(w.sum() ** 2) - float(w @ w)
        if denominator == 0.0:
            return float("nan")
        return numerator / denominator

    def concentration_vs_diversification_chart(
        self, portfolio: Portfolio, cov_method: str = "sample"
    ) -> pd.DataFrame:
        """Plot-ready data contrasting concentration and diversification.

        Provides, per asset, the weight, standalone volatility and the
        standalone (undiversified) versus marginal-risk contributions, useful
        for a concentration-versus-diversification visualisation.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        cov_method : str, default 'sample'
            Covariance estimation method.

        Returns
        -------
        pd.DataFrame
            Columns ``Asset``, ``Weight``, ``StdDev``,
            ``StandaloneRiskShare`` (share of summed standalone vol) and
            ``DiversifiedRiskShare`` (share of portfolio variance via the
            Euler decomposition).
        """
        w = portfolio.weights
        cov = self._covariance(portfolio, cov_method)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        standalone = np.abs(w) * std
        standalone_share = (
            standalone / standalone.sum() if standalone.sum() != 0 else np.zeros_like(standalone)
        )

        sigma_w = cov @ w
        sigma_p2 = float(w @ sigma_w)
        risk_contrib = w * sigma_w  # sums to portfolio variance
        diversified_share = (
            risk_contrib / sigma_p2 if sigma_p2 != 0 else np.zeros_like(risk_contrib)
        )

        return pd.DataFrame(
            {
                "Asset": list(portfolio.asset_names),
                "Weight": w,
                "StdDev": std,
                "StandaloneRiskShare": standalone_share,
                "DiversifiedRiskShare": diversified_share,
            }
        )
