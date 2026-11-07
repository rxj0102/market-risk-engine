"""Component and marginal Value-at-Risk decomposition.

This module decomposes total parametric (variance-covariance) VaR into
additive per-asset *component* contributions and provides portfolio-level
diversification and concentration diagnostics.

The parametric VaR of a portfolio with weights :math:`w`, daily covariance
:math:`\\Sigma` and notional :math:`N` at confidence with normal quantile
:math:`z` is

.. math::

    \\mathrm{VaR} = z \\, \\sigma_p \\, N, \\qquad
    \\sigma_p = \\sqrt{w^\\top \\Sigma w}.

Because VaR is positively homogeneous of degree one in :math:`w`, Euler's
theorem yields the exact additive decomposition

.. math::

    \\mathrm{VaR} = \\sum_i \\mathrm{CVaR}_i, \\qquad
    \\mathrm{CVaR}_i = w_i \\frac{(\\Sigma w)_i}{\\sigma_p} z N.

References
----------
Litterman, R. "Hot Spots and Hedges." Goldman Sachs Risk Management
Series, 1996.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import Portfolio
from models import CovarianceEstimator
from utils.statistics import herfindahl_index, z_score


class ComponentVaR:
    """Additive decomposition of parametric VaR into per-asset components.

    Parameters
    ----------
    cov_estimator : CovarianceEstimator, optional
        Covariance estimator used to obtain the daily covariance matrix.
        A fresh :class:`~models.CovarianceEstimator` is created if omitted.
    """

    def __init__(self, cov_estimator: CovarianceEstimator | None = None) -> None:
        self._cov = cov_estimator if cov_estimator is not None else CovarianceEstimator()

    # ------------------------------------------------------------------ utils
    def _covariance(self, portfolio: Portfolio, cov_method: str) -> np.ndarray:
        """Return the daily covariance matrix aligned with ``asset_names``."""
        cov_df = self._cov.estimate(portfolio.returns, method=cov_method)
        cov_df = cov_df.reindex(index=portfolio.asset_names, columns=portfolio.asset_names)
        return cov_df.to_numpy()

    @staticmethod
    def _portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> tuple[float, np.ndarray]:
        """Daily portfolio volatility and the marginal vector ``Sigma w``."""
        sigma_w = cov @ weights
        variance = float(weights @ sigma_w)
        sigma_p = float(np.sqrt(max(variance, 0.0)))
        return sigma_p, sigma_w

    # ----------------------------------------------------------------- public
    def compute(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        cov_method: str = "ewma",
    ) -> pd.DataFrame:
        """Decompose total parametric VaR into additive per-asset components.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        confidence : float, default 0.99
            VaR confidence level.
        cov_method : str, default 'ewma'
            Covariance estimation method passed to
            :meth:`CovarianceEstimator.estimate`.

        Returns
        -------
        pd.DataFrame
            One row per asset with columns ``Asset``, ``Weight``, ``StdDev``,
            ``Correlation_with_Portfolio``, ``ComponentVaR($)``,
            ``ComponentVaR(%)`` and ``MarginalVaR``. The sum of
            ``ComponentVaR($)`` equals the total parametric VaR to within
            floating-point precision.
        """
        w = portfolio.weights
        cov = self._covariance(portfolio, cov_method)
        z = z_score(confidence)
        notional = portfolio.notional

        sigma_p, sigma_w = self._portfolio_vol(w, cov)
        std = np.sqrt(np.diag(cov))

        if sigma_p == 0.0:
            marginal = np.zeros_like(w)
            component = np.zeros_like(w)
            corr_with_p = np.full_like(w, np.nan, dtype=float)
        else:
            marginal = sigma_w / sigma_p * z * notional
            component = w * sigma_w / sigma_p * z * notional
            # corr(asset_i, portfolio) = (Sigma w)_i / (std_i * sigma_p)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_with_p = np.where(std > 0, sigma_w / (std * sigma_p), np.nan)

        total_var = z * sigma_p * notional
        component_pct = component / total_var if total_var != 0 else np.zeros_like(component)

        return pd.DataFrame(
            {
                "Asset": list(portfolio.asset_names),
                "Weight": w,
                "StdDev": std,
                "Correlation_with_Portfolio": corr_with_p,
                "ComponentVaR($)": component,
                "ComponentVaR(%)": component_pct,
                "MarginalVaR": marginal,
            }
        )

    def diversification_ratio(self, portfolio: Portfolio, cov_method: str = "ewma") -> float:
        """Diversification benefit ``1 - sigma_p / (w . diag(sigma))``.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        cov_method : str, default 'ewma'
            Covariance estimation method.

        Returns
        -------
        float
            Value in ``(0, 1)`` for a diversified portfolio; ``0`` for a
            single-asset portfolio (no diversification possible).
        """
        if portfolio.n_assets <= 1:
            return 0.0
        w = portfolio.weights
        cov = self._covariance(portfolio, cov_method)
        sigma_p, _ = self._portfolio_vol(w, cov)
        std = np.sqrt(np.diag(cov))
        weighted_standalone = float(np.abs(w) @ std)
        if weighted_standalone == 0.0:
            return 0.0
        return 1.0 - sigma_p / weighted_standalone

    def risk_concentration(self, portfolio: Portfolio, confidence: float = 0.99) -> float:
        """Herfindahl-Hirschman index of the component-VaR shares.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        confidence : float, default 0.99
            VaR confidence level.

        Returns
        -------
        float
            HHI in ``(0, 1]``; ``1`` indicates all risk in a single asset,
            ``1/n`` an equal split.
        """
        df = self.compute(portfolio, confidence=confidence)
        return herfindahl_index(df["ComponentVaR($)"].to_numpy())
