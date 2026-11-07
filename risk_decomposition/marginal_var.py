"""Marginal and incremental Value-at-Risk, hedging diagnostics.

*Marginal VaR* is the partial derivative of portfolio VaR with respect to an
asset weight; *incremental VaR* is the exact change in VaR from fully removing
(and renormalising) a position. This module also provides simple best-hedge
identification and a one-dimensional optimal-hedge-ratio search.

References
----------
Jorion, P. "Value at Risk." 3rd ed., McGraw-Hill, 2007, ch. 7.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from core import Portfolio
from models import CovarianceEstimator
from utils.statistics import z_score


class MarginalVaR:
    """Marginal, incremental and hedging VaR analytics.

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
    def _var_from_weights(
        weights: np.ndarray, cov: np.ndarray, z: float, notional: float
    ) -> float:
        """Parametric VaR (positive dollars) for arbitrary ``weights``."""
        variance = float(weights @ cov @ weights)
        return z * np.sqrt(max(variance, 0.0)) * notional

    # ----------------------------------------------------------------- public
    def marginal_var(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        cov_method: str = "ewma",
    ) -> np.ndarray:
        """Marginal VaR vector ``(Sigma w)_i / sigma_p * z * N``.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        confidence : float, default 0.99
            VaR confidence level.
        cov_method : str, default 'ewma'
            Covariance estimation method.

        Returns
        -------
        np.ndarray
            Rate of change of portfolio VaR with respect to each asset weight.
        """
        w = portfolio.weights
        cov = self._covariance(portfolio, cov_method)
        z = z_score(confidence)
        sigma_w = cov @ w
        sigma_p = float(np.sqrt(max(w @ sigma_w, 0.0)))
        if sigma_p == 0.0:
            return np.zeros_like(w)
        return sigma_w / sigma_p * z * portfolio.notional

    def incremental_var(
        self,
        portfolio: Portfolio,
        confidence: float,
        asset_idx: int,
        cov_method: str = "ewma",
    ) -> float:
        """Exact incremental VaR from removing asset ``asset_idx``.

        Computed by full repricing: VaR of the complete portfolio minus VaR of
        the portfolio with asset ``asset_idx`` removed and the remaining
        weights renormalised to sum to their original total.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        confidence : float
            VaR confidence level.
        asset_idx : int
            Index of the asset to remove.
        cov_method : str, default 'ewma'
            Covariance estimation method.

        Returns
        -------
        float
            ``VaR(full) - VaR(without asset)``. Positive means the asset adds
            risk; negative means it is a net hedge.
        """
        w = portfolio.weights
        cov = self._covariance(portfolio, cov_method)
        z = z_score(confidence)
        notional = portfolio.notional

        full_var = self._var_from_weights(w, cov, z, notional)

        w_reduced = w.copy()
        removed = w_reduced[asset_idx]
        w_reduced[asset_idx] = 0.0
        remaining = w.sum() - removed
        if remaining != 0.0:
            w_reduced = w_reduced * (w.sum() / remaining)
        reduced_var = self._var_from_weights(w_reduced, cov, z, notional)
        return full_var - reduced_var

    def incremental_var_all(
        self,
        portfolio: Portfolio,
        confidence: float,
        cov_method: str = "ewma",
    ) -> np.ndarray:
        """Incremental VaR for every asset (see :meth:`incremental_var`).

        Returns
        -------
        np.ndarray
            One incremental-VaR value per asset.
        """
        return np.array(
            [
                self.incremental_var(portfolio, confidence, i, cov_method=cov_method)
                for i in range(portfolio.n_assets)
            ]
        )

    def best_hedge(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        cov_method: str = "ewma",
    ) -> dict:
        """Identify the asset with the most negative marginal VaR.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        confidence : float, default 0.99
            VaR confidence level.
        cov_method : str, default 'ewma'
            Covariance estimation method.

        Returns
        -------
        dict
            ``{'asset', 'asset_idx', 'marginal_var'}`` describing the best
            marginal hedge (smallest marginal VaR).
        """
        mvar = self.marginal_var(portfolio, confidence, cov_method=cov_method)
        idx = int(np.argmin(mvar))
        return {
            "asset": portfolio.asset_names[idx],
            "asset_idx": idx,
            "marginal_var": float(mvar[idx]),
        }

    def optimal_hedge_ratio(
        self,
        portfolio: Portfolio,
        confidence: float,
        hedge_asset: int | str,
        cov_method: str = "ewma",
    ) -> float:
        """Weight in ``hedge_asset`` that minimises portfolio VaR.

        A scalar position ``h`` is added to the existing weight of the hedge
        asset and the resulting VaR is minimised over ``h`` via a bounded
        one-dimensional search.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        confidence : float
            VaR confidence level.
        hedge_asset : int or str
            Index or name of the hedging asset.
        cov_method : str, default 'ewma'
            Covariance estimation method.

        Returns
        -------
        float
            The hedge weight ``h`` (incremental, relative to the current
            weight) that minimises portfolio VaR.
        """
        if isinstance(hedge_asset, str):
            hedge_idx = portfolio.asset_names.index(hedge_asset)
        else:
            hedge_idx = int(hedge_asset)

        w = portfolio.weights
        cov = self._covariance(portfolio, cov_method)
        z = z_score(confidence)
        notional = portfolio.notional

        def objective(h: float) -> float:
            w_h = w.copy()
            w_h[hedge_idx] = w[hedge_idx] + h
            return self._var_from_weights(w_h, cov, z, notional)

        result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
        return float(result.x)
