"""Historical-simulation Value-at-Risk estimators.

This module implements non-parametric (empirical) VaR together with several
well-known refinements: the Boudoukh-Richardson-Whitelaw (BRW, 1998)
age-weighted scheme and the Hull-White (1998) volatility-scaling adjustment.

References
----------
Boudoukh, J., Richardson, M., Whitelaw, R. (1998).
    "The Best of Both Worlds." Risk, 11, 64-67.
Hull, J., White, A. (1998).
    "Incorporating Volatility Updating into the Historical Simulation Method
    for Value-at-Risk." Journal of Risk, 1(1), 5-19.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from core import AGE_WEIGHT_LAMBDA, Portfolio, VaRResult
from models import CovarianceEstimator, GARCHModel
from utils.statistics import weighted_quantile


class HistoricalVaR:
    """Empirical (historical-simulation) Value-at-Risk.

    The estimator reads the realised portfolio P&L distribution and reports the
    loss quantile.  No distributional assumption is imposed on returns.

    Notes
    -----
    VaR is reported as a positive dollar loss, ``VaR = -Q_{1-c}(PnL)`` where
    ``Q`` is the empirical quantile of the dollar P&L series.
    """

    def compute(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizon_days: int = 1,
        lookback_days: int = 500,
    ) -> VaRResult:
        """Empirical 1-day (sqrt-time scaled) VaR.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio whose P&L distribution is sampled.
        confidence : float, default 0.99
            VaR confidence level.
        horizon_days : int, default 1
            Risk horizon; the 1-day figure is scaled by ``sqrt(horizon_days)``.
        lookback_days : int, default 500
            Number of most-recent observations used.

        Returns
        -------
        VaRResult
            Positive dollar VaR with ``method='Historical'``.
        """
        start = time.perf_counter()
        pnl = portfolio.pnl().to_numpy()[-lookback_days:]
        var_1d = -np.quantile(pnl, 1.0 - confidence)
        var = float(var_1d * np.sqrt(horizon_days))
        return VaRResult(
            var=var,
            var_pct=var / portfolio.notional,
            confidence=confidence,
            horizon_days=horizon_days,
            method="Historical",
            n_scenarios=len(pnl),
            elapsed_seconds=time.perf_counter() - start,
        )

    def compute_multi_horizon(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizons: list[int] | None = None,
        lookback_days: int = 500,
    ) -> pd.DataFrame:
        """VaR across horizons: sqrt-time scaled vs. directly aggregated.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizons : list of int, optional
            Horizons in days. Defaults to ``[1, 5, 10, 20]``.
        lookback_days : int, default 500
            Lookback for the 1-day distribution (sqrt-time column).

        Returns
        -------
        pd.DataFrame
            Indexed by horizon with columns ``var_sqrt_time`` and
            ``var_h_day`` (non-overlapping aggregated h-day P&L).
        """
        if horizons is None:
            horizons = [1, 5, 10, 20]
        pnl_full = portfolio.pnl().to_numpy()
        pnl = pnl_full[-lookback_days:]
        var_1d = -np.quantile(pnl, 1.0 - confidence)
        rows = []
        for h in horizons:
            sqrt_var = float(var_1d * np.sqrt(h))
            # Non-overlapping aggregation of h-day P&L (use full sample for depth).
            n_blocks = len(pnl_full) // h
            agg = pnl_full[: n_blocks * h].reshape(n_blocks, h).sum(axis=1)
            h_var = float(-np.quantile(agg, 1.0 - confidence))
            rows.append(
                {
                    "horizon": h,
                    "var_sqrt_time": sqrt_var,
                    "var_h_day": h_var,
                    "var_sqrt_time_pct": sqrt_var / portfolio.notional,
                    "var_h_day_pct": h_var / portfolio.notional,
                }
            )
        return pd.DataFrame(rows).set_index("horizon")

    def compute_age_weighted(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        lambda_: float = AGE_WEIGHT_LAMBDA,
        lookback: int = 500,
    ) -> VaRResult:
        """Age-weighted (BRW 1998) historical VaR.

        Observation ``i`` (oldest = 1, most recent = T) receives weight
        proportional to ``lambda^{T-i}``, so recent observations dominate.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        lambda_ : float, default 0.98
            Decay factor in ``(0, 1)``.
        lookback : int, default 500
            Number of observations.

        Returns
        -------
        VaRResult
            ``method='Age-Weighted'``.

        References
        ----------
        Boudoukh, Richardson, Whitelaw (1998).
        """
        start = time.perf_counter()
        pnl = portfolio.pnl().to_numpy()[-lookback:]
        losses = -pnl
        t = len(pnl)
        # i runs 1..T with most recent last; weight lambda^{T-i}.
        exponents = np.arange(t - 1, -1, -1, dtype=float)
        weights = lambda_**exponents
        weights /= weights.sum()
        var = float(weighted_quantile(losses, weights, confidence))
        return VaRResult(
            var=var,
            var_pct=var / portfolio.notional,
            confidence=confidence,
            horizon_days=1,
            method="Age-Weighted",
            n_scenarios=t,
            elapsed_seconds=time.perf_counter() - start,
        )

    def compute_vol_scaled(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        lookback: int = 500,
        garch_vol: bool = True,
    ) -> VaRResult:
        """Volatility-scaled (Hull-White 1998) historical VaR.

        Each historical portfolio return is rescaled by
        ``sigma_current / sigma_t`` so that the simulated distribution reflects
        today's volatility regime.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        lookback : int, default 500
        garch_vol : bool, default True
            If True use GARCH(1,1) conditional vols; otherwise an EWMA vol.

        Returns
        -------
        VaRResult
            ``method='Vol-Scaled'``.

        References
        ----------
        Hull, White (1998).
        """
        start = time.perf_counter()
        ret = portfolio.return_series().to_numpy()
        if garch_vol:
            params = GARCHModel().fit(ret)
            sigma = GARCHModel().conditional_vols(ret, params)
        else:
            sigma = _ewma_vol(ret, lambda_=0.94)
        sigma = sigma[-lookback:]
        r = ret[-lookback:]
        sigma_current = sigma[-1]
        scaled = r * (sigma_current / sigma)
        pnl = scaled * portfolio.notional
        var = float(-np.quantile(pnl, 1.0 - confidence))
        return VaRResult(
            var=var,
            var_pct=var / portfolio.notional,
            confidence=confidence,
            horizon_days=1,
            method="Vol-Scaled",
            n_scenarios=len(pnl),
            elapsed_seconds=time.perf_counter() - start,
        )

    def pnl_distribution(self, portfolio: Portfolio, lookback: int = 500) -> pd.Series:
        """Sorted (ascending) historical P&L with dates retained.

        Parameters
        ----------
        portfolio : Portfolio
        lookback : int, default 500

        Returns
        -------
        pd.Series
            P&L sorted from worst loss to best gain, indexed by date.
        """
        pnl = portfolio.pnl().iloc[-lookback:]
        return pnl.sort_values()

    def rolling_var(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        window: int = 250,
        step: int = 1,
    ) -> pd.Series:
        """Rolling-window historical VaR time series.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        window : int, default 250
            Rolling estimation window.
        step : int, default 1
            Stride between successive estimates.

        Returns
        -------
        pd.Series
            VaR (positive dollars) indexed by the window end date.
        """
        pnl = portfolio.pnl()
        vals = pnl.to_numpy()
        idx = pnl.index
        out_dates = []
        out_var = []
        for end in range(window, len(vals) + 1, step):
            w = vals[end - window : end]
            out_var.append(-np.quantile(w, 1.0 - confidence))
            out_dates.append(idx[end - 1])
        return pd.Series(out_var, index=pd.Index(out_dates, name=idx.name), name="rolling_var")

    def var_term_structure(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        lookback: int = 500,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """VaR term structure across horizons (alias of multi-horizon view).

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        lookback : int, default 500
        horizons : list of int, optional
            Defaults to ``[1, 5, 10, 20, 60, 120, 252]``.

        Returns
        -------
        pd.DataFrame
            Indexed by horizon, columns for sqrt-time and aggregated VaR.
        """
        if horizons is None:
            horizons = [1, 5, 10, 20, 60, 120, 252]
        return self.compute_multi_horizon(portfolio, confidence, horizons, lookback)


def _ewma_vol(returns: np.ndarray, lambda_: float = 0.94) -> np.ndarray:
    """Exponentially-weighted recursive volatility series (helper)."""
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()
    var = np.empty_like(r)
    var[0] = np.var(r)
    for t in range(1, len(r)):
        var[t] = lambda_ * var[t - 1] + (1.0 - lambda_) * r[t - 1] ** 2
    return np.sqrt(var)
