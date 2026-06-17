"""Time-varying correlation diagnostics for multi-asset portfolios.

The :class:`CorrelationDynamicsAnalyzer` exposes rolling, DCC-GARCH and
regime-conditional views of the portfolio's correlation structure -- the key
driver of diversification breakdown ("correlations go to one") during stress.

References
----------
Engle, R. (2002). "Dynamic Conditional Correlation." *JBES* 20(3), 339-350.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from core import Portfolio
from models import CovarianceEstimator


class CorrelationDynamicsAnalyzer:
    """Analyse how the portfolio's correlation matrix evolves over time."""

    def __init__(self) -> None:
        self._cov = CovarianceEstimator()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _asset_index(portfolio: Portfolio, asset: str | int) -> int:
        """Resolve an asset name or positional index to an integer index."""
        if isinstance(asset, str):
            return portfolio.asset_names.index(asset)
        return int(asset)

    # ------------------------------------------------ rolling correlation
    def rolling_correlation(
        self,
        portfolio: Portfolio,
        asset_i: str | int,
        asset_j: str | int,
        window: int = 60,
    ) -> pd.Series:
        """Rolling pairwise Pearson correlation between two assets.

        Parameters
        ----------
        portfolio : Portfolio
        asset_i, asset_j : str or int
            Asset names or positional indices.
        window : int, default 60

        Returns
        -------
        pd.Series
            Rolling correlation indexed by date (leading ``NaN`` until full).
        """
        i = self._asset_index(portfolio, asset_i)
        j = self._asset_index(portfolio, asset_j)
        ri = portfolio.returns.iloc[:, i]
        rj = portfolio.returns.iloc[:, j]
        corr = ri.rolling(window).corr(rj)
        corr.name = f"corr_{portfolio.asset_names[i]}_{portfolio.asset_names[j]}"
        return corr

    # --------------------------------------------------- DCC correlation
    def dcc_correlation_series(
        self,
        portfolio: Portfolio,
        asset_i: str | int,
        asset_j: str | int,
    ) -> pd.Series:
        """DCC-GARCH conditional correlation path between two assets.

        Parameters
        ----------
        portfolio : Portfolio
        asset_i, asset_j : str or int

        Returns
        -------
        pd.Series
            The off-diagonal DCC correlation ``rho_{ij,t}`` indexed by date.
        """
        i = self._asset_index(portfolio, asset_i)
        j = self._asset_index(portfolio, asset_j)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, corr_series = self._cov.dcc_covariance(portfolio.returns)
        rho = np.array([R[i, j] for R in corr_series])
        name = f"dcc_{portfolio.asset_names[i]}_{portfolio.asset_names[j]}"
        return pd.Series(rho, index=portfolio.returns.index, name=name)

    # --------------------------------------- average rolling correlation
    def average_rolling_correlation(
        self, portfolio: Portfolio, window: int = 60
    ) -> pd.Series:
        """Mean pairwise correlation across all asset pairs over time.

        A rising average correlation is the classic signature of a
        diversification breakdown during market stress.

        Parameters
        ----------
        portfolio : Portfolio
        window : int, default 60

        Returns
        -------
        pd.Series
            Average off-diagonal correlation indexed by date (leading ``NaN``
            until the window is full).
        """
        returns = portfolio.returns
        n_assets = returns.shape[1]
        x = returns.to_numpy()
        idx = returns.index
        n = x.shape[0]
        n_pairs = n_assets * (n_assets - 1) / 2

        avg = np.full(n, np.nan)
        # iu: upper-triangular off-diagonal indices.
        iu = np.triu_indices(n_assets, k=1)
        for end in range(window, n + 1):
            block = x[end - window : end]
            corr = np.corrcoef(block, rowvar=False)
            avg[end - 1] = corr[iu].sum() / n_pairs
        return pd.Series(avg, index=idx, name="avg_correlation")

    # ----------------------------------------- regime correlation compare
    def correlation_regime_comparison(
        self,
        portfolio: Portfolio,
        calm_window: tuple | slice | None = None,
        stress_window: tuple | slice | None = None,
        vol_window: int = 60,
    ) -> dict[str, pd.DataFrame | float]:
        """Compare correlation matrices in calm vs. stressed periods.

        Parameters
        ----------
        portfolio : Portfolio
        calm_window, stress_window : tuple, slice or None
            Date selections passed to ``DataFrame.loc`` (e.g. a
            ``(start, end)`` tuple or a ``slice``). When ``None`` the calm and
            stressed periods are detected automatically as the lowest- and
            highest-volatility ``vol_window``-day blocks of the portfolio
            return series.
        vol_window : int, default 60
            Window used for automatic regime detection.

        Returns
        -------
        dict
            ``calm`` and ``stress`` correlation matrices (``pd.DataFrame``),
            ``calm_avg_corr`` / ``stress_avg_corr`` mean off-diagonal
            correlations, and ``correlation_increase`` (their difference).
        """
        returns = portfolio.returns

        if calm_window is not None and stress_window is not None:
            calm_ret = self._select(returns, calm_window)
            stress_ret = self._select(returns, stress_window)
        else:
            port_ret = portfolio.return_series()
            rolling_vol = port_ret.rolling(vol_window).std(ddof=1)
            calm_end = int(np.nanargmin(rolling_vol.to_numpy())) + 1
            stress_end = int(np.nanargmax(rolling_vol.to_numpy())) + 1
            calm_ret = returns.iloc[calm_end - vol_window : calm_end]
            stress_ret = returns.iloc[stress_end - vol_window : stress_end]

        calm_corr = calm_ret.corr()
        stress_corr = stress_ret.corr()

        iu = np.triu_indices(returns.shape[1], k=1)
        calm_avg = float(calm_corr.to_numpy()[iu].mean())
        stress_avg = float(stress_corr.to_numpy()[iu].mean())

        return {
            "calm": calm_corr,
            "stress": stress_corr,
            "calm_avg_corr": calm_avg,
            "stress_avg_corr": stress_avg,
            "correlation_increase": stress_avg - calm_avg,
        }

    @staticmethod
    def _select(returns: pd.DataFrame, window: tuple | slice) -> pd.DataFrame:
        """Apply a date selection (tuple of bounds or slice) via ``.loc``."""
        if isinstance(window, slice):
            return returns.loc[window]
        start, end = window
        return returns.loc[start:end]
