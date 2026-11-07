"""Return computation and cleaning utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ReturnCalculator:
    """Compute and clean asset return series from prices or returns.

    Methods are static; the class groups related return utilities used by the
    data loader and the diagnostics modules.
    """

    @staticmethod
    def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Simple (arithmetic) returns ``r_t = P_t / P_{t-1} - 1``.

        Parameters
        ----------
        prices : pd.DataFrame
            Price levels indexed by date.

        Returns
        -------
        pd.DataFrame
            Simple returns with the first (NaN) row dropped.
        """
        return prices.pct_change().dropna(how="all")

    @staticmethod
    def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Continuously-compounded returns ``r_t = ln(P_t / P_{t-1})``."""
        return np.log(prices / prices.shift(1)).dropna(how="all")

    @staticmethod
    def prices_from_returns(returns: pd.DataFrame, initial: float = 100.0) -> pd.DataFrame:
        """Reconstruct a price index from simple returns."""
        return (1.0 + returns).cumprod() * initial

    @staticmethod
    def clean(returns: pd.DataFrame, max_abs: float = 1.0) -> pd.DataFrame:
        """Clean a return panel.

        Drops all-NaN rows, forward/back fills sporadic gaps, and clips
        absurd values (``|r| > max_abs``) that indicate data errors.

        Parameters
        ----------
        returns : pd.DataFrame
            Raw return panel.
        max_abs : float
            Maximum absolute daily return retained before clipping.

        Returns
        -------
        pd.DataFrame
            Cleaned return panel.
        """
        cleaned = returns.dropna(how="all").ffill().bfill()
        cleaned = cleaned.clip(lower=-max_abs, upper=max_abs)
        return cleaned

    @staticmethod
    def winsorize(returns: pd.DataFrame, lower: float = 0.001, upper: float = 0.999) -> pd.DataFrame:
        """Winsorize returns at the given quantiles (per column)."""
        lo = returns.quantile(lower)
        hi = returns.quantile(upper)
        return returns.clip(lower=lo, upper=hi, axis=1)

    @staticmethod
    def to_horizon(returns: pd.Series, horizon: int) -> pd.Series:
        """Aggregate single-period simple returns to non-overlapping h-period returns."""
        grouped = (1.0 + returns).groupby(np.arange(len(returns)) // horizon).prod() - 1.0
        return grouped
