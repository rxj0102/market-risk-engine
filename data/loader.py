"""Market data loading and Portfolio construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import Portfolio
from data.returns import ReturnCalculator


class MarketDataLoader:
    """Load return/price panels from CSV and assemble :class:`Portfolio` objects."""

    @staticmethod
    def load_returns_csv(path: str | Path, date_col: str = "date") -> pd.DataFrame:
        """Load a returns CSV with a date index.

        Parameters
        ----------
        path : str or Path
            Path to a CSV of simple returns (dates x assets).
        date_col : str
            Name of the date column.

        Returns
        -------
        pd.DataFrame
            Cleaned return panel.
        """
        df = pd.read_csv(path)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            df.index = pd.to_datetime(df.index)
        return ReturnCalculator.clean(df)

    @staticmethod
    def load_prices_csv(
        path: str | Path, date_col: str = "date", log: bool = False
    ) -> pd.DataFrame:
        """Load a price CSV and convert to returns."""
        df = pd.read_csv(path)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            df.index = pd.to_datetime(df.index)
        rets = ReturnCalculator.log_returns(df) if log else ReturnCalculator.simple_returns(df)
        return ReturnCalculator.clean(rets)

    @staticmethod
    def build_portfolio(
        returns: pd.DataFrame,
        weights: np.ndarray | list[float] | None = None,
        notional: float = 10_000_000.0,
        currency: str = "USD",
    ) -> Portfolio:
        """Assemble a :class:`Portfolio` from a return panel and weights.

        If ``weights`` is ``None`` an equal-weight allocation is used. Weights
        are normalised to sum to one.
        """
        names = list(returns.columns)
        if weights is None:
            weights = np.full(len(names), 1.0 / len(names))
        weights = np.asarray(weights, dtype=float)
        total = weights.sum()
        if not np.isclose(total, 0.0):
            weights = weights / total
        return Portfolio(weights=weights, asset_names=names, returns=returns, notional=notional, currency=currency)

    @staticmethod
    def equal_weights(n: int) -> np.ndarray:
        """Equal-weight vector of length ``n``."""
        return np.full(n, 1.0 / n)

    @staticmethod
    def minimum_variance_weights(returns: pd.DataFrame) -> np.ndarray:
        """Long/short global minimum-variance weights from the sample covariance.

        Notes
        -----
        ``w = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)``. Uses a pseudo-inverse for
        numerical robustness on near-singular covariance matrices.
        """
        cov = returns.cov().to_numpy()
        ones = np.ones(cov.shape[0])
        inv = np.linalg.pinv(cov)
        w = inv @ ones
        return w / (ones @ inv @ ones)
