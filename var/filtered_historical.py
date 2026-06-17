"""Filtered Historical Simulation (FHS) Value-at-Risk.

The Barone-Adesi / Hull-White (1998) filtered historical simulation combines a
GARCH volatility filter with bootstrap resampling of standardised residuals.
This captures volatility clustering (parametric) while leaving the residual
shape non-parametric (fat tails, skew).

References
----------
Barone-Adesi, G., Giannopoulos, K., Vosper, L. (1999).
    "VaR without Correlations for Portfolios of Derivative Securities."
    Journal of Futures Markets, 19(5), 583-602.
Hull, J., White, A. (1998). Journal of Risk, 1(1), 5-19.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from core import DEFAULT_SEED, Portfolio, VaRResult
from models import GARCHModel


class FilteredHistoricalSimulation:
    """Filtered Historical Simulation (Hull-White 1998).

    For each asset a GARCH(1,1) model is fitted, standardised residuals
    ``z_t = r_t / sigma_t`` are extracted, and the one-step-ahead vol forecast
    ``sigma_{T+1}`` is computed. Residual *rows* (preserving cross-asset
    dependence) are resampled and rescaled by the current vol forecast.
    """

    def __init__(self) -> None:
        self._garch = GARCHModel()

    def _filter(
        self, portfolio: Portfolio, lookback: int
    ) -> tuple[np.ndarray, np.ndarray, pd.Index, list[str]]:
        """Return (standardised residuals, current vol forecast) per asset.

        Returns
        -------
        z : np.ndarray, shape (lookback, n_assets)
        sigma_forecast : np.ndarray, shape (n_assets,)
        index : pd.Index
        columns : list of str
        """
        rets = portfolio.returns.iloc[-lookback:]
        cols = list(rets.columns)
        n = portfolio.n_assets
        z = np.empty((len(rets), n))
        sigma_fc = np.empty(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(n):
                r = rets.iloc[:, j].to_numpy()
                params = self._garch.fit(r)
                z[:, j] = self._garch.standardized_residuals(r, params)
                sigma_fc[j] = self._garch.forecast_vol(r, params, horizon=1)
        return z, sigma_fc, rets.index, cols

    def compute(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizon_days: int = 1,
        n_resample: int = 10_000,
        lookback: int = 1_000,
        seed: int = DEFAULT_SEED,
    ) -> VaRResult:
        """Filtered Historical Simulation VaR.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 1
        n_resample : int, default 10000
            Number of bootstrap resamples of residual rows.
        lookback : int, default 1000
            Window used for GARCH fitting and the residual pool.
        seed : int, default 42

        Returns
        -------
        VaRResult
            ``method='FHS'``.
        """
        start = time.perf_counter()
        z, sigma_fc, _, _ = self._filter(portfolio, lookback)
        rng = np.random.default_rng(seed)
        rows = rng.integers(0, z.shape[0], size=n_resample)
        z_sim = z[rows, :]
        r_sim = z_sim * sigma_fc * np.sqrt(horizon_days)
        pnl = r_sim @ portfolio.weights * portfolio.notional
        var = float(-np.quantile(pnl, 1.0 - confidence))
        return VaRResult(
            var=var,
            var_pct=var / portfolio.notional,
            confidence=confidence,
            horizon_days=horizon_days,
            method="FHS",
            n_scenarios=n_resample,
            elapsed_seconds=time.perf_counter() - start,
        )

    def garch_vols(self, portfolio: Portfolio, lookback: int = 1_000) -> pd.DataFrame:
        """Per-asset GARCH conditional-volatility series.

        Parameters
        ----------
        portfolio : Portfolio
        lookback : int, default 1000

        Returns
        -------
        pd.DataFrame
            Conditional vols, dates x assets.
        """
        rets = portfolio.returns.iloc[-lookback:]
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in rets.columns:
                r = rets[col].to_numpy()
                params = self._garch.fit(r)
                out[col] = self._garch.conditional_vols(r, params)
        return pd.DataFrame(out, index=rets.index)

    def standardized_residuals(self, portfolio: Portfolio, lookback: int = 1_000) -> pd.DataFrame:
        """Per-asset standardised GARCH residuals.

        Parameters
        ----------
        portfolio : Portfolio
        lookback : int, default 1000

        Returns
        -------
        pd.DataFrame
            Residuals ``z_t``, dates x assets.
        """
        z, _, idx, cols = self._filter(portfolio, lookback)
        return pd.DataFrame(z, index=idx, columns=cols)

    def compare_to_historical(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        lookback: int = 1_000,
        n_resample: int = 10_000,
        seed: int = DEFAULT_SEED,
    ) -> pd.DataFrame:
        """Compare FHS VaR to plain historical-simulation VaR.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        lookback : int, default 1000
        n_resample : int, default 10000
        seed : int, default 42

        Returns
        -------
        pd.DataFrame
            Rows for ``FHS`` and ``Historical`` with VaR and VaR(%).
        """
        fhs = self.compute(
            portfolio, confidence=confidence, lookback=lookback, n_resample=n_resample, seed=seed
        )
        pnl = portfolio.pnl().to_numpy()[-lookback:]
        hist_var = float(-np.quantile(pnl, 1.0 - confidence))
        return pd.DataFrame(
            [
                {"method": "FHS", "var": fhs.var, "var_pct": fhs.var_pct},
                {
                    "method": "Historical",
                    "var": hist_var,
                    "var_pct": hist_var / portfolio.notional,
                },
            ]
        ).set_index("method")
