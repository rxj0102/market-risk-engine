"""Historical (non-parametric) Expected Shortfall estimators.

Expected Shortfall (ES, a.k.a. CVaR) is the average loss conditional on the
loss exceeding VaR.  The Basel IV / FRTB framework adopts ES at 97.5% as the
regulatory tail-risk measure.

References
----------
Acerbi, C., Tasche, D. (2002). "On the coherence of Expected Shortfall."
    Journal of Banking & Finance, 26(7), 1487-1503.
Basel Committee on Banking Supervision. "Minimum capital requirements for
    market risk." (FRTB), Jan 2019.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import ESResult, Portfolio


class HistoricalES:
    """Empirical Expected Shortfall from the realised P&L distribution."""

    def compute(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizon_days: int = 1,
        lookback: int = 500,
    ) -> ESResult:
        """Historical Expected Shortfall.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 1
            Scaled by ``sqrt(horizon_days)``.
        lookback : int, default 500

        Returns
        -------
        ESResult
            ``ES = -mean(PnL | PnL < -VaR)`` (positive dollars).
        """
        scale = np.sqrt(horizon_days)
        pnl = portfolio.pnl().to_numpy()[-lookback:] * scale
        var = -np.quantile(pnl, 1.0 - confidence)
        tail = pnl[pnl <= -var]
        if tail.size == 0:
            tail = np.array([pnl.min()])
        es = float(-tail.mean())
        return ESResult(
            es=es,
            es_pct=es / portfolio.notional,
            var=float(var),
            confidence=confidence,
            method="Historical-ES",
            tail_scenarios=tail,
            horizon_days=horizon_days,
        )

    def tail_scenarios(
        self, portfolio: Portfolio, confidence: float = 0.99, lookback: int = 500
    ) -> pd.DataFrame:
        """Dated P&L observations in the tail beyond VaR.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        lookback : int, default 500

        Returns
        -------
        pd.DataFrame
            Columns ``pnl`` and ``loss`` indexed by date, sorted worst-first.
        """
        pnl = portfolio.pnl().iloc[-lookback:]
        var = -np.quantile(pnl.to_numpy(), 1.0 - confidence)
        tail = pnl[pnl <= -var].sort_values()
        return pd.DataFrame({"pnl": tail, "loss": -tail})

    def es_by_asset(
        self, portfolio: Portfolio, confidence: float = 0.99, lookback: int = 500
    ) -> pd.DataFrame:
        """Per-asset P&L contribution averaged over the portfolio tail.

        This is the Euler/component decomposition of ES: the mean dollar P&L of
        each asset conditional on the *portfolio* being in its loss tail.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        lookback : int, default 500

        Returns
        -------
        pd.DataFrame
            Columns ``component_es`` and ``pct_of_total`` per asset.
        """
        rets = portfolio.returns.iloc[-lookback:]
        w = portfolio.weights
        pnl = rets.to_numpy() @ w * portfolio.notional
        var = -np.quantile(pnl, 1.0 - confidence)
        mask = pnl <= -var
        # Dollar P&L contribution of each asset on tail days.
        asset_pnl = rets.to_numpy()[mask] * w * portfolio.notional
        comp = -asset_pnl.mean(axis=0)
        total = comp.sum()
        return pd.DataFrame(
            {
                "component_es": comp,
                "pct_of_total": comp / total if total else np.full_like(comp, np.nan),
            },
            index=portfolio.asset_names,
        )

    def rolling_es(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        window: int = 250,
        step: int = 1,
    ) -> pd.Series:
        """Rolling-window historical ES time series.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        window : int, default 250
        step : int, default 1

        Returns
        -------
        pd.Series
            ES (positive dollars) indexed by window end date.
        """
        pnl = portfolio.pnl()
        vals = pnl.to_numpy()
        idx = pnl.index
        dates, out = [], []
        for end in range(window, len(vals) + 1, step):
            block = vals[end - window : end]
            var = -np.quantile(block, 1.0 - confidence)
            tail = block[block <= -var]
            if tail.size == 0:
                tail = np.array([block.min()])
            out.append(-tail.mean())
            dates.append(idx[end - 1])
        return pd.Series(out, index=pd.Index(dates, name=idx.name), name="rolling_es")

    def stressed_es(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        stress_start: str | None = None,
        stress_end: str | None = None,
    ) -> ESResult:
        """Stressed ES computed over a historical stress window.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        stress_start, stress_end : str, optional
            Date bounds (inclusive) selecting the stress period. If omitted the
            full sample is used.

        Returns
        -------
        ESResult
            ``method='Stressed-ES'``.
        """
        pnl = portfolio.pnl()
        if stress_start is not None or stress_end is not None:
            pnl = pnl.loc[stress_start:stress_end]
        arr = pnl.to_numpy()
        var = -np.quantile(arr, 1.0 - confidence)
        tail = arr[arr <= -var]
        if tail.size == 0:
            tail = np.array([arr.min()])
        es = float(-tail.mean())
        return ESResult(
            es=es,
            es_pct=es / portfolio.notional,
            var=float(var),
            confidence=confidence,
            method="Stressed-ES",
            tail_scenarios=tail,
            horizon_days=1,
        )
