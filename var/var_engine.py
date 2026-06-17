"""Unified VaR engine aggregating every estimator in the package.

Provides a one-call computation of all methods, a side-by-side comparison table
(including Expected Shortfall) and a rolling out-of-sample comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import Portfolio, VaRResult

from var.filtered_historical import FilteredHistoricalSimulation
from var.historical import HistoricalVaR
from var.monte_carlo import MonteCarloVaR
from var.parametric import ParametricVaR


class VaREngine:
    """Aggregate front-end over every VaR estimator in the package."""

    def __init__(self) -> None:
        self.historical = HistoricalVaR()
        self.parametric = ParametricVaR()
        self.monte_carlo = MonteCarloVaR()
        self.fhs = FilteredHistoricalSimulation()

    def compute_all(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizon_days: int = 1,
        n_paths: int = 50_000,
        n_resample: int = 5_000,
        seed: int = 42,
    ) -> dict[str, VaRResult]:
        """Compute every VaR method and return them keyed by name.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 1
        n_paths : int, default 50000
            Monte-Carlo path count (kept modest for speed).
        n_resample : int, default 5000
            FHS resample count.
        seed : int, default 42

        Returns
        -------
        dict[str, VaRResult]
            Keys: ``historical``, ``age_weighted``, ``vol_scaled``,
            ``parametric_normal``, ``parametric_t``, ``monte_carlo_normal``,
            ``monte_carlo_t``, ``fhs``.
        """
        return {
            "historical": self.historical.compute(
                portfolio, confidence, horizon_days=horizon_days
            ),
            "age_weighted": self.historical.compute_age_weighted(portfolio, confidence),
            "vol_scaled": self.historical.compute_vol_scaled(portfolio, confidence),
            "parametric_normal": self.parametric.compute(
                portfolio, confidence, horizon_days=horizon_days
            ),
            "parametric_t": self.parametric.compute_student_t(
                portfolio, confidence, horizon_days=horizon_days
            ),
            "monte_carlo_normal": self.monte_carlo.compute(
                portfolio, confidence, horizon_days=horizon_days,
                n_paths=n_paths, distribution="normal", seed=seed,
            ),
            "monte_carlo_t": self.monte_carlo.compute(
                portfolio, confidence, horizon_days=horizon_days,
                n_paths=n_paths, distribution="student_t", seed=seed,
            ),
            "fhs": self.fhs.compute(
                portfolio, confidence, horizon_days=horizon_days, n_resample=n_resample, seed=seed
            ),
        }

    def comparison_table(
        self, portfolio: Portfolio, confidence: float = 0.99, horizon_days: int = 1
    ) -> pd.DataFrame:
        """Side-by-side VaR and (where available) ES comparison.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 1

        Returns
        -------
        pd.DataFrame
            Columns ``Method``, ``VaR($)``, ``VaR(%)``, ``ES($)``, ``ES(%)``.
        """
        # Imported here to avoid a hard import cycle at module load time.
        from expected_shortfall import HistoricalES, ParametricES

        results = self.compute_all(portfolio, confidence, horizon_days=horizon_days)

        hist_es = HistoricalES().compute(portfolio, confidence)
        param_es_n = ParametricES().compute(portfolio, confidence, distribution="normal")
        param_es_t = ParametricES().compute(portfolio, confidence, distribution="student_t")

        es_map = {
            "historical": hist_es,
            "parametric_normal": param_es_n,
            "parametric_t": param_es_t,
        }

        rows = []
        for key, res in results.items():
            es = es_map.get(key)
            rows.append(
                {
                    "Method": res.method,
                    "VaR($)": res.var,
                    "VaR(%)": res.var_pct,
                    "ES($)": es.es if es is not None else np.nan,
                    "ES(%)": es.es_pct if es is not None else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def rolling_comparison(
        self, portfolio: Portfolio, confidence: float = 0.99, window: int = 250
    ) -> pd.DataFrame:
        """Rolling historical vs. parametric VaR over time.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        window : int, default 250

        Returns
        -------
        pd.DataFrame
            Columns ``historical`` and ``parametric`` indexed by date.
        """
        pnl = portfolio.pnl()
        port_ret = portfolio.return_series().to_numpy()
        vals = pnl.to_numpy()
        idx = pnl.index
        z = float(-__import__("scipy.stats", fromlist=["norm"]).norm.ppf(1.0 - confidence))
        dates, hist, param = [], [], []
        for end in range(window, len(vals) + 1):
            block = vals[end - window : end]
            hist.append(-np.quantile(block, 1.0 - confidence))
            sigma = port_ret[end - window : end].std(ddof=1)
            param.append(z * sigma * portfolio.notional)
            dates.append(idx[end - 1])
        return pd.DataFrame(
            {"historical": hist, "parametric": param},
            index=pd.Index(dates, name=idx.name),
        )
