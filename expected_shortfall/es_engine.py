"""Unified Expected Shortfall engine aggregating every ES estimator."""

from __future__ import annotations

import pandas as pd

from core import ESResult, Portfolio

from expected_shortfall.es_historical import HistoricalES
from expected_shortfall.es_monte_carlo import MCExpectedShortfall
from expected_shortfall.es_parametric import ParametricES


class ESEngine:
    """Aggregate front-end over every Expected Shortfall estimator."""

    def __init__(self) -> None:
        self.historical = HistoricalES()
        self.parametric = ParametricES()
        self.monte_carlo = MCExpectedShortfall()

    def compute_all(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        n_paths: int = 200_000,
        seed: int = 42,
    ) -> dict[str, ESResult]:
        """Compute every ES method and return them keyed by name.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        n_paths : int, default 200000
            Monte-Carlo path count.
        seed : int, default 42

        Returns
        -------
        dict[str, ESResult]
            Keys: ``historical``, ``parametric_normal``, ``parametric_t``,
            ``cornish_fisher``, ``monte_carlo_normal``, ``monte_carlo_t``.
        """
        return {
            "historical": self.historical.compute(portfolio, confidence),
            "parametric_normal": self.parametric.compute(
                portfolio, confidence, distribution="normal"
            ),
            "parametric_t": self.parametric.compute(
                portfolio, confidence, distribution="student_t"
            ),
            "cornish_fisher": self.parametric.cornish_fisher_es(portfolio, confidence),
            "monte_carlo_normal": self.monte_carlo.compute(
                portfolio, confidence, n_paths=n_paths, distribution="normal", seed=seed
            ),
            "monte_carlo_t": self.monte_carlo.compute(
                portfolio, confidence, n_paths=n_paths, distribution="student_t", seed=seed
            ),
        }

    def comparison_table(
        self, portfolio: Portfolio, confidence: float = 0.99
    ) -> pd.DataFrame:
        """Side-by-side ES comparison across methods.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99

        Returns
        -------
        pd.DataFrame
            Columns ``Method``, ``ES($)``, ``ES(%)``, ``VaR($)``, ``ES/VaR``.
        """
        results = self.compute_all(portfolio, confidence)
        rows = []
        for res in results.values():
            rows.append(
                {
                    "Method": res.method,
                    "ES($)": res.es,
                    "ES(%)": res.es_pct,
                    "VaR($)": res.var,
                    "ES/VaR": res.es_var_ratio,
                }
            )
        return pd.DataFrame(rows)
