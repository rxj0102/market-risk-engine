"""Top-level stress-testing orchestration.

Aggregates the historical, hypothetical, reverse and sensitivity engines into
a single entry point and produces a ranked worst-scenario summary.

References
----------
Basel Committee on Banking Supervision. "Principles for sound stress testing
    practices and supervision." May 2009.
"""

from __future__ import annotations

import pandas as pd

from core import Portfolio
from stress_testing.historical_scenarios import HistoricalScenarioEngine
from stress_testing.hypothetical_scenarios import HypotheticalScenarioEngine
from stress_testing.reverse_stress import ReverseStressTester
from stress_testing.sensitivity_analysis import SensitivityAnalyzer


class StressTestEngine:
    """Unified facade over the stress-testing sub-engines.

    Parameters
    ----------
    reverse_loss_threshold : float
        Loss fraction used for the reverse-stress example in :meth:`run_all`.
    """

    def __init__(self, reverse_loss_threshold: float = 0.10) -> None:
        self.historical = HistoricalScenarioEngine()
        self.hypothetical = HypotheticalScenarioEngine()
        self.reverse = ReverseStressTester()
        self.sensitivity = SensitivityAnalyzer()
        self.reverse_loss_threshold = reverse_loss_threshold

    # --------------------------------------------------------------- run_all
    def run_all(self, portfolio: Portfolio) -> dict[str, object]:
        """Run the full stress-testing suite.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to stress.

        Returns
        -------
        dict
            ``{'historical', 'hypothetical', 'reverse_stress', 'tornado',
            'correlation_shock'}``.
        """
        return {
            "historical": self.historical.run_all(portfolio),
            "hypothetical": self.hypothetical.run_all_hypothetical(portfolio),
            "reverse_stress": self.reverse.minimum_stress_scenario(
                portfolio, self.reverse_loss_threshold
            ),
            "tornado": self.sensitivity.tornado_chart_data(portfolio),
            "correlation_shock": self.hypothetical.apply_correlation_shock(portfolio),
        }

    # --------------------------------------------------------- summary_table
    def summary_table(self, portfolio: Portfolio) -> pd.DataFrame:
        """Worst scenarios across engines, ranked by dollar loss.

        Combines historical and hypothetical scenarios into one table sorted
        by ascending P&L (most negative first).

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to stress.

        Returns
        -------
        pd.DataFrame
            Columns ``Scenario``, ``Type``, ``P&L($)``, ``P&L(%)``.
        """
        rows: list[dict[str, object]] = []

        for rec in self.historical.scenarios:
            res = self.historical.apply_scenario(portfolio, rec["name"])
            if "(outside sample data window)" in res.description:
                continue
            rows.append(
                {
                    "Scenario": res.name,
                    "Type": "Historical",
                    "P&L($)": res.pnl,
                    "P&L(%)": res.pnl_pct,
                }
            )

        for sc in self.hypothetical.scenarios:
            res = self.hypothetical.apply_scenario(portfolio, sc)
            rows.append(
                {
                    "Scenario": res.name,
                    "Type": "Hypothetical",
                    "P&L($)": res.pnl,
                    "P&L(%)": res.pnl_pct,
                }
            )

        df = pd.DataFrame(rows).sort_values("P&L($)").reset_index(drop=True)
        return df
