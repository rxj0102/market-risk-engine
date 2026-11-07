"""Basel II/III traffic-light backtesting zones and capital multipliers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core import BASE_CAPITAL_MULTIPLIER


class BaselTrafficLight:
    """Basel traffic-light classification for a 250-day VaR backtest.

    Notes
    -----
    Over a 250-day window at 99% confidence the Basel framework defines:

    * **Green** (0-4 exceptions): model accepted, multiplier 3.00.
    * **Yellow** (5-9 exceptions): scrutiny, multiplier 3.00 + plus-factor.
    * **Red** (10+ exceptions): model rejected, multiplier 4.00.

    Plus-factors: 5→0.40, 6→0.50, 7→0.65, 8→0.75, 9→0.85.

    References
    ----------
    BCBS (1996, revised). "Supervisory framework for the use of 'backtesting'
    in conjunction with the internal models approach to market risk capital."
    """

    PLUS_FACTORS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}

    def capital_add_on(self, n_exceptions: int) -> float:
        """Plus-factor added to the base multiplier of 3.0."""
        if n_exceptions <= 4:
            return 0.0
        if n_exceptions >= 10:
            return 1.0
        return self.PLUS_FACTORS[n_exceptions]

    def classify(
        self, n_exceptions: int, n_days: int = 250, confidence: float = 0.99
    ) -> tuple[str, float]:
        """Classify the backtest into a zone and capital multiplier.

        The Basel zones are defined on the cumulative binomial probability of
        the exception count: green below 95%, red from 99.99%, yellow in
        between.  For 250 days at 99% this reproduces the published 0-4 / 5-9
        / 10+ table exactly; other window lengths use the same probability
        cut-offs instead of scaling the 250-day counts.  Yellow-zone plus
        factors come from the published table at the 250-day-equivalent
        exception count.
        """
        if n_days <= 0:
            return "green", BASE_CAPITAL_MULTIPLIER
        cdf = float(stats.binom.cdf(n_exceptions, n_days, 1.0 - confidence))
        if cdf < 0.95:
            return "green", BASE_CAPITAL_MULTIPLIER
        if cdf >= 0.9999:
            return "red", BASE_CAPITAL_MULTIPLIER + 1.0
        equivalent = int(round(n_exceptions * 250.0 / n_days))
        equivalent = min(max(equivalent, 5), 9)
        return "yellow", BASE_CAPITAL_MULTIPLIER + self.PLUS_FACTORS[equivalent]

    def backtest_report(
        self, var_series: pd.Series, pnl_series: pd.Series, confidence: float = 0.99
    ) -> dict:
        """Full traffic-light report for a VaR/P&L pair."""
        var = np.abs(np.asarray(var_series, dtype=float))
        pnl = np.asarray(pnl_series, dtype=float)
        mask = pnl < -var
        n_exc = int(mask.sum())
        T = len(pnl)
        zone, mult = self.classify(n_exc, T, confidence)
        return {
            "exceptions": n_exc,
            "days_tested": T,
            "expected_exceptions": (1 - confidence) * T,
            "exception_rate": n_exc / T if T else float("nan"),
            "traffic_light": zone,
            "capital_multiplier": mult,
            "capital_add_on": mult - BASE_CAPITAL_MULTIPLIER,
        }
