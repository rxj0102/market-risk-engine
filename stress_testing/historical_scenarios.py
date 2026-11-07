"""Historical (event-replay) stress testing.

Replays realised market crises onto the current portfolio by slicing the
historical return panel to each crisis window and computing the resulting
profit-and-loss path.

References
----------
Basel Committee on Banking Supervision. "Principles for sound stress testing
    practices and supervision." May 2009.
Berkowitz, J. (2000). "A coherent framework for stress testing."
    *Journal of Risk* 2(2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import Portfolio, ScenarioResult
from utils.statistics import max_drawdown as _max_drawdown

# ---------------------------------------------------------------------------
# Built-in crisis library
# ---------------------------------------------------------------------------
SCENARIOS: list[dict[str, str]] = [
    {
        "name": "Black Monday 1987",
        "start_date": "1987-10-14",
        "end_date": "1987-10-26",
        "description": "October 1987 global equity crash (-20% single day).",
        "peak_loss_date": "1987-10-19",
    },
    {
        "name": "LTCM/Russia 1998",
        "start_date": "1998-08-01",
        "end_date": "1998-10-31",
        "description": "Russian default and LTCM collapse, liquidity spiral.",
        "peak_loss_date": "1998-08-31",
    },
    {
        "name": "Dot-com Crash",
        "start_date": "2000-03-01",
        "end_date": "2002-10-31",
        "description": "Bursting of the technology/internet bubble.",
        "peak_loss_date": "2001-09-21",
    },
    {
        "name": "9/11 Attacks 2001",
        "start_date": "2001-09-10",
        "end_date": "2001-09-28",
        "description": "Market reopening shock after the September 11 attacks.",
        "peak_loss_date": "2001-09-17",
    },
    {
        "name": "Lehman/GFC 2008",
        "start_date": "2008-09-01",
        "end_date": "2009-03-31",
        "description": "Global financial crisis triggered by Lehman bankruptcy.",
        "peak_loss_date": "2008-10-10",
    },
    {
        "name": "European Sovereign Debt",
        "start_date": "2010-04-01",
        "end_date": "2012-07-31",
        "description": "Eurozone sovereign debt and banking crisis.",
        "peak_loss_date": "2011-08-08",
    },
    {
        "name": "Taper Tantrum 2013",
        "start_date": "2013-05-01",
        "end_date": "2013-06-30",
        "description": "Bond sell-off on Fed tapering signals.",
        "peak_loss_date": "2013-06-20",
    },
    {
        "name": "China Devaluation 2015",
        "start_date": "2015-08-01",
        "end_date": "2015-09-30",
        "description": "CNY devaluation and global equity rout.",
        "peak_loss_date": "2015-08-24",
    },
    {
        "name": "COVID Crash 2020",
        "start_date": "2020-02-19",
        "end_date": "2020-03-23",
        "description": "Pandemic-driven liquidation and fastest bear market.",
        "peak_loss_date": "2020-03-16",
    },
    {
        "name": "2022 Rate-Hike Bear",
        "start_date": "2022-01-03",
        "end_date": "2022-10-12",
        "description": "Inflation-driven Fed hiking cycle equity/bond drawdown.",
        "peak_loss_date": "2022-09-30",
    },
]


class HistoricalScenarioEngine:
    """Replay historical market crises onto a portfolio.

    The engine carries a built-in library of ten named crises
    (:data:`SCENARIOS`) and applies each by slicing the portfolio's
    historical return panel to the crisis window.

    Notes
    -----
    The bundled sample data only spans 2015-2024.  Crises falling entirely
    before 2015 therefore have no data and are reported as zero-P&L results
    flagged ``"(outside sample data window)"`` rather than raising.
    """

    def __init__(self, scenarios: list[dict[str, str]] | None = None) -> None:
        self.scenarios = list(scenarios) if scenarios is not None else list(SCENARIOS)
        self._by_name = {s["name"]: s for s in self.scenarios}

    # ------------------------------------------------------------------ utils
    def _lookup(self, scenario_name: str) -> dict[str, str]:
        """Return the scenario record for ``scenario_name``."""
        if scenario_name not in self._by_name:
            raise KeyError(f"Unknown scenario {scenario_name!r}")
        return self._by_name[scenario_name]

    @staticmethod
    def _window(portfolio: Portfolio, start: str, end: str) -> pd.DataFrame:
        """Slice the portfolio returns to ``[start, end]`` (inclusive)."""
        idx = portfolio.returns.index
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        return portfolio.returns.loc[mask]

    # -------------------------------------------------------- apply_scenario
    def apply_scenario(self, portfolio: Portfolio, scenario_name: str) -> ScenarioResult:
        """Apply a named historical crisis to ``portfolio``.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to stress.
        scenario_name : str
            Name of a crisis in the library.

        Returns
        -------
        ScenarioResult
            Period P&L and risk statistics.  If the window has no data
            (pre-2015) all numeric fields are zero and the description is
            flagged ``"(outside sample data window)"``.
        """
        rec = self._lookup(scenario_name)
        window = self._window(portfolio, rec["start_date"], rec["end_date"])

        if window.empty:
            return ScenarioResult(
                name=scenario_name,
                pnl=0.0,
                pnl_pct=0.0,
                max_1d_loss=0.0,
                max_drawdown=0.0,
                total_loss=0.0,
                asset_pnl=np.zeros(portfolio.n_assets),
                description=f"{rec['description']} (outside sample data window)",
                recovery_days=-1,
            )

        w = portfolio.weights
        notional = portfolio.notional
        ret = window.to_numpy()

        # Per-day portfolio return and P&L.
        port_ret = ret @ w
        daily_pnl = port_ret * notional
        cum_pnl = np.cumsum(daily_pnl)

        # Per-asset P&L: w_i * sum_t r_{i,t} * notional (simple-return sum).
        asset_pnl = (ret.sum(axis=0) * w) * notional

        total_pnl = float(cum_pnl[-1])
        max_1d_loss = float(-daily_pnl.min()) if daily_pnl.min() < 0 else 0.0
        mdd = float(_max_drawdown(port_ret))
        # Total loss: magnitude of deepest cumulative drawdown in $ (>= 0).
        total_loss = float(-cum_pnl.min()) if cum_pnl.min() < 0 else 0.0

        recovery_days = self._recovery_days(cum_pnl)

        return ScenarioResult(
            name=scenario_name,
            pnl=total_pnl,
            pnl_pct=total_pnl / notional,
            max_1d_loss=max_1d_loss,
            max_drawdown=mdd,
            total_loss=total_loss,
            asset_pnl=asset_pnl,
            description=rec["description"],
            recovery_days=recovery_days,
        )

    @staticmethod
    def _recovery_days(cum_pnl: np.ndarray) -> int:
        """Trading days from the trough back to the prior cumulative peak.

        Returns ``-1`` if the prior peak is never recovered within the window.
        """
        if cum_pnl.size == 0:
            return -1
        running_peak = np.maximum.accumulate(cum_pnl)
        drawdown = cum_pnl - running_peak
        trough = int(np.argmin(drawdown))
        if drawdown[trough] >= 0:
            return 0
        peak_value = running_peak[trough]
        # First day at/after the trough whose cumulative P&L recovers the peak.
        after = cum_pnl[trough:]
        recovered = np.where(after >= peak_value)[0]
        if recovered.size == 0:
            return -1
        return int(recovered[0])

    # --------------------------------------------------------------- run_all
    def run_all(self, portfolio: Portfolio) -> pd.DataFrame:
        """Apply every library scenario and tabulate the results.

        Returns
        -------
        pd.DataFrame
            Columns ``Scenario``, ``1D Loss($)``, ``1D Loss(%)``,
            ``Max Drawdown``, ``Total Loss($)``, ``Period Return(%)``.
            Scenarios with no data (pre-2015) carry ``NaN`` numerics.
        """
        rows: list[dict[str, object]] = []
        for rec in self.scenarios:
            res = self.apply_scenario(portfolio, rec["name"])
            has_data = "(outside sample data window)" not in res.description
            rows.append(
                {
                    "Scenario": rec["name"],
                    "1D Loss($)": res.max_1d_loss if has_data else np.nan,
                    "1D Loss(%)": (res.max_1d_loss / portfolio.notional) if has_data else np.nan,
                    "Max Drawdown": res.max_drawdown if has_data else np.nan,
                    "Total Loss($)": res.total_loss if has_data else np.nan,
                    "Period Return(%)": res.pnl_pct if has_data else np.nan,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------- worst_day
    def worst_day(self, portfolio: Portfolio, lookback_years: int = 20) -> dict[str, object]:
        """Single worst historical portfolio P&L day in the lookback window.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.
        lookback_years : int
            Number of trailing calendar years to consider.

        Returns
        -------
        dict
            ``{'date', 'pnl', 'pnl_pct', 'return'}`` for the worst day.
        """
        pnl = portfolio.pnl()
        cutoff = pnl.index.max() - pd.DateOffset(years=lookback_years)
        pnl = pnl.loc[pnl.index >= cutoff]
        if pnl.empty:
            return {"date": None, "pnl": float("nan"), "pnl_pct": float("nan"), "return": float("nan")}
        worst_date = pnl.idxmin()
        worst_pnl = float(pnl.min())
        return {
            "date": worst_date,
            "pnl": worst_pnl,
            "pnl_pct": worst_pnl / portfolio.notional,
            "return": worst_pnl / portfolio.notional,
        }

    # ----------------------------------------------- scenario_correlation_stress
    def scenario_correlation_stress(
        self, portfolio: Portfolio, scenario_name: str
    ) -> dict[str, pd.DataFrame]:
        """Compare asset correlations inside a crisis vs. the full sample.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio whose asset return panel is used.
        scenario_name : str
            Name of a crisis in the library.

        Returns
        -------
        dict
            ``{'crisis': pd.DataFrame, 'normal': pd.DataFrame}`` correlation
            matrices.  ``'crisis'`` is ``None``-safe (empty frame) when the
            window has no data.
        """
        rec = self._lookup(scenario_name)
        window = self._window(portfolio, rec["start_date"], rec["end_date"])
        normal_corr = portfolio.returns.corr()
        if window.empty or len(window) < 2:
            crisis_corr = pd.DataFrame(
                np.nan, index=portfolio.asset_names, columns=portfolio.asset_names
            )
        else:
            crisis_corr = window.corr()
        return {"crisis": crisis_corr, "normal": normal_corr}
