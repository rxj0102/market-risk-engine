"""Unified rolling VaR backtesting pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from core import RISKMETRICS_LAMBDA, AGE_WEIGHT_LAMBDA, BacktestResult, Portfolio
from utils.statistics import z_score, weighted_quantile
from backtesting.christoffersen import ChristoffersenTest
from backtesting.kupiec import KupiecTest
from backtesting.traffic_light import BaselTrafficLight

#: VaR methods supported by the rolling engine.
SUPPORTED_METHODS = ("historical", "parametric_normal", "ewma", "age_weighted")


class BacktestEngine:
    """Rolling out-of-sample VaR backtesting over a fixed test window.

    For each test day the VaR is estimated from the preceding
    ``estimation_window`` of portfolio returns and compared against the realised
    next-day P&L. Exception statistics feed the Kupiec, Christoffersen and Basel
    traffic-light diagnostics.

    Notes
    -----
    Historical VaR is vectorised over rolling windows via
    :func:`numpy.lib.stride_tricks.sliding_window_view`; the EWMA path is a
    single recursive pass. No per-day re-slicing of the full history occurs.
    """

    def __init__(self) -> None:
        self._kupiec = KupiecTest()
        self._christ = ChristoffersenTest()
        self._tl = BaselTrafficLight()

    # ------------------------------------------------------ VaR series
    def _rolling_var(
        self,
        returns: np.ndarray,
        method: str,
        confidence: float,
        estimation_window: int,
    ) -> np.ndarray:
        """Rolling 1-day VaR (as a positive return-space loss) per test day.

        Returns an array aligned to days ``estimation_window .. T-1`` (the days
        that have a full estimation window preceding them).
        """
        z = z_score(confidence)
        windows = sliding_window_view(returns, estimation_window)[:-1]  # window ends at t-1
        if method == "historical":
            return -np.quantile(windows, 1 - confidence, axis=1)
        if method == "parametric_normal":
            return z * windows.std(axis=1, ddof=1)
        if method == "age_weighted":
            T = estimation_window
            w = AGE_WEIGHT_LAMBDA ** (T - 1 - np.arange(T))
            w /= w.sum()
            out = np.empty(windows.shape[0])
            for i in range(windows.shape[0]):
                out[i] = -weighted_quantile(windows[i], w, 1 - confidence)
            return out
        if method == "ewma":
            # Recursive EWMA variance over the full series, sampled at window ends.
            var = np.empty_like(returns)
            var[0] = returns[:estimation_window].var()
            lam = RISKMETRICS_LAMBDA
            for t in range(1, len(returns)):
                var[t] = lam * var[t - 1] + (1 - lam) * returns[t - 1] ** 2
            sigma = np.sqrt(var[estimation_window - 1 : -1])
            return z * sigma
        raise ValueError(f"Unsupported method {method!r}; choose from {SUPPORTED_METHODS}")

    # --------------------------------------------------------------- run
    def run(
        self,
        portfolio: Portfolio,
        var_method: str = "historical",
        confidence: float = 0.99,
        test_window: int = 250,
        estimation_window: int = 500,
    ) -> tuple[BacktestResult, pd.DataFrame]:
        """Run a rolling backtest.

        Returns
        -------
        tuple
            ``(BacktestResult, frame)`` where ``frame`` has columns ``pnl``,
            ``var`` (positive) and ``exception`` indexed by date.
        """
        r = portfolio.return_series()
        dates = r.index
        rv = r.to_numpy()
        notional = portfolio.notional

        var_returns = self._rolling_var(rv, var_method, confidence, estimation_window)
        var_dollar = var_returns * notional
        aligned_dates = dates[estimation_window:]
        aligned_pnl = rv[estimation_window:] * notional

        # Restrict to the last test_window observations.
        if test_window < len(aligned_pnl):
            var_dollar = var_dollar[-test_window:]
            aligned_pnl = aligned_pnl[-test_window:]
            aligned_dates = aligned_dates[-test_window:]

        var_series = pd.Series(var_dollar, index=aligned_dates, name="var")
        pnl_series = pd.Series(aligned_pnl, index=aligned_dates, name="pnl")
        exceptions = (pnl_series < -var_series).astype(int)

        frame = pd.DataFrame(
            {"pnl": pnl_series, "var": var_series, "exception": exceptions}
        )

        T = len(pnl_series)
        n_exc = int(exceptions.sum())
        kup_stat, kup_p, _ = self._kupiec.test(var_series, pnl_series, confidence)
        cc_stat, cc_p, _ = self._christ.conditional_coverage_test(var_series, pnl_series, confidence)
        zone, mult = self._tl.classify(n_exc, T)

        result = BacktestResult(
            exceptions=n_exc,
            expected_exceptions=(1 - confidence) * T,
            exception_rate=n_exc / T if T else float("nan"),
            kupiec_stat=kup_stat,
            kupiec_pvalue=kup_p,
            christoffersen_stat=cc_stat,
            christoffersen_pvalue=cc_p,
            traffic_light=zone,
            days_tested=T,
            capital_multiplier=mult,
        )
        return result, frame

    def run_all_methods(
        self, portfolio: Portfolio, confidence: float = 0.99, test_window: int = 250,
        estimation_window: int = 500,
    ) -> pd.DataFrame:
        """Backtest every supported method and tabulate the diagnostics."""
        rows = []
        for method in SUPPORTED_METHODS:
            res, _ = self.run(portfolio, method, confidence, test_window, estimation_window)
            rows.append(
                {
                    "Method": method,
                    "Exceptions": res.exceptions,
                    "Expected": round(res.expected_exceptions, 2),
                    "Kupiec p-val": round(res.kupiec_pvalue, 4),
                    "CC p-val": round(res.christoffersen_pvalue, 4),
                    "Traffic Light": res.traffic_light,
                    "Capital Mult.": round(res.capital_multiplier, 2),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def exception_report(var_series: pd.Series, pnl_series: pd.Series) -> pd.DataFrame:
        """Table of breach dates with loss magnitude and VaR overshoot."""
        var = np.abs(var_series)
        mask = pnl_series < -var
        df = pd.DataFrame(
            {
                "pnl": pnl_series[mask],
                "var": var[mask],
                "overshoot": (-pnl_series[mask] - var[mask]),
            }
        )
        return df

    @staticmethod
    def rolling_exception_rate(
        var_series: pd.Series, pnl_series: pd.Series, rolling_window: int = 250
    ) -> pd.Series:
        """Rolling realised exception rate over ``rolling_window`` days."""
        exc = (pnl_series < -np.abs(var_series)).astype(int)
        return exc.rolling(rolling_window, min_periods=1).mean()
