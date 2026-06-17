"""Comparison and selection of VaR models via rolling backtesting.

The :class:`VaRModelComparison` ranks the supported VaR methods on their
out-of-sample backtest performance and recommends a method that best balances
statistical adequacy (Kupiec / Christoffersen p-values) against accuracy of the
exception rate relative to the target.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from core import Portfolio
from backtesting import SUPPORTED_METHODS, BacktestEngine

#: Penalty weights used to combine the backtest diagnostics into a score.
_W_KUPIEC: float = 1.0
_W_CC: float = 1.0
_W_RATE: float = 50.0

#: Basel traffic-light penalties (lower is better).
_TRAFFIC_PENALTY: dict[str, float] = {"green": 0.0, "yellow": 1.0, "red": 3.0}


class VaRModelComparison:
    """Backtest-driven comparison and selection of VaR estimators."""

    def __init__(self) -> None:
        self._engine = BacktestEngine()

    # ----------------------------------------------------------- ranking
    def rank_models(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        window: int = 250,
        estimation_window: int = 500,
    ) -> pd.DataFrame:
        """Backtest and rank every supported VaR method.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        window : int, default 250
            Out-of-sample test window (days).
        estimation_window : int, default 500
            Rolling estimation window (days).

        Returns
        -------
        pd.DataFrame
            Columns ``Method``, ``Exception Rate``, ``Kupiec p-val``,
            ``CC p-val``, ``Traffic Light``, ``Avg VaR`` (mean dollar VaR over
            the test window), ``Score`` and ``Rank`` (1 = best). Higher
            p-values, an exception rate near the ``1 - confidence`` target and a
            greener traffic light all improve the score.
        """
        target = 1.0 - confidence
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for method in SUPPORTED_METHODS:
                res, frame = self._engine.run(
                    portfolio,
                    var_method=method,
                    confidence=confidence,
                    test_window=window,
                    estimation_window=estimation_window,
                )
                score = (
                    _W_KUPIEC * res.kupiec_pvalue
                    + _W_CC * res.christoffersen_pvalue
                    - _W_RATE * abs(res.exception_rate - target)
                    - _TRAFFIC_PENALTY.get(res.traffic_light, 0.0)
                )
                rows.append(
                    {
                        "Method": method,
                        "Exception Rate": res.exception_rate,
                        "Kupiec p-val": res.kupiec_pvalue,
                        "CC p-val": res.christoffersen_pvalue,
                        "Traffic Light": res.traffic_light,
                        "Avg VaR": float(frame["var"].mean()),
                        "Score": float(score),
                    }
                )

        df = pd.DataFrame(rows)
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"] = np.arange(1, len(df) + 1)
        return df

    # ------------------------------------------------ time-series compare
    def time_series_comparison(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        window: int = 250,
        estimation_window: int = 500,
    ) -> pd.DataFrame:
        """Aligned rolling VaR series for each supported method.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        window : int, default 250
            Out-of-sample test window length used for alignment.
        estimation_window : int, default 500

        Returns
        -------
        pd.DataFrame
            One column per method holding the rolling dollar VaR, plus a
            ``pnl`` column, all aligned on the common test dates.
        """
        out: dict[str, pd.Series] = {}
        pnl_series: pd.Series | None = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for method in SUPPORTED_METHODS:
                _, frame = self._engine.run(
                    portfolio,
                    var_method=method,
                    confidence=confidence,
                    test_window=window,
                    estimation_window=estimation_window,
                )
                out[method] = frame["var"]
                if pnl_series is None:
                    pnl_series = frame["pnl"]

        df = pd.DataFrame(out)
        if pnl_series is not None:
            df["pnl"] = pnl_series
        return df

    # ------------------------------------------------- selection report
    def model_selection_report(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        window: int = 250,
        estimation_window: int = 500,
    ) -> dict[str, object]:
        """Recommend a VaR method with a textual justification.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        window : int, default 250
        estimation_window : int, default 500

        Returns
        -------
        dict
            ``recommended_method``, ``reasoning`` (a human-readable string),
            ``confidence`` and ``ranking`` (the full :meth:`rank_models` table).
        """
        ranking = self.rank_models(portfolio, confidence, window, estimation_window)
        best = ranking.iloc[0]
        target = 1.0 - confidence

        reasoning = (
            f"Recommended method '{best['Method']}' ranked #1 across "
            f"{len(ranking)} candidates over a {window}-day backtest at "
            f"{confidence:.0%} confidence. It achieved an exception rate of "
            f"{best['Exception Rate']:.2%} (target {target:.2%}), a Kupiec "
            f"p-value of {best['Kupiec p-val']:.3f} and a Christoffersen "
            f"conditional-coverage p-value of {best['CC p-val']:.3f}, landing in "
            f"the Basel '{best['Traffic Light']}' zone. Higher p-values indicate "
            f"the model's coverage is statistically consistent with the target, "
            f"while the exception rate closest to target reflects accurate tail "
            f"calibration."
        )

        return {
            "recommended_method": str(best["Method"]),
            "reasoning": reasoning,
            "confidence": confidence,
            "ranking": ranking,
        }
