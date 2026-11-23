"""Regulatory market-risk capital: Basel III IMA and Basel IV (FRTB).

Two regimes are implemented:

* **Basel 2.5 / III** Internal Models Approach (IMA), VaR-based, on a 10-day
  99% VaR (one-day VaR scaled by sqrt(10), as the rules allow)::

      MRC = max(VaR_{t-1}, mc * avg(VaR, last 60 days))
          + max(sVaR_{t-1}, ms * avg(sVaR, last 60 days))

  mc = ms = 3 plus the traffic-light add-on. Here sVaR is a single stressed
  estimate, so its second term is ms * sVaR.

* **FRTB** IMA, Expected Shortfall at 97.5% on a 10-day base horizon, with the
  liquidity-horizon cascade of MAR33.4::

      ES = sqrt( sum_j ( ES_T(P, j) * sqrt((LH_j - LH_{j-1}) / T) )^2 )

  where T = 10, LH_j runs over 10, 20, 40, 60, 120 days and ES_T(P, j) is the
  10-day ES of the positions whose liquidity horizon is at least LH_j.

References
----------
BCBS (2019). "Minimum capital requirements for market risk." (FRTB).
BCBS (2009). "Revisions to the Basel II market risk framework." (SVaR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core import BASE_CAPITAL_MULTIPLIER, FRTB_CONFIDENCE, TRADING_DAYS, Portfolio
from backtesting import BaselTrafficLight
from expected_shortfall import ParametricES
from utils.statistics import z_score
from var import ParametricVaR

# FRTB regulatory liquidity horizons (trading days) by broad risk class.
LIQUIDITY_HORIZONS: dict[str, int] = {
    "general_ir": 10,
    "ig_credit": 20,
    "equity_large": 10,
    "equity_small": 20,
    "fx_major": 10,
    "fx_other": 20,
    "commodity": 20,
}

#: Per-asset liquidity horizon (days) for the built-in sample equity universe.
ASSET_LIQUIDITY_HORIZON: dict[str, int] = {
    "SPY": 10,   # large-cap equity
    "QQQ": 10,   # large-cap equity
    "XLF": 10,   # large-cap equity (financials)
    "VNQ": 10,   # large-cap equity (REIT)
    "IWM": 20,   # small-cap equity
    "EEM": 20,   # emerging-market equity
    "TLT": 10,   # general interest rate
    "HYG": 20,   # high-yield (non-IG) credit
    "USO": 20,   # commodity
    "GLD": 20,   # commodity
}

LH_BASE: int = 10
"""Base liquidity horizon T (days) of the FRTB ES."""

LH_BUCKETS: tuple[int, ...] = (10, 20, 40, 60, 120)
"""FRTB liquidity-horizon buckets (MAR33.12)."""

BASEL_HORIZON: int = 10
"""Holding period (days) of the Basel 2.5 / III IMA VaR."""

SVAR_WINDOW: int = 250
"""Length (trading days) of the stressed-VaR look-back window."""

ROLLING_VAR_WINDOW: int = 60
"""Number of days averaged in the Basel III IMA VaR term."""


class BaselCapitalEngine:
    """Basel III (VaR) and Basel IV / FRTB (ES) regulatory capital.

    Parameters
    ----------
    cov_method : str, default 'sample'
        Covariance estimator passed through to the VaR/ES engines.
    """

    def __init__(self, cov_method: str = "sample") -> None:
        self._cov_method = cov_method
        self._pv = ParametricVaR()
        self._pes = ParametricES()
        self._tl = BaselTrafficLight()

    # --------------------------------------------------------- internal VaR
    def _rolling_parametric_var(
        self,
        portfolio: Portfolio,
        confidence: float,
        window: int = SVAR_WINDOW,
    ) -> pd.Series:
        """Rolling one-day parametric VaR series.

        For each date the portfolio volatility is estimated from the trailing
        ``window`` returns and converted to a one-day VaR
        ``z * sigma * notional``. This is a fast, vectorised proxy for re-running
        the full covariance engine each day and is sufficient for deriving the
        60-day IMA average and the stressed-VaR window.
        """
        port_ret = portfolio.return_series()
        z = z_score(confidence)
        # shift(1): the VaR for day t uses returns up to t - 1 only, so the
        # traffic-light backtest compares it with a P&L it has not seen.
        rolling_vol = port_ret.rolling(window=window, min_periods=window).std(ddof=1).shift(1)
        var = z * rolling_vol * portfolio.notional
        return var.dropna()

    def _stressed_var(self, portfolio: Portfolio, confidence: float) -> float:
        """Stressed VaR from the worst (highest-volatility) 250-day window.

        The portfolio return volatility is computed over every rolling
        ``SVAR_WINDOW``-day window; the maximum volatility identifies the stress
        period. SVaR is the one-day parametric VaR implied by that volatility.
        """
        port_ret = portfolio.return_series()
        rolling_vol = port_ret.rolling(
            window=SVAR_WINDOW, min_periods=SVAR_WINDOW
        ).std(ddof=1).dropna()
        if rolling_vol.empty:
            stressed_vol = float(port_ret.std(ddof=1))
        else:
            stressed_vol = float(rolling_vol.max())
        z = z_score(confidence)
        return float(z * stressed_vol * portfolio.notional)

    # ----------------------------------------------------- Basel III charge
    def var_capital_charge(
        self,
        portfolio: Portfolio,
        var_series: pd.Series | None = None,
        confidence: float = 0.99,
    ) -> dict:
        """Basel 2.5 / III IMA market-risk capital charge.

        ``MRC = max(VaR_{t-1}, mc * avg60(VaR)) + max(sVaR, ms * sVaR)`` on a
        10-day horizon, with one-day figures scaled by sqrt(10).

        Parameters
        ----------
        portfolio : Portfolio
        var_series : pd.Series, optional
            Historical one-day VaR forecasts (positive losses), each made before its
            date. If ``None``, a 60-day rolling parametric VaR is derived internally.
        confidence : float, default 0.99

        Returns
        -------
        dict
            Keys ``var_charge``, ``svar_charge``, ``total`` (all 10-day) and
            ``multiplier`` (``mc = ms`` from the traffic light).
        """
        if var_series is None:
            var_series = self._rolling_parametric_var(
                portfolio, confidence, window=ROLLING_VAR_WINDOW
            )
        var_series = pd.Series(var_series).dropna()

        # Supervisory multiplier from the traffic-light backtest of one-day VaR.
        pnl = portfolio.pnl()
        report = self._tl.backtest_report(var_series, pnl.loc[var_series.index], confidence)
        multiplier = float(report["capital_multiplier"])

        scale = np.sqrt(BASEL_HORIZON)
        var_t = float(var_series.iloc[-1]) * scale
        avg_60 = float(var_series.iloc[-ROLLING_VAR_WINDOW:].mean()) * scale
        var_charge = float(max(var_t, multiplier * avg_60))
        svar = self._stressed_var(portfolio, confidence) * scale
        svar_charge = float(max(svar, multiplier * svar))
        return {
            "var_charge": var_charge,
            "svar_charge": svar_charge,
            "total": var_charge + svar_charge,
            "multiplier": multiplier,
        }

    # ------------------------------------------------------------ FRTB ES
    def _es_10d(self, portfolio: Portfolio, mask: np.ndarray, confidence: float) -> float:
        """Normal 10-day ES of the positions selected by ``mask`` (others set to 0)."""
        cov = portfolio.returns.cov().to_numpy()
        w = np.where(mask, portfolio.weights, 0.0)
        sigma = float(np.sqrt(max(w @ cov @ w, 0.0)) * np.sqrt(LH_BASE))
        alpha = 1.0 - confidence
        return float(stats.norm.pdf(stats.norm.ppf(confidence)) / alpha * sigma * portfolio.notional)

    def frtb_es_capital(self, portfolio: Portfolio, confidence: float = FRTB_CONFIDENCE) -> float:
        """FRTB base ES: normal 97.5% ES of the whole portfolio on the 10-day horizon.

        Returns
        -------
        float
            10-day ES as a positive dollar loss.
        """
        return self._es_10d(portfolio, np.ones(len(portfolio.weights), dtype=bool), confidence)

    def _asset_lh(self, name: str) -> int:
        """Liquidity horizon (days) for a sample asset, defaulting to 20."""
        return ASSET_LIQUIDITY_HORIZON.get(name, 20)

    def frtb_capital(self, portfolio: Portfolio, confidence: float = FRTB_CONFIDENCE) -> dict:
        """FRTB ES with the liquidity-horizon cascade (BCBS MAR33.4).

        ``ES = sqrt( sum_j (ES_T(P, j) * sqrt((LH_j - LH_{j-1}) / T))^2 )`` with
        ``T = 10``. Bucket j = 1 is the whole portfolio; bucket j > 1 shocks only the
        positions whose liquidity horizon is at least ``LH_j``. When every position
        has a 10-day horizon the charge equals the base 10-day ES.

        Returns
        -------
        dict
            Keys ``bucket_es`` (10-day ES of each bucket's subportfolio),
            ``liquidity_horizons`` (per-asset LH) and ``total_charge``.
        """
        lhs = np.array([self._asset_lh(name) for name in portfolio.asset_names], dtype=float)
        bucket_es, terms = [], []
        prev = 0
        for j, lh in enumerate(LH_BUCKETS):
            mask = lhs >= lh if j > 0 else np.ones(len(lhs), dtype=bool)
            es_j = self._es_10d(portfolio, mask, confidence) if mask.any() else 0.0
            width = LH_BASE if j == 0 else lh - prev
            bucket_es.append(es_j)
            terms.append(es_j * np.sqrt(width / LH_BASE))
            prev = lh
        return {
            "bucket_es": np.array(bucket_es),
            "liquidity_horizons": lhs,
            "total_charge": float(np.sqrt(np.sum(np.square(terms)))),
        }

    # ----------------------------------------------------- comparison table
    def capital_comparison(self, portfolio: Portfolio) -> pd.DataFrame:
        """Side-by-side Basel III (VaR) vs Basel IV (ES/FRTB) capital.

        Parameters
        ----------
        portfolio : Portfolio

        Returns
        -------
        pd.DataFrame
            Rows for each framework with a ``Capital($)`` column and the key
            inputs (confidence level and basis).
        """
        basel3 = self.var_capital_charge(portfolio, confidence=0.99)
        frtb_base = self.frtb_es_capital(portfolio, confidence=FRTB_CONFIDENCE)
        frtb = self.frtb_capital(portfolio, confidence=FRTB_CONFIDENCE)
        return pd.DataFrame(
            [
                {
                    "Framework": "Basel III (IMA VaR)",
                    "Basis": "10-day VaR 99% + SVaR",
                    "Confidence": 0.99,
                    "Capital($)": basel3["total"],
                },
                {
                    "Framework": "Basel IV (FRTB ES, base)",
                    "Basis": "ES 97.5% (10d)",
                    "Confidence": FRTB_CONFIDENCE,
                    "Capital($)": frtb_base,
                },
                {
                    "Framework": "Basel IV (FRTB ES, LH-scaled)",
                    "Basis": "ES 97.5%, liquidity-horizon cascade",
                    "Confidence": FRTB_CONFIDENCE,
                    "Capital($)": frtb["total_charge"],
                },
            ]
        )
