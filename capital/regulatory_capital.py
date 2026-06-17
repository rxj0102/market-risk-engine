"""Regulatory market-risk capital: Basel III IMA and Basel IV (FRTB).

Two regimes are implemented:

* **Basel III** Internal Models Approach (IMA), VaR-based. The market-risk
  capital charge (MRC) combines current VaR, an average of the last 60 days'
  VaR scaled by the supervisory multiplier ``mc``, and a stressed-VaR (SVaR)
  add-on::

      MRC = max(VaR_t, mc * avg(VaR_{t-59..t})) + SVaR

* **Basel IV / FRTB** IMA, Expected-Shortfall-based at 97.5% with
  liquidity-horizon scaling across risk classes.

References
----------
BCBS (2019). "Minimum capital requirements for market risk." (FRTB).
BCBS (2009). "Revisions to the Basel II market risk framework." (SVaR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
"""Base liquidity horizon (days) used as the FRTB scaling denominator."""

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
        rolling_vol = port_ret.rolling(window=window, min_periods=window).std(ddof=1)
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
        """Basel III IMA market-risk capital charge.

        ``MRC = max(VaR_t, mc * avg(VaR_last60)) + SVaR``.

        Parameters
        ----------
        portfolio : Portfolio
        var_series : pd.Series, optional
            Historical one-day VaR series (positive losses). If ``None``, a
            60-day rolling parametric VaR is derived internally.
        confidence : float, default 0.99

        Returns
        -------
        dict
            Keys ``var_charge`` (the ``max(...)`` term), ``svar_charge``,
            ``total`` and ``multiplier`` (``mc`` from the traffic light).
        """
        if var_series is None:
            var_series = self._rolling_parametric_var(
                portfolio, confidence, window=ROLLING_VAR_WINDOW
            )
        var_series = pd.Series(var_series).dropna()

        var_t = float(var_series.iloc[-1])
        avg_60 = float(var_series.iloc[-ROLLING_VAR_WINDOW:].mean())

        # Supervisory multiplier from the traffic-light backtest.
        pnl = portfolio.pnl()
        report = self._tl.backtest_report(var_series, pnl.loc[var_series.index], confidence)
        multiplier = float(report["capital_multiplier"])

        var_charge = float(max(var_t, multiplier * avg_60))
        svar_charge = self._stressed_var(portfolio, confidence)
        total = var_charge + svar_charge
        return {
            "var_charge": var_charge,
            "svar_charge": svar_charge,
            "total": total,
            "multiplier": multiplier,
        }

    # ------------------------------------------------------------ FRTB ES
    def frtb_es_capital(self, portfolio: Portfolio, confidence: float = FRTB_CONFIDENCE) -> float:
        """FRTB base Expected Shortfall (97.5%) capital, one-day horizon.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default ``FRTB_CONFIDENCE`` (0.975)

        Returns
        -------
        float
            One-day ES as a positive dollar loss.
        """
        es = self._pes.compute(portfolio, confidence=confidence, distribution="normal")
        return float(es.es)

    def _asset_lh(self, name: str) -> int:
        """Liquidity horizon (days) for a sample asset, defaulting to 20."""
        return ASSET_LIQUIDITY_HORIZON.get(name, 20)

    def frtb_capital(self, portfolio: Portfolio, confidence: float = FRTB_CONFIDENCE) -> dict:
        """FRTB ES capital with liquidity-horizon scaling.

        Each asset receives a standalone (component) ES scaled to its own
        liquidity horizon. The aggregate charge applies the FRTB square-root
        liquidity-horizon adjustment::

            total = sqrt( base_ES^2 + sum_k ES_k^2 * (LH_k / LH_base - 1) )

        with ``LH_base = 10``. The per-asset ``ES_k`` are the Euler component
        ES's that sum to the (correlated, diversified) base portfolio ES, so the
        first term reproduces ``base_ES`` exactly. The second term adds the FRTB
        liquidity-horizon surcharge ``ES_k^2 (LH_k/LH_base - 1) >= 0`` for every
        asset whose horizon exceeds ``LH_base``. This is the equivalent,
        diversification-preserving rearrangement of the FRTB scaling
        ``sqrt(sum_k ES_k^2 LH_k/LH_base)`` that guarantees the LH-scaled charge
        never falls below the base ES.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default ``FRTB_CONFIDENCE`` (0.975)

        Returns
        -------
        dict
            Keys ``es_components`` (per-asset Euler ES, ``np.ndarray``),
            ``liquidity_horizons`` (per-asset LH, ``np.ndarray``) and
            ``total_charge`` (LH-scaled aggregate ES).
        """
        # Euler-allocate the diversified portfolio ES to assets via component-VaR
        # shares so the components sum to the base ES.
        es_total = self.frtb_es_capital(portfolio, confidence=confidence)
        comp_var = self._pv.component_var_analytical(
            portfolio, confidence=confidence, cov_method=self._cov_method
        )
        shares = comp_var / comp_var.sum() if comp_var.sum() else np.full_like(comp_var, np.nan)
        es_components = shares * es_total

        lhs = np.array([self._asset_lh(name) for name in portfolio.asset_names], dtype=float)
        surcharge = np.sum(es_components**2 * (lhs / LH_BASE - 1.0))
        total_charge = float(np.sqrt(es_total**2 + surcharge))
        return {
            "es_components": es_components,
            "liquidity_horizons": lhs,
            "total_charge": total_charge,
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
                    "Basis": "VaR 99% + SVaR",
                    "Confidence": 0.99,
                    "Capital($)": basel3["total"],
                },
                {
                    "Framework": "Basel IV (FRTB ES, base)",
                    "Basis": "ES 97.5% (1d)",
                    "Confidence": FRTB_CONFIDENCE,
                    "Capital($)": frtb_base,
                },
                {
                    "Framework": "Basel IV (FRTB ES, LH-scaled)",
                    "Basis": "ES 97.5% + liquidity horizons",
                    "Confidence": FRTB_CONFIDENCE,
                    "Capital($)": frtb["total_charge"],
                },
            ]
        )
