"""Sensitivity / scenario-grid analysis.

Computes how portfolio P&L (or VaR) responds to shocks in individual risk
factors: equity level, rates, volatility, FX and credit spreads.  Produces
tornado-chart and two-factor-grid data for reporting.

References
----------
Jorion, P. (2007). "Value at Risk: The New Benchmark for Managing Financial
    Risk." 3rd ed., McGraw-Hill (Greeks and scenario grids).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import BASEL_CONFIDENCE, Portfolio
from utils.statistics import z_score

# Asset groupings used to map factor shocks onto the ETF universe.
_EQUITY_ASSETS = ("SPY", "QQQ", "IWM", "VNQ", "XLF", "EEM")
# Assumed effective durations (years) for the rate-sensitive sleeves.
_DURATION = {"TLT": 17.0, "HYG": 4.0}
# Assumed spread durations (years) for credit-sensitive sleeves.
_SPREAD_DURATION = {"HYG": 4.0}


class SensitivityAnalyzer:
    """Single- and two-factor sensitivity analysis for a portfolio.

    Parameters
    ----------
    confidence : float
        Confidence level for VaR-based sensitivities (default Basel 99%).
    """

    def __init__(self, confidence: float = BASEL_CONFIDENCE) -> None:
        self.confidence = confidence

    # --------------------------------------------------------------- helpers
    def _asset_idx(self, portfolio: Portfolio, asset: str) -> int | None:
        """Index of ``asset`` in the portfolio, or ``None`` if absent."""
        try:
            return portfolio.asset_names.index(asset)
        except ValueError:
            return None

    def _pnl_from_shock(self, portfolio: Portfolio, shock_vec: np.ndarray) -> float:
        """Portfolio P&L for an asset-level return-shock vector."""
        return float((portfolio.weights * shock_vec).sum() * portfolio.notional)

    # ------------------------------------------------------- equity_sensitivity
    def equity_sensitivity(
        self, portfolio: Portfolio, shocks: np.ndarray | None = None
    ) -> pd.DataFrame:
        """P&L profile under a uniform shock to equity-like assets.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.
        shocks : np.ndarray, optional
            Grid of equity return shocks.  Defaults to ``-50%..+50%`` by 1%.

        Returns
        -------
        pd.DataFrame
            Columns ``shock`` and ``pnl``.
        """
        if shocks is None:
            shocks = np.arange(-0.5, 0.51, 0.01)
        is_equity = np.array(
            [a in _EQUITY_ASSETS for a in portfolio.asset_names], dtype=float
        )
        # Vectorised: pnl(shock) = shock * notional * sum(w_i * 1_equity_i).
        equity_exposure = float((portfolio.weights * is_equity).sum())
        pnl = shocks * portfolio.notional * equity_exposure
        return pd.DataFrame({"shock": shocks, "pnl": pnl})

    # --------------------------------------------------------- rate_sensitivity
    def rate_sensitivity(
        self, portfolio: Portfolio, rate_shocks_bps: np.ndarray | None = None
    ) -> pd.DataFrame:
        """P&L profile under parallel interest-rate shocks.

        Maps a basis-point rate move to bond-sleeve returns via assumed
        durations: ``r = -duration * delta_y``.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.
        rate_shocks_bps : np.ndarray, optional
            Grid of rate shocks in basis points.  Defaults to
            ``-300..+300`` by 25 bp.

        Returns
        -------
        pd.DataFrame
            Columns ``rate_shock_bps`` and ``pnl``.
        """
        if rate_shocks_bps is None:
            rate_shocks_bps = np.arange(-300, 301, 25)
        dy = rate_shocks_bps / 10_000.0  # bp -> decimal yield change

        # P&L slope per unit yield change: -sum(w_i * duration_i) * notional.
        slope = 0.0
        for asset, dur in _DURATION.items():
            idx = self._asset_idx(portfolio, asset)
            if idx is not None:
                slope += portfolio.weights[idx] * (-dur)
        pnl = dy * slope * portfolio.notional
        return pd.DataFrame({"rate_shock_bps": rate_shocks_bps, "pnl": pnl})

    # ---------------------------------------------------------- vol_sensitivity
    def vol_sensitivity(
        self, portfolio: Portfolio, vol_shocks: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Parametric 99% VaR under multiplicative volatility shocks.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.
        vol_shocks : np.ndarray, optional
            Relative changes to portfolio volatility (e.g. ``0.10`` = +10%).
            Defaults to ``-20%..+30%`` by 1%.

        Returns
        -------
        pd.DataFrame
            Columns ``vol_shock``, ``var`` (positive dollar VaR), and
            ``var_change`` (vs. the base case).
        """
        if vol_shocks is None:
            vol_shocks = np.arange(-0.2, 0.31, 0.01)
        ret = portfolio.return_series()
        base_vol = float(ret.std(ddof=1))
        z = z_score(self.confidence)
        base_var = z * base_vol * portfolio.notional
        stressed_var = z * base_vol * (1.0 + vol_shocks) * portfolio.notional
        return pd.DataFrame(
            {
                "vol_shock": vol_shocks,
                "var": stressed_var,
                "var_change": stressed_var - base_var,
            }
        )

    # ----------------------------------------------------------- fx_sensitivity
    def fx_sensitivity(
        self, portfolio: Portfolio, currency: str, shocks: np.ndarray | None = None
    ) -> pd.DataFrame:
        """P&L profile under shocks to a foreign-currency leg.

        Assets whose name contains the ``currency`` code (e.g. an FX pair
        leg) are treated as exposed; for the equity ETF universe ``EEM`` is
        used as a proxy USD/EM-FX sensitivity when ``currency`` is non-USD.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.
        currency : str
            Currency code of the shock (e.g. ``'EUR'``).
        shocks : np.ndarray, optional
            Grid of FX return shocks.  Defaults to ``-20%..+20%`` by 1%.

        Returns
        -------
        pd.DataFrame
            Columns ``fx_shock`` and ``pnl``.
        """
        if shocks is None:
            shocks = np.arange(-0.2, 0.21, 0.01)
        cur = currency.upper()
        is_exposed = np.array(
            [
                (cur in a.upper()) or (cur != "USD" and a == "EEM")
                for a in portfolio.asset_names
            ],
            dtype=float,
        )
        exposure = float((portfolio.weights * is_exposed).sum())
        pnl = shocks * portfolio.notional * exposure
        return pd.DataFrame({"fx_shock": shocks, "pnl": pnl})

    # ------------------------------------------------ credit_spread_sensitivity
    def credit_spread_sensitivity(
        self, portfolio: Portfolio, spread_shocks_bps: np.ndarray | None = None
    ) -> pd.DataFrame:
        """P&L profile under credit-spread widening/tightening.

        Maps a spread move to credit-sleeve returns via spread duration:
        ``r = -spread_duration * delta_spread``.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.
        spread_shocks_bps : np.ndarray, optional
            Grid of spread shocks in basis points.  Defaults to
            ``-200..+200`` by 25 bp.

        Returns
        -------
        pd.DataFrame
            Columns ``spread_shock_bps`` and ``pnl``.
        """
        if spread_shocks_bps is None:
            spread_shocks_bps = np.arange(-200, 201, 25)
        ds = spread_shocks_bps / 10_000.0
        slope = 0.0
        for asset, dur in _SPREAD_DURATION.items():
            idx = self._asset_idx(portfolio, asset)
            if idx is not None:
                slope += portfolio.weights[idx] * (-dur)
        pnl = ds * slope * portfolio.notional
        return pd.DataFrame({"spread_shock_bps": spread_shocks_bps, "pnl": pnl})

    # ------------------------------------------------------ tornado_chart_data
    def tornado_chart_data(self, portfolio: Portfolio) -> pd.DataFrame:
        """P&L at +/-1 standard-deviation shocks per risk factor.

        Each factor is shocked down (``-1 sigma``) and up (``+1 sigma``)
        using realised factor proxies, and results are sorted by absolute
        impact for a tornado chart.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.

        Returns
        -------
        pd.DataFrame
            Columns ``Factor``, ``Down``, ``Up``, ``Impact`` (sorted by
            descending absolute impact).
        """
        returns = portfolio.returns
        notional = portfolio.notional
        w = portfolio.weights
        names = portfolio.asset_names

        def group_pnl(asset_group: tuple[str, ...], sigma_sign: float) -> float:
            """P&L from a +/-1 sigma move of the group's average daily vol."""
            mask = np.array([a in asset_group for a in names], dtype=float)
            present = [a for a in asset_group if a in names]
            if not present:
                return 0.0
            sigma = float(returns[present].std(ddof=1).mean())
            shock = sigma_sign * sigma
            return float((w * mask).sum() * shock * notional)

        factor_groups: dict[str, tuple[str, ...]] = {
            "Equity": _EQUITY_ASSETS,
            "Rates": ("TLT",),
            "Credit": ("HYG",),
            "Commodity": ("USO", "GLD"),
            "EM/FX": ("EEM",),
        }

        rows: list[dict[str, object]] = []
        for factor, group in factor_groups.items():
            down = group_pnl(group, -1.0)
            up = group_pnl(group, +1.0)
            rows.append(
                {
                    "Factor": factor,
                    "Down": down,
                    "Up": up,
                    "Impact": max(abs(down), abs(up)),
                }
            )
        df = pd.DataFrame(rows).sort_values("Impact", ascending=False).reset_index(drop=True)
        return df

    # --------------------------------------------------------- two_factor_grid
    def two_factor_grid(
        self,
        portfolio: Portfolio,
        factor1: str,
        factor2: str,
        grid1: np.ndarray,
        grid2: np.ndarray,
    ) -> pd.DataFrame:
        """2-D P&L matrix over two simultaneous factor shocks.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.
        factor1, factor2 : str
            Factor names; one of ``{'equity', 'rates', 'credit', 'fx',
            'commodity'}``.
        grid1, grid2 : np.ndarray
            Shock grids for ``factor1`` (rows) and ``factor2`` (columns).
            Equity/FX/commodity shocks are return shocks; rates/credit
            shocks are in basis points.

        Returns
        -------
        pd.DataFrame
            P&L matrix indexed by ``grid1`` with columns ``grid2``.
        """
        grid1 = np.asarray(grid1, dtype=float)
        grid2 = np.asarray(grid2, dtype=float)
        f1 = self._factor_pnl_per_unit(portfolio, factor1)
        f2 = self._factor_pnl_per_unit(portfolio, factor2)
        # Outer sum: linear, additive factor responses.
        matrix = np.add.outer(grid1 * f1, grid2 * f2)
        return pd.DataFrame(matrix, index=grid1, columns=grid2)

    def _factor_pnl_per_unit(self, portfolio: Portfolio, factor: str) -> float:
        """P&L per unit shock of a named factor (return units or per-bp)."""
        factor = factor.lower()
        w = portfolio.weights
        notional = portfolio.notional
        names = portfolio.asset_names
        if factor in ("equity", "equities"):
            mask = np.array([a in _EQUITY_ASSETS for a in names], dtype=float)
            return float((w * mask).sum() * notional)
        if factor in ("rates", "rate"):
            slope = 0.0
            for asset, dur in _DURATION.items():
                idx = self._asset_idx(portfolio, asset)
                if idx is not None:
                    slope += w[idx] * (-dur)
            return slope * notional / 10_000.0  # per-bp
        if factor == "credit":
            slope = 0.0
            for asset, dur in _SPREAD_DURATION.items():
                idx = self._asset_idx(portfolio, asset)
                if idx is not None:
                    slope += w[idx] * (-dur)
            return slope * notional / 10_000.0  # per-bp
        if factor == "fx":
            idx = self._asset_idx(portfolio, "EEM")
            return (w[idx] if idx is not None else 0.0) * notional
        if factor in ("commodity", "commodities"):
            mask = np.array([a in ("USO", "GLD") for a in names], dtype=float)
            return float((w * mask).sum() * notional)
        raise ValueError(f"Unknown factor {factor!r}")
