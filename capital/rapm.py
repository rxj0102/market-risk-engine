"""Risk-adjusted performance measurement (RAPM).

Implements RAROC, risk-adjusted Sharpe variants (standard, VaR-based and
ES-based) and a RAROC-maximising capital allocation. All annualised risk
figures use square-root-of-time scaling (``sqrt(TRADING_DAYS)``).

References
----------
Matten, C. (2000). "Managing Bank Capital." 2nd ed., Wiley.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from core import TRADING_DAYS, Portfolio
from utils.statistics import sharpe_ratio, z_score
from var import ParametricVaR
from expected_shortfall import ParametricES

from capital.economic_capital import EconomicCapitalEngine


class RiskAdjustedPerformance:
    """Risk-adjusted performance metrics and capital allocation.

    Parameters
    ----------
    cov_method : str, default 'sample'
        Covariance estimator passed through to the underlying engines.
    """

    def __init__(self, cov_method: str = "sample") -> None:
        self._cov_method = cov_method
        self._pv = ParametricVaR()
        self._pes = ParametricES()
        self._ec = EconomicCapitalEngine(cov_method=cov_method)

    # ------------------------------------------------------------------ RAROC
    def raroc(self, portfolio: Portfolio, expected_annual_return: float, ec: float) -> float:
        """Risk-adjusted return on capital ``RAROC = E[P&L] / EC``.

        Parameters
        ----------
        portfolio : Portfolio
            Present for API symmetry; not used directly when ``ec`` is supplied.
        expected_annual_return : float
            Expected annual P&L in dollars.
        ec : float
            Economic capital (dollars).

        Returns
        -------
        float
            RAROC, or ``nan`` if ``ec`` is zero.
        """
        return float(expected_annual_return / ec) if ec else float("nan")

    # ----------------------------------------------------------- Sharpe ratios
    def all_sharpe_ratios(
        self, portfolio: Portfolio, expected_return: float, rf: float = 0.05
    ) -> dict:
        """Standard, VaR-based and ES-based Sharpe ratios.

        Parameters
        ----------
        portfolio : Portfolio
        expected_return : float
            Expected annual return as a fraction of notional (e.g. 0.08).
        rf : float, default 0.05
            Annual risk-free rate (fraction).

        Returns
        -------
        dict
            Keys ``sharpe`` (standard, annualised from the realised return
            series), ``var_sharpe`` and ``es_sharpe``. The VaR/ES Sharpe ratios
            use the annual (sqrt-time scaled) 99% VaR/ES as the denominator,
            both expressed as a fraction of notional.
        """
        # Standard annualised Sharpe from realised returns.
        sharpe = sharpe_ratio(portfolio.return_series(), rf=rf, periods=TRADING_DAYS)

        ann_scale = np.sqrt(TRADING_DAYS)
        var99 = self._pv.compute(
            portfolio, confidence=0.99, horizon_days=1, cov_method=self._cov_method
        )
        es99 = self._pes.compute(portfolio, confidence=0.99, distribution="normal")
        var_pct_annual = var99.var_pct * ann_scale
        es_pct_annual = es99.es_pct * ann_scale

        excess = expected_return - rf
        var_sharpe = float(excess / var_pct_annual) if var_pct_annual else float("nan")
        es_sharpe = float(excess / es_pct_annual) if es_pct_annual else float("nan")
        return {"sharpe": float(sharpe), "var_sharpe": var_sharpe, "es_sharpe": es_sharpe}

    # ------------------------------------------------------- marginal RAROC
    def marginal_raroc(
        self, portfolio: Portfolio, expected_returns: np.ndarray, confidence: float = 0.999
    ) -> np.ndarray:
        """Per-asset RAROC using component economic capital.

        ``RAROC_i = E[r_i] * w_i * notional / component_EC_i``.

        Parameters
        ----------
        portfolio : Portfolio
        expected_returns : np.ndarray
            Per-asset expected annual returns (fractions).
        confidence : float, default 0.999

        Returns
        -------
        np.ndarray
            RAROC per asset.
        """
        expected_returns = np.asarray(expected_returns, dtype=float)
        ec_df = self._ec.ec_by_asset(portfolio, confidence=confidence)
        comp_ec = ec_df["EC($)"].to_numpy()
        pnl_i = expected_returns * portfolio.weights * portfolio.notional
        with np.errstate(divide="ignore", invalid="ignore"):
            raroc = np.where(comp_ec != 0, pnl_i / comp_ec, np.nan)
        return raroc

    # ----------------------------------------------------- optimal allocation
    def optimal_allocation(
        self,
        portfolio: Portfolio,
        expected_returns: np.ndarray,
        target_return: float,
        confidence: float = 0.999,
    ) -> np.ndarray:
        """RAROC-maximising long-only weights subject to a return floor.

        Maximises portfolio RAROC ``= E[P&L] / EC`` subject to
        ``sum(w) = 1``, ``w >= 0`` and portfolio expected return
        ``>= target_return``.

        Parameters
        ----------
        portfolio : Portfolio
        expected_returns : np.ndarray
            Per-asset expected annual returns (fractions).
        target_return : float
            Minimum acceptable portfolio expected return (fraction).
        confidence : float, default 0.999

        Returns
        -------
        np.ndarray
            Optimal weights (sum to 1, non-negative).
        """
        expected_returns = np.asarray(expected_returns, dtype=float)
        cov = self._pv.covariance_matrix(portfolio, method=self._cov_method).to_numpy()
        z = z_score(confidence)
        scale = np.sqrt(TRADING_DAYS)
        notional = portfolio.notional
        n = portfolio.n_assets

        def neg_raroc(w: np.ndarray) -> float:
            sigma_p = float(np.sqrt(w @ cov @ w))
            ec = z * sigma_p * scale * notional
            if ec <= 0:
                return 0.0
            exp_pnl = float(expected_returns @ w) * notional
            return -exp_pnl / ec

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: float(expected_returns @ w) - target_return},
        ]
        bounds = [(0.0, 1.0)] * n
        x0 = np.full(n, 1.0 / n)
        result = minimize(
            neg_raroc,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        w_opt = np.asarray(result.x, dtype=float)
        w_opt = np.clip(w_opt, 0.0, None)
        total = w_opt.sum()
        return w_opt / total if total else x0
