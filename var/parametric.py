"""Parametric (variance-covariance) Value-at-Risk estimators.

Implements the delta-normal (RiskMetrics) VaR, a Student-t variant for
fat tails, analytical Expected Shortfall and analytical component-VaR risk
decomposition.

References
----------
J.P. Morgan/Reuters. "RiskMetrics Technical Document." 4th ed., 1996.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from core import FELLER_THRESHOLD, TRADING_DAYS, ESResult, Portfolio, VaRResult
from models import CovarianceEstimator
from utils.statistics import normal_pdf_at_quantile, z_score


class ParametricVaR:
    """Delta-normal / variance-covariance Value-at-Risk.

    The portfolio is treated as linear in its assets; the loss is Gaussian
    (or Student-t) with volatility ``sigma_p = sqrt(w' Sigma w)``.
    """

    def __init__(self) -> None:
        self._cov = CovarianceEstimator()

    # ------------------------------------------------------------------ utils
    def _portfolio_sigma_daily(
        self, portfolio: Portfolio, cov_method: str = "sample", **kwargs
    ) -> float:
        """Daily portfolio volatility ``sqrt(w' Sigma w)``."""
        cov = self._cov.estimate(portfolio.returns, method=cov_method, **kwargs)
        w = portfolio.weights
        return float(np.sqrt(w @ cov.to_numpy() @ w))

    # --------------------------------------------------------------- normal
    def compute(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizon_days: int = 1,
        cov_method: str = "sample",
        **kwargs,
    ) -> VaRResult:
        """Delta-normal VaR.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 1
            Scaled by ``sqrt(horizon_days)``.
        cov_method : str, default 'sample'
            One of ``{'sample', 'ewma', 'dcc', 'ledoit_wolf'}``.

        Returns
        -------
        VaRResult
            ``VaR = z * sigma_p * sqrt(h) * notional``.
        """
        start = time.perf_counter()
        sigma_p = self._portfolio_sigma_daily(portfolio, cov_method, **kwargs)
        z = z_score(confidence)
        var = float(z * sigma_p * np.sqrt(horizon_days) * portfolio.notional)
        method = "Parametric-Normal" if cov_method == "sample" else f"Parametric-{cov_method}"
        return VaRResult(
            var=var,
            var_pct=var / portfolio.notional,
            confidence=confidence,
            horizon_days=horizon_days,
            method=method,
            n_scenarios=len(portfolio.returns),
            elapsed_seconds=time.perf_counter() - start,
        )

    # -------------------------------------------------------------- student-t
    def _estimate_nu(self, portfolio: Portfolio) -> float:
        """Estimate Student-t dof from excess kurtosis of portfolio returns."""
        ret = portfolio.return_series().to_numpy()
        exkurt = float(stats.kurtosis(ret))
        nu = 4.0 + 6.0 / exkurt if exkurt > 1e-6 else 30.0
        return float(np.clip(nu, FELLER_THRESHOLD + 0.5, 100.0))

    def compute_student_t(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizon_days: int = 1,
        nu: float | None = None,
        cov_method: str = "sample",
    ) -> VaRResult:
        """Student-t parametric VaR (fat-tailed).

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 1
        nu : float, optional
            Degrees of freedom. If None, estimated from excess kurtosis
            (``nu = 4 + 6/excess_kurt``, clamped ``> FELLER_THRESHOLD``).
        cov_method : str, default 'sample'

        Returns
        -------
        VaRResult
            ``method='Parametric-t'``. The t-quantile is variance-standardised
            by ``sqrt((nu-2)/nu)`` so ``sigma_p`` keeps its usual meaning.
        """
        start = time.perf_counter()
        if nu is None:
            nu = self._estimate_nu(portfolio)
        sigma_p = self._portfolio_sigma_daily(portfolio, cov_method)
        t_q = float(stats.t.ppf(confidence, nu))
        scale = np.sqrt((nu - 2.0) / nu)
        var = float(t_q * sigma_p * scale * np.sqrt(horizon_days) * portfolio.notional)
        return VaRResult(
            var=var,
            var_pct=var / portfolio.notional,
            confidence=confidence,
            horizon_days=horizon_days,
            method="Parametric-t",
            n_scenarios=len(portfolio.returns),
            elapsed_seconds=time.perf_counter() - start,
        )

    # ----------------------------------------------------------------- ES
    def compute_es(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        distribution: str = "normal",
        nu: float | None = None,
        horizon_days: int = 1,
    ) -> ESResult:
        """Analytical Expected Shortfall.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        distribution : str, default 'normal'
            ``'normal'`` or ``'student_t'``.
        nu : float, optional
            Degrees of freedom for the t-distribution.
        horizon_days : int, default 1

        Returns
        -------
        ESResult

        Notes
        -----
        Normal: ``ES = phi(z)/(1-c) * sigma_p * notional``.
        Student-t: ``ES = (nu + t_q^2)/(nu - 1) * f_t(t_q)/(1-c) *
        sqrt((nu-2)/nu) * sigma_p * notional``.
        """
        sigma_p = self._portfolio_sigma_daily(portfolio, "sample") * np.sqrt(horizon_days)
        alpha = 1.0 - confidence
        if distribution == "normal":
            z = z_score(confidence)
            es = float(normal_pdf_at_quantile(confidence) / alpha * sigma_p * portfolio.notional)
            var = float(z * sigma_p * portfolio.notional)
            method = "Parametric-Normal-ES"
        elif distribution in ("student_t", "t"):
            if nu is None:
                nu = self._estimate_nu(portfolio)
            t_q = float(stats.t.ppf(confidence, nu))
            scale = np.sqrt((nu - 2.0) / nu)
            f_tq = float(stats.t.pdf(t_q, nu))
            es_std = (nu + t_q**2) / (nu - 1.0) * f_tq / alpha
            es = float(es_std * scale * sigma_p * portfolio.notional)
            var = float(t_q * scale * sigma_p * portfolio.notional)
            method = "Parametric-t-ES"
        else:
            raise ValueError(f"unknown distribution {distribution!r}")
        return ESResult(
            es=es,
            es_pct=es / portfolio.notional,
            var=var,
            confidence=confidence,
            method=method,
            tail_scenarios=np.array([]),
            horizon_days=horizon_days,
        )

    # ------------------------------------------------------------ diagnostics
    def covariance_matrix(
        self, portfolio: Portfolio, method: str = "sample", **kwargs
    ) -> pd.DataFrame:
        """Estimated (daily) asset covariance matrix."""
        return self._cov.estimate(portfolio.returns, method=method, **kwargs)

    def portfolio_vol(self, portfolio: Portfolio, cov_method: str = "sample") -> float:
        """Annualised portfolio volatility."""
        return self._portfolio_sigma_daily(portfolio, cov_method) * np.sqrt(TRADING_DAYS)

    def component_var_analytical(
        self, portfolio: Portfolio, confidence: float = 0.99, cov_method: str = "sample"
    ) -> np.ndarray:
        """Component VaR per asset (sums to total VaR by Euler allocation).

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        cov_method : str, default 'sample'

        Returns
        -------
        np.ndarray
            ``CVaR_i = w_i (Sigma w)_i / sigma_p * z * notional``.
        """
        cov = self._cov.estimate(portfolio.returns, method=cov_method).to_numpy()
        w = portfolio.weights
        sigma_p = float(np.sqrt(w @ cov @ w))
        z = z_score(confidence)
        marginal = cov @ w / sigma_p
        return w * marginal * z * portfolio.notional

    def sensitivity_to_correlation(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        rho_range: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """VaR sensitivity to a uniform pairwise-correlation shift.

        Off-diagonal correlations are set to a common ``rho`` while volatilities
        are held at their sample values.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        rho_range : np.ndarray, optional
            Correlation grid. Defaults to ``linspace(0, 0.95, 20)``.

        Returns
        -------
        pd.DataFrame
            Columns ``rho`` and ``var``.
        """
        if rho_range is None:
            rho_range = np.linspace(0.0, 0.95, 20)
        cov = self._cov.estimate(portfolio.returns, method="sample").to_numpy()
        vol = np.sqrt(np.diag(cov))
        w = portfolio.weights
        z = z_score(confidence)
        n = len(vol)
        rows = []
        for rho in rho_range:
            corr = np.full((n, n), rho)
            np.fill_diagonal(corr, 1.0)
            cov_r = corr * np.outer(vol, vol)
            sigma_p = float(np.sqrt(w @ cov_r @ w))
            rows.append({"rho": float(rho), "var": z * sigma_p * portfolio.notional})
        return pd.DataFrame(rows)
