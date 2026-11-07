"""Parametric (closed-form) Expected Shortfall estimators.

Provides Gaussian and Student-t analytical ES together with a Cornish-Fisher
expansion that incorporates skewness and excess kurtosis.

References
----------
McNeil, A., Frey, R., Embrechts, P. (2015).
    "Quantitative Risk Management." 2nd ed., Princeton University Press.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core import FELLER_THRESHOLD, ESResult, Portfolio
from models import CovarianceEstimator
from utils.statistics import (
    cornish_fisher_quantile,
    normal_pdf_at_quantile,
    z_score,
)


class ParametricES:
    """Closed-form Expected Shortfall under parametric loss distributions."""

    def __init__(self) -> None:
        self._cov = CovarianceEstimator()

    def _sigma_p(self, portfolio: Portfolio, horizon_days: int = 1) -> float:
        """Daily (sqrt-time scaled) portfolio volatility."""
        cov = self._cov.estimate(portfolio.returns, method="sample").to_numpy()
        w = portfolio.weights
        return float(np.sqrt(w @ cov @ w) * np.sqrt(horizon_days))

    def _estimate_nu(self, portfolio: Portfolio) -> float:
        ret = portfolio.return_series().to_numpy()
        exkurt = float(stats.kurtosis(ret))
        nu = 4.0 + 6.0 / exkurt if exkurt > 1e-6 else 30.0
        return float(np.clip(nu, FELLER_THRESHOLD + 0.5, 100.0))

    def compute(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        distribution: str = "normal",
        nu: float | None = None,
        horizon_days: int = 1,
    ) -> ESResult:
        """Parametric Expected Shortfall.

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
        sigma_p = self._sigma_p(portfolio, horizon_days)
        notional = portfolio.notional
        alpha = 1.0 - confidence
        if distribution == "normal":
            z = z_score(confidence)
            es = float(normal_pdf_at_quantile(confidence) / alpha * sigma_p * notional)
            var = float(z * sigma_p * notional)
            method = "Parametric-Normal-ES"
        elif distribution in ("student_t", "t"):
            if nu is None:
                nu = self._estimate_nu(portfolio)
            t_q = float(stats.t.ppf(confidence, nu))
            scale = np.sqrt((nu - 2.0) / nu)
            f_tq = float(stats.t.pdf(t_q, nu))
            es_std = (nu + t_q**2) / (nu - 1.0) * f_tq / alpha
            es = float(es_std * scale * sigma_p * notional)
            var = float(t_q * scale * sigma_p * notional)
            method = "Parametric-t-ES"
        else:
            raise ValueError(f"unknown distribution {distribution!r}")
        return ESResult(
            es=es,
            es_pct=es / notional,
            var=var,
            confidence=confidence,
            method=method,
            tail_scenarios=np.array([]),
            horizon_days=horizon_days,
        )

    def cornish_fisher_es(
        self, portfolio: Portfolio, confidence: float = 0.99, horizon_days: int = 1
    ) -> ESResult:
        """Cornish-Fisher Expected Shortfall (skew/kurtosis adjusted).

        VaR uses the Cornish-Fisher adjusted quantile ``z_CF``. ES is obtained
        by averaging the CF-expanded quantiles across the tail
        ``p in (confidence, 1)``, i.e. ``ES = (1/(1-c)) * integral_c^1 q_CF(p)
        dp * sigma_p * notional``. This tail-average form guarantees
        ``ES >= VaR`` (coherence) and remains stable under strong skew, unlike
        the naive ``phi(z_CF)/(1-c)`` Gaussian density factor which collapses
        once the CF quantile moves far from the Gaussian one (see Notes).

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 1

        Returns
        -------
        ESResult
            ``method='Cornish-Fisher-ES'``.

        Notes
        -----
        The spec's literal ``phi(z_CF)/(1-c)`` factor is only exact for the
        Gaussian quantile; for fat-tailed/skewed data it understates ES (and can
        even yield ES < VaR). The tail-average of CF quantiles is the standard,
        coherent generalisation and is used here.
        """
        ret = portfolio.return_series().to_numpy()
        skew = float(stats.skew(ret))
        exkurt = float(stats.kurtosis(ret))
        z_cf = cornish_fisher_quantile(confidence, skew, exkurt)  # positive
        sigma_p = self._sigma_p(portfolio, horizon_days)
        notional = portfolio.notional
        # Tail-average of CF quantiles over p in (confidence, 1).
        grid = np.linspace(confidence, 1.0, 1000, endpoint=False)[1:]
        q_cf = np.array([cornish_fisher_quantile(p, skew, exkurt) for p in grid])
        es_factor = float(np.mean(q_cf))
        es = float(es_factor * sigma_p * notional)
        var = float(z_cf * sigma_p * notional)
        return ESResult(
            es=es,
            es_pct=es / notional,
            var=var,
            confidence=confidence,
            method="Cornish-Fisher-ES",
            tail_scenarios=np.array([]),
            horizon_days=horizon_days,
        )

    def es_var_ratio(
        self, portfolio: Portfolio, confidence: float = 0.99, distribution: str = "normal"
    ) -> float:
        """Analytical ES/VaR ratio (>= 1).

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        distribution : str, default 'normal'

        Returns
        -------
        float
            For the Normal at 99% this is ``phi(z)/((1-c) z) ≈ 1.146``.
        """
        res = self.compute(portfolio, confidence, distribution=distribution)
        return res.es_var_ratio
