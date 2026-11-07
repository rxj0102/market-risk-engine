"""GARCH(1,1) volatility modelling for FHS, EWMA and vol forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GARCHParams:
    """Fitted GARCH(1,1) parameters.

    Attributes
    ----------
    omega, alpha, beta : float
        ``sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2``.
    log_likelihood : float
        Maximised Gaussian log-likelihood.
    aic : float
        Akaike information criterion.
    """

    omega: float
    alpha: float
    beta: float
    log_likelihood: float = np.nan
    aic: float = np.nan

    @property
    def persistence(self) -> float:
        """Volatility persistence ``alpha + beta`` (< 1 for stationarity)."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """Unconditional variance ``omega / (1 - alpha - beta)``."""
        denom = 1.0 - self.persistence
        return self.omega / denom if denom > 1e-12 else np.nan

    @property
    def long_run_vol(self) -> float:
        """Unconditional (daily) volatility."""
        return float(np.sqrt(self.long_run_variance))


class GARCHModel:
    """Maximum-likelihood GARCH(1,1) estimation and volatility forecasting.

    Notes
    -----
    Returns are demeaned before fitting. The conditional variance recursion is
    seeded with the sample variance. Parameters are estimated by maximising the
    Gaussian quasi-log-likelihood subject to ``omega>0``, ``alpha,beta>=0`` and
    ``alpha+beta<1``.

    References
    ----------
    Bollerslev, T. (1986). "Generalized autoregressive conditional
    heteroskedasticity." *Journal of Econometrics* 31(3), 307-327.
    """

    def __init__(self) -> None:
        self._eps = 1e-8

    # ------------------------------------------------------------------ fit
    def _conditional_var(self, r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
        n = len(r)
        sigma2 = np.empty(n)
        sigma2[0] = np.var(r) if n > 1 else r[0] ** 2 + self._eps
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        return np.maximum(sigma2, self._eps)

    def _neg_loglik(self, theta: np.ndarray, r: np.ndarray) -> float:
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
            return 1e10
        sigma2 = self._conditional_var(r, omega, alpha, beta)
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + r**2 / sigma2)
        return -ll

    def fit(self, returns: pd.Series | np.ndarray, method: str = "MLE") -> GARCHParams:
        """Fit GARCH(1,1) by (quasi) maximum likelihood.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Single-asset return series.
        method : str
            Placeholder for the estimator (only ``'MLE'`` is implemented).

        Returns
        -------
        GARCHParams
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        r = r - r.mean()
        var = np.var(r)

        # Sensible starting point: low alpha, high beta, matching unconditional var.
        x0 = np.array([var * 0.05, 0.08, 0.90])
        bounds = [(1e-12, None), (0.0, 0.999), (0.0, 0.999)]
        cons = {"type": "ineq", "fun": lambda th: 0.999 - th[1] - th[2]}
        res = minimize(
            self._neg_loglik,
            x0,
            args=(r,),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        omega, alpha, beta = res.x
        ll = -res.fun
        aic = 2 * 3 - 2 * ll
        return GARCHParams(omega=omega, alpha=alpha, beta=beta, log_likelihood=ll, aic=aic)

    # -------------------------------------------------------------- forecast
    def conditional_vols(self, returns: pd.Series | np.ndarray, params: GARCHParams) -> np.ndarray:
        """Filtered conditional volatility series ``sigma_t``."""
        r = np.asarray(returns, dtype=float)
        r = r - np.nanmean(r)
        sigma2 = self._conditional_var(r, params.omega, params.alpha, params.beta)
        return np.sqrt(sigma2)

    def forecast_vol(self, returns: pd.Series | np.ndarray, params: GARCHParams, horizon: int = 1) -> float:
        """One-step (or h-step terminal) ahead volatility forecast."""
        r = np.asarray(returns, dtype=float)
        r = r - np.nanmean(r)
        sigma2 = self._conditional_var(r, params.omega, params.alpha, params.beta)
        last_sigma2 = sigma2[-1]
        next_var = params.omega + params.alpha * r[-1] ** 2 + params.beta * last_sigma2
        if horizon == 1:
            return float(np.sqrt(next_var))
        lr = params.long_run_variance
        h_var = lr + (params.persistence) ** (horizon - 1) * (next_var - lr)
        return float(np.sqrt(h_var))

    def multi_step_forecast(
        self, returns: pd.Series | np.ndarray, params: GARCHParams, horizon: int = 10
    ) -> np.ndarray:
        """Volatility term structure ``sigma_{T+1..T+h}``.

        Uses ``E[sigma_{T+h}^2] = sigma_LR^2 + (alpha+beta)^{h-1}(sigma_{T+1}^2 - sigma_LR^2)``.
        """
        r = np.asarray(returns, dtype=float)
        r = r - np.nanmean(r)
        sigma2 = self._conditional_var(r, params.omega, params.alpha, params.beta)
        next_var = params.omega + params.alpha * r[-1] ** 2 + params.beta * sigma2[-1]
        lr = params.long_run_variance
        h = np.arange(1, horizon + 1)
        var_path = lr + params.persistence ** (h - 1) * (next_var - lr)
        return np.sqrt(var_path)

    def standardized_residuals(self, returns: pd.Series | np.ndarray, params: GARCHParams) -> np.ndarray:
        """Standardised residuals ``z_t = r_t / sigma_t`` (for FHS)."""
        r = np.asarray(returns, dtype=float)
        mu = np.nanmean(r)
        sigma = self.conditional_vols(r, params)
        return (r - mu) / sigma
