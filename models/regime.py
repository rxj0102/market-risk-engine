"""Two-regime Markov switching volatility model (EM / Baum-Welch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from core import Portfolio


@dataclass
class RegimeParams:
    """Fitted Markov-switching parameters.

    Attributes
    ----------
    mu : np.ndarray
        Per-regime mean returns.
    sigma : np.ndarray
        Per-regime volatilities (sorted ascending: calm, stressed).
    transition_matrix : np.ndarray
        Row-stochastic transition matrix ``P[i, j] = P(s_t=j | s_{t-1}=i)``.
    smoothed_probabilities : np.ndarray
        Smoothed ``P(s_t = stressed)`` per date.
    log_likelihood : float
        Final EM log-likelihood.
    """

    mu: np.ndarray
    sigma: np.ndarray
    transition_matrix: np.ndarray
    smoothed_probabilities: np.ndarray
    log_likelihood: float = np.nan


class RegimeSwitchingModel:
    """Two-state Gaussian hidden Markov model for volatility regimes.

    Fitted by the EM (Baum-Welch) algorithm. State 0 is the calm (low-vol)
    regime and state 1 the stressed (high-vol) regime; states are ordered by
    volatility after estimation.

    References
    ----------
    Hamilton, J. (1989). "A new approach to the economic analysis of
    nonstationary time series and the business cycle." *Econometrica* 57(2).
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, returns: pd.Series | np.ndarray, n_regimes: int = 2) -> RegimeParams:
        """Fit a two-regime model by EM.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Return series (typically portfolio returns).
        n_regimes : int
            Only ``2`` is supported.
        """
        if n_regimes != 2:
            raise NotImplementedError("Only two-regime estimation is implemented")
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)

        # Initialise: split by volatility tertiles.
        lo, hi = np.quantile(np.abs(r - r.mean()), [0.5, 0.5])
        mu = np.array([r.mean(), r.mean()])
        sigma = np.array([r.std() * 0.7, r.std() * 1.8])
        P = np.array([[0.95, 0.05], [0.10, 0.90]])
        pi = np.array([0.9, 0.1])

        ll_old = -np.inf
        gamma = np.full((n, 2), 0.5)
        for _ in range(self.max_iter):
            # E-step: forward-backward.
            B = np.column_stack([stats.norm.pdf(r, mu[k], sigma[k]) for k in range(2)])
            B = np.maximum(B, 1e-300)
            alpha = np.zeros((n, 2))
            scale = np.zeros(n)
            alpha[0] = pi * B[0]
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0]
            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ P) * B[t]
                scale[t] = alpha[t].sum()
                alpha[t] /= scale[t]
            beta = np.zeros((n, 2))
            beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                beta[t] = (P @ (B[t + 1] * beta[t + 1])) / scale[t + 1]

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            xi = np.zeros((2, 2))
            for t in range(n - 1):
                denom = (alpha[t][:, None] * P * (B[t + 1] * beta[t + 1])[None, :]).sum()
                xi += (alpha[t][:, None] * P * (B[t + 1] * beta[t + 1])[None, :]) / denom

            # M-step.
            pi = gamma[0]
            P = xi / xi.sum(axis=1, keepdims=True)
            for k in range(2):
                w = gamma[:, k]
                mu[k] = (w @ r) / w.sum()
                sigma[k] = np.sqrt((w @ (r - mu[k]) ** 2) / w.sum())
            sigma = np.maximum(sigma, 1e-8)

            ll = np.sum(np.log(scale))
            if abs(ll - ll_old) < self.tol:
                break
            ll_old = ll

        # Order states by volatility (calm first).
        order = np.argsort(sigma)
        mu, sigma = mu[order], sigma[order]
        P = P[np.ix_(order, order)]
        smoothed = gamma[:, order][:, 1]  # P(stressed)
        return RegimeParams(mu=mu, sigma=sigma, transition_matrix=P,
                            smoothed_probabilities=smoothed, log_likelihood=ll_old)

    def current_regime_probability(self, returns: pd.Series | np.ndarray, params: RegimeParams) -> float:
        """Probability the most recent observation is in the stressed regime."""
        return float(params.smoothed_probabilities[-1])

    def regime_conditional_var(
        self, portfolio: Portfolio, params: RegimeParams, confidence: float, regime: int
    ) -> float:
        """VaR (dollar) conditional on being in ``regime`` (0=calm, 1=stressed)."""
        z = stats.norm.ppf(1 - confidence)
        var = -(params.mu[regime] + z * params.sigma[regime]) * portfolio.notional
        return float(max(var, 0.0))

    def expected_var(self, portfolio: Portfolio, params: RegimeParams, confidence: float) -> float:
        """Regime-probability-weighted VaR using current smoothed probabilities."""
        p_stress = self.current_regime_probability(portfolio.return_series(), params)
        var_calm = self.regime_conditional_var(portfolio, params, confidence, 0)
        var_stress = self.regime_conditional_var(portfolio, params, confidence, 1)
        return (1 - p_stress) * var_calm + p_stress * var_stress

    def regime_time_series(self, returns: pd.Series | np.ndarray, params: RegimeParams) -> np.ndarray:
        """Smoothed stressed-regime probability series."""
        return params.smoothed_probabilities
