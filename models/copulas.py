"""Gaussian and Student-t copulas for dependency modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core import DEFAULT_SEED, FELLER_THRESHOLD


class CopulaModel:
    """Fit and simulate Gaussian / Student-t copulas with empirical marginals.

    Notes
    -----
    Marginals are handled empirically (rank transform to uniforms, inverse via
    empirical quantiles). The Gaussian copula has zero tail dependence; the
    t-copula introduces symmetric tail dependence governed by ``nu``.

    References
    ----------
    Embrechts, McNeil & Straumann (2002). "Correlation and dependence in risk
    management." In *Risk Management: Value at Risk and Beyond*.
    """

    @staticmethod
    def _empirical_uniforms(returns: pd.DataFrame) -> np.ndarray:
        """Rank-transform each column to (0,1) uniforms."""
        ranks = returns.rank(method="average").to_numpy()
        n = len(returns)
        return ranks / (n + 1.0)

    # --------------------------------------------------------------- fit
    def fit_gaussian(self, returns: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Fit a Gaussian copula. Returns (correlation matrix, marginal data)."""
        u = self._empirical_uniforms(returns)
        z = stats.norm.ppf(u)
        corr = np.corrcoef(z, rowvar=False)
        return corr, returns

    def fit_t_copula(self, returns: pd.DataFrame) -> tuple[np.ndarray, float, pd.DataFrame]:
        """Fit a Student-t copula. Returns (correlation, nu, marginal data).

        ``nu`` is estimated from the average excess kurtosis of the marginals
        via ``nu = 4 + 6 / kurtosis`` (clamped to ``> FELLER_THRESHOLD``).
        """
        u = self._empirical_uniforms(returns)
        z = stats.norm.ppf(u)
        corr = np.corrcoef(z, rowvar=False)
        exkurt = returns.kurtosis().mean()
        nu = 4.0 + 6.0 / exkurt if exkurt > 1e-6 else 30.0
        nu = float(np.clip(nu, FELLER_THRESHOLD + 0.5, 100.0))
        return corr, nu, returns

    # ---------------------------------------------------------- simulate
    @staticmethod
    def _inverse_marginals(u_sim: np.ndarray, marginals: pd.DataFrame) -> np.ndarray:
        """Map uniform draws back to return space via empirical quantiles."""
        out = np.empty_like(u_sim)
        for j in range(marginals.shape[1]):
            col = marginals.iloc[:, j].to_numpy()
            out[:, j] = np.quantile(col, np.clip(u_sim[:, j], 1e-6, 1 - 1e-6))
        return out

    def simulate_gaussian(
        self,
        n_paths: int,
        correlation_matrix: np.ndarray,
        marginals: pd.DataFrame,
        seed: int = DEFAULT_SEED,
    ) -> np.ndarray:
        """Simulate returns from a Gaussian copula with empirical marginals."""
        rng = np.random.default_rng(seed)
        chol = np.linalg.cholesky(self._nearest_psd(correlation_matrix))
        z = rng.standard_normal((n_paths, correlation_matrix.shape[0])) @ chol.T
        u = stats.norm.cdf(z)
        return self._inverse_marginals(u, marginals)

    def simulate_t_copula(
        self,
        n_paths: int,
        correlation_matrix: np.ndarray,
        nu: float,
        marginals: pd.DataFrame,
        seed: int = DEFAULT_SEED,
    ) -> np.ndarray:
        """Simulate returns from a Student-t copula with empirical marginals."""
        rng = np.random.default_rng(seed)
        k = correlation_matrix.shape[0]
        chol = np.linalg.cholesky(self._nearest_psd(correlation_matrix))
        z = rng.standard_normal((n_paths, k)) @ chol.T
        chi = rng.chisquare(nu, size=n_paths)
        t = z / np.sqrt(chi / nu)[:, None]
        u = stats.t.cdf(t, df=nu)
        return self._inverse_marginals(u, marginals)

    # -------------------------------------------------------- tail dep.
    @staticmethod
    def tail_dependence(correlation_matrix: np.ndarray, nu: float) -> dict[str, np.ndarray]:
        """Upper/lower tail-dependence coefficients of the t-copula.

        ``lambda = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))`` (symmetric).
        """
        rho = np.clip(correlation_matrix, -0.999, 0.999)
        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lam = 2 * stats.t.cdf(arg, df=nu + 1)
        return {"upper": lam, "lower": lam}

    @staticmethod
    def _nearest_psd(mat: np.ndarray) -> np.ndarray:
        """Project a symmetric matrix to the nearest PSD matrix (eigenvalue clip)."""
        sym = (mat + mat.T) / 2.0
        vals, vecs = np.linalg.eigh(sym)
        vals = np.clip(vals, 1e-10, None)
        psd = (vecs * vals) @ vecs.T
        d = np.sqrt(np.diag(psd))
        return psd / np.outer(d, d)
