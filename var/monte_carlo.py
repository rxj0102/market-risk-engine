"""Monte-Carlo Value-at-Risk estimators.

Simulates the joint return distribution (Gaussian, Student-t or via an
empirical Gaussian copula), revalues the linear portfolio and reads the loss
quantile.  Includes a path-dependent GBM variant, antithetic variance
reduction and a convergence study.

References
----------
Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering." Springer.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from core import DEFAULT_SEED, FELLER_THRESHOLD, Portfolio, VaRResult
from models import CopulaModel, CovarianceEstimator
from scipy import stats


class MonteCarloVaR:
    """Monte-Carlo Value-at-Risk via simulated joint returns."""

    def __init__(self) -> None:
        self._cov = CovarianceEstimator()

    def _cholesky(self, portfolio: Portfolio, cov_method: str = "sample") -> np.ndarray:
        """Lower-triangular Cholesky factor of the daily covariance matrix."""
        cov = self._cov.estimate(portfolio.returns, method=cov_method).to_numpy()
        # Symmetrise and add tiny jitter for numerical PSD safety.
        cov = 0.5 * (cov + cov.T)
        try:
            return np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            eps = 1e-12 * np.trace(cov) / cov.shape[0]
            return np.linalg.cholesky(cov + eps * np.eye(cov.shape[0]))

    def _estimate_nu(self, portfolio: Portfolio) -> float:
        ret = portfolio.return_series().to_numpy()
        exkurt = float(stats.kurtosis(ret))
        nu = 4.0 + 6.0 / exkurt if exkurt > 1e-6 else 30.0
        return float(np.clip(nu, FELLER_THRESHOLD + 0.5, 100.0))

    def compute(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizon_days: int = 1,
        n_paths: int = 100_000,
        distribution: str = "normal",
        seed: int = DEFAULT_SEED,
        nu: float | None = None,
    ) -> VaRResult:
        """Monte-Carlo VaR.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 1
            Applied via ``sqrt(horizon_days)`` scaling of the 1-day returns.
        n_paths : int, default 100000
            Number of simulated scenarios.
        distribution : str, default 'normal'
            ``'normal'``, ``'student_t'`` or ``'historical_copula'``.
        seed : int, default 42
            RNG seed for reproducibility.
        nu : float, optional
            Degrees of freedom for the Student-t case.

        Returns
        -------
        VaRResult
            ``method='MonteCarlo-{distribution}'``.
        """
        start = time.perf_counter()
        w = portfolio.weights
        mean = portfolio.returns.to_numpy().mean(axis=0)
        rng = np.random.default_rng(seed)
        n = portfolio.n_assets

        if distribution == "normal":
            chol = self._cholesky(portfolio)
            z = rng.standard_normal((n_paths, n))
            ret = mean + z @ chol.T
        elif distribution in ("student_t", "t"):
            if nu is None:
                nu = self._estimate_nu(portfolio)
            chol = self._cholesky(portfolio)
            z = rng.standard_normal((n_paths, n))
            chi = rng.chisquare(nu, size=n_paths)
            # Standardise so the t-draws have unit variance per the covariance.
            t_draw = z / np.sqrt(chi / nu)[:, None] * np.sqrt((nu - 2.0) / nu)
            ret = mean + t_draw @ chol.T
        elif distribution == "historical_copula":
            copula = CopulaModel()
            corr, marginals = copula.fit_gaussian(portfolio.returns)
            ret = copula.simulate_gaussian(n_paths, corr, marginals, seed=seed)
        else:
            raise ValueError(f"unknown distribution {distribution!r}")

        pnl = ret @ w * portfolio.notional * np.sqrt(horizon_days)
        var = float(-np.quantile(pnl, 1.0 - confidence))
        return VaRResult(
            var=var,
            var_pct=var / portfolio.notional,
            confidence=confidence,
            horizon_days=horizon_days,
            method=f"MonteCarlo-{distribution}",
            n_scenarios=n_paths,
            elapsed_seconds=time.perf_counter() - start,
        )

    def compute_path_dependent(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizon_days: int = 10,
        n_paths: int = 50_000,
        n_steps: int = 10,
        seed: int = DEFAULT_SEED,
    ) -> VaRResult:
        """Path-dependent VaR via simulated GBM price paths.

        Each asset follows geometric Brownian motion calibrated to its sample
        daily drift and covariance; the terminal portfolio value determines the
        loss.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 10
        n_paths : int, default 50000
        n_steps : int, default 10
            Number of time steps over the horizon.
        seed : int, default 42

        Returns
        -------
        VaRResult
            ``method='MonteCarlo-GBM-path'``.
        """
        start = time.perf_counter()
        w = portfolio.weights
        rets = portfolio.returns.to_numpy()
        mu = rets.mean(axis=0)
        chol = self._cholesky(portfolio)
        cov_diag = np.diag(chol @ chol.T)
        rng = np.random.default_rng(seed)
        n = portfolio.n_assets
        dt = horizon_days / n_steps

        # Log-price increments: drift (mu - 0.5 sigma^2) dt + sqrt(dt) L z.
        drift = (mu - 0.5 * cov_diag) * dt
        log_price = np.zeros((n_paths, n))
        for _ in range(n_steps):
            z = rng.standard_normal((n_paths, n))
            log_price += drift + np.sqrt(dt) * (z @ chol.T)
        gross = np.exp(log_price)  # terminal price relatives
        port_value = gross @ w  # weighted terminal relative value
        pnl = (port_value - 1.0) * portfolio.notional
        var = float(-np.quantile(pnl, 1.0 - confidence))
        return VaRResult(
            var=var,
            var_pct=var / portfolio.notional,
            confidence=confidence,
            horizon_days=horizon_days,
            method="MonteCarlo-GBM-path",
            n_scenarios=n_paths,
            elapsed_seconds=time.perf_counter() - start,
        )

    def compute_with_vr(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        horizon_days: int = 1,
        n_paths: int = 100_000,
        method: str = "antithetic",
        seed: int = DEFAULT_SEED,
    ) -> VaRResult:
        """Monte-Carlo VaR with variance reduction.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        horizon_days : int, default 1
        n_paths : int, default 100000
            Total number of scenarios (half antithetic when applicable).
        method : str, default 'antithetic'
            Currently ``'antithetic'`` (simulate ``Z`` and ``-Z``).
        seed : int, default 42

        Returns
        -------
        VaRResult
            ``method='MonteCarlo-antithetic'``.
        """
        start = time.perf_counter()
        w = portfolio.weights
        mean = portfolio.returns.to_numpy().mean(axis=0)
        chol = self._cholesky(portfolio)
        rng = np.random.default_rng(seed)
        n = portfolio.n_assets
        if method == "antithetic":
            half = n_paths // 2
            z = rng.standard_normal((half, n))
            z_full = np.vstack([z, -z])
        else:
            z_full = rng.standard_normal((n_paths, n))
        ret = mean + z_full @ chol.T
        pnl = ret @ w * portfolio.notional * np.sqrt(horizon_days)
        var = float(-np.quantile(pnl, 1.0 - confidence))
        return VaRResult(
            var=var,
            var_pct=var / portfolio.notional,
            confidence=confidence,
            horizon_days=horizon_days,
            method=f"MonteCarlo-{method}",
            n_scenarios=len(pnl),
            elapsed_seconds=time.perf_counter() - start,
        )

    def convergence_analysis(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        n_paths_list: list[int] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> pd.DataFrame:
        """Study VaR estimate vs. number of simulated paths.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        n_paths_list : list of int, optional
            Defaults to ``[1000, 5000, 10000, 50000, 100000, 500000]``.
        seed : int, default 42

        Returns
        -------
        pd.DataFrame
            Columns ``n_paths``, ``var``, ``var_pct``, ``elapsed_seconds``.
        """
        if n_paths_list is None:
            n_paths_list = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
        rows = []
        for n in n_paths_list:
            res = self.compute(portfolio, confidence=confidence, n_paths=n, seed=seed)
            rows.append(
                {
                    "n_paths": n,
                    "var": res.var,
                    "var_pct": res.var_pct,
                    "elapsed_seconds": res.elapsed_seconds,
                }
            )
        return pd.DataFrame(rows).set_index("n_paths")
