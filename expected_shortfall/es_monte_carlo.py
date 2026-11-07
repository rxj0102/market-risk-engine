"""Monte-Carlo Expected Shortfall estimators.

Simulates the joint return distribution and averages the simulated losses
beyond VaR, with optional antithetic variance reduction and a bootstrap
confidence interval for the ES estimate.

References
----------
Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering." Springer.
"""

from __future__ import annotations

import numpy as np

from core import DEFAULT_SEED, FELLER_THRESHOLD, ESResult, Portfolio
from models import CovarianceEstimator
from scipy import stats


class MCExpectedShortfall:
    """Monte-Carlo Expected Shortfall via simulated joint returns."""

    def __init__(self) -> None:
        self._cov = CovarianceEstimator()

    def _cholesky(self, portfolio: Portfolio) -> np.ndarray:
        cov = self._cov.estimate(portfolio.returns, method="sample").to_numpy()
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

    def _simulate_pnl(
        self,
        portfolio: Portfolio,
        n_paths: int,
        distribution: str,
        antithetic: bool,
        seed: int,
        horizon_days: int,
        nu: float | None,
    ) -> np.ndarray:
        """Simulate the portfolio P&L vector."""
        w = portfolio.weights
        mean = portfolio.returns.to_numpy().mean(axis=0)
        chol = self._cholesky(portfolio)
        rng = np.random.default_rng(seed)
        n = portfolio.n_assets

        if antithetic:
            half = n_paths // 2
            z = rng.standard_normal((half, n))
            z = np.vstack([z, -z])
        else:
            z = rng.standard_normal((n_paths, n))

        if distribution == "normal":
            ret = mean + z @ chol.T
        elif distribution in ("student_t", "t"):
            if nu is None:
                nu = self._estimate_nu(portfolio)
            chi = rng.chisquare(nu, size=z.shape[0])
            t_draw = z / np.sqrt(chi / nu)[:, None] * np.sqrt((nu - 2.0) / nu)
            ret = mean + t_draw @ chol.T
        else:
            raise ValueError(f"unknown distribution {distribution!r}")
        return ret @ w * portfolio.notional * np.sqrt(horizon_days)

    def compute(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        n_paths: int = 500_000,
        distribution: str = "normal",
        antithetic: bool = True,
        seed: int = DEFAULT_SEED,
        horizon_days: int = 1,
        nu: float | None = None,
    ) -> ESResult:
        """Monte-Carlo Expected Shortfall.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        n_paths : int, default 500000
        distribution : str, default 'normal'
            ``'normal'`` or ``'student_t'``.
        antithetic : bool, default True
            Use antithetic variates for variance reduction.
        seed : int, default 42
        horizon_days : int, default 1
        nu : float, optional

        Returns
        -------
        ESResult
            ``method='MonteCarlo-ES'``.
        """
        pnl = self._simulate_pnl(
            portfolio, n_paths, distribution, antithetic, seed, horizon_days, nu
        )
        var = -np.quantile(pnl, 1.0 - confidence)
        tail = pnl[pnl <= -var]
        if tail.size == 0:
            tail = np.array([pnl.min()])
        es = float(-tail.mean())
        return ESResult(
            es=es,
            es_pct=es / portfolio.notional,
            var=float(var),
            confidence=confidence,
            method=f"MonteCarlo-{distribution}-ES",
            tail_scenarios=tail,
            horizon_days=horizon_days,
        )

    def es_confidence_interval(
        self,
        portfolio: Portfolio,
        confidence: float = 0.99,
        n_paths: int = 100_000,
        n_bootstrap: int = 1_000,
        distribution: str = "normal",
        ci_level: float = 0.95,
        seed: int = DEFAULT_SEED,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for the Monte-Carlo ES estimate.

        Parameters
        ----------
        portfolio : Portfolio
        confidence : float, default 0.99
        n_paths : int, default 100000
        n_bootstrap : int, default 1000
            Number of bootstrap resamples of the simulated P&L.
        distribution : str, default 'normal'
        ci_level : float, default 0.95
            Two-sided coverage of the returned interval.
        seed : int, default 42

        Returns
        -------
        tuple[float, float]
            ``(es_lower, es_upper)`` in positive dollars.
        """
        pnl = self._simulate_pnl(
            portfolio, n_paths, distribution, True, seed, 1, None
        )
        rng = np.random.default_rng(seed + 1)
        n = len(pnl)
        es_samples = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            sample = pnl[idx]
            var = -np.quantile(sample, 1.0 - confidence)
            tail = sample[sample <= -var]
            es_samples[b] = -tail.mean() if tail.size else -sample.min()
        lo = (1.0 - ci_level) / 2.0
        hi = 1.0 - lo
        return float(np.quantile(es_samples, lo)), float(np.quantile(es_samples, hi))
