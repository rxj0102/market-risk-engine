"""Reverse stress testing.

Whereas forward stress testing asks "how bad is scenario X?", reverse stress
testing inverts the question: "what scenario produces a loss of size L?".  We
provide a minimum-norm (most plausible) factor shock that achieves a target
loss, a Monte-Carlo search for breaking scenarios, and analytic break
probabilities.

References
----------
Basel Committee on Banking Supervision. "Principles for sound stress testing
    practices and supervision." May 2009 (reverse stress tests).
Glasserman, P., Kang, C. & Kang, W. (2015). "Stress scenario selection by
    empirical likelihood." *Quantitative Finance* 15(1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core import DEFAULT_SEED, Portfolio

# Factor universe and per-asset loadings onto
# [equity, rates, credit, fx, commodity].
_FACTORS = ["equity", "rates", "credit", "fx", "commodity"]
_FACTOR_LOADINGS: dict[str, dict[str, float]] = {
    "SPY": {"equity": 1.00},
    "QQQ": {"equity": 1.20},
    "IWM": {"equity": 1.10},
    "GLD": {"rates": -0.30, "commodity": 0.40},
    "TLT": {"rates": 1.00},
    "HYG": {"credit": 1.00, "equity": 0.30},
    "EEM": {"equity": 0.90, "fx": 0.60},
    "USO": {"commodity": 1.00},
    "VNQ": {"equity": 0.80, "rates": 0.30},
    "XLF": {"equity": 1.10, "credit": 0.20},
}


class ReverseStressTester:
    """Find the scenarios that break a portfolio at a target loss.

    Parameters
    ----------
    seed : int
        RNG seed for reproducible Monte-Carlo searches.
    """

    #: Hard cap on internal Monte-Carlo paths to bound runtime.
    MAX_SIM = 200_000

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed

    # ---------------------------------------------------- factor machinery
    def _beta(self, asset_names: list[str]) -> np.ndarray:
        """Build the ``assets x factors`` loading matrix ``beta``."""
        beta = np.zeros((len(asset_names), len(_FACTORS)))
        for i, a in enumerate(asset_names):
            loadings = _FACTOR_LOADINGS.get(a, {})
            for f, v in loadings.items():
                beta[i, _FACTORS.index(f)] = v
        return beta

    # -------------------------------------------- minimum_stress_scenario
    def minimum_stress_scenario(
        self, portfolio: Portfolio, loss_threshold_pct: float
    ) -> dict[str, object]:
        """Smallest (min-norm) factor shock that delivers a target loss.

        Solves, in return space, ``min ||s||`` subject to
        ``w^T beta s = -L`` where ``L = loss_threshold_pct`` (a positive
        loss fraction).  The closed-form minimum-norm solution is
        ``s* = (beta^T w)(-L) / (w^T beta beta^T w)``.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to break.
        loss_threshold_pct : float
            Target loss as a positive fraction of notional (e.g. ``0.10``).

        Returns
        -------
        dict
            ``{'factor_shocks', 'asset_shocks', 'implied_loss_pct',
            'implied_loss'}``.
        """
        beta = self._beta(portfolio.asset_names)
        w = portfolio.weights
        a = beta.T @ w  # factor sensitivity of the portfolio (n_factors,)
        denom = float(a @ a)
        L = float(loss_threshold_pct)

        if denom == 0.0:
            factor_shocks = np.zeros(len(_FACTORS))
        else:
            factor_shocks = a * (-L) / denom

        asset_shocks = beta @ factor_shocks  # per-asset return shock
        implied_loss_pct = float(-(w @ asset_shocks))  # positive loss fraction

        return {
            "factor_shocks": {f: float(s) for f, s in zip(_FACTORS, factor_shocks)},
            "asset_shocks": {
                a_: float(s) for a_, s in zip(portfolio.asset_names, asset_shocks)
            },
            "implied_loss_pct": implied_loss_pct,
            "implied_loss": implied_loss_pct * portfolio.notional,
        }

    # ------------------------------------------------ find_breaking_scenarios
    def find_breaking_scenarios(
        self,
        portfolio: Portfolio,
        loss_threshold: float,
        n_sim: int = 1_000_000,
        tolerance: float = 0.1,
    ) -> dict[str, object]:
        """Monte-Carlo search for asset moves that break the portfolio.

        Simulates one-day asset returns from the sample covariance and
        isolates paths whose portfolio loss lands *near* the threshold,
        returning the conditional mean/std of asset moves on those paths.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to break.
        loss_threshold : float
            Target loss as a positive fraction of notional.
        n_sim : int
            Requested number of simulated paths.  Internally capped at
            :attr:`MAX_SIM` for runtime; the cap is reported in the result.
        tolerance : float
            Relative half-width of the band around the threshold (e.g. ``0.1``
            keeps losses within +/-10% of the target).

        Returns
        -------
        dict
            ``{'n_sim', 'n_breaking', 'prob_break', 'conditional_mean',
            'conditional_std', 'capped'}``.
        """
        capped = n_sim > self.MAX_SIM
        n = min(int(n_sim), self.MAX_SIM)

        rng = np.random.default_rng(self.seed)
        cov = portfolio.returns.cov().to_numpy()
        mean = portfolio.returns.mean().to_numpy()
        sims = rng.multivariate_normal(mean, cov, size=n)

        port_ret = sims @ portfolio.weights
        loss = -port_ret  # positive = loss fraction
        target = float(loss_threshold)
        band = tolerance * target

        near = np.abs(loss - target) <= band
        breaking = loss >= target

        names = portfolio.asset_names
        if near.sum() > 0:
            cond = sims[near]
            cond_mean = dict(zip(names, cond.mean(axis=0)))
            cond_std = dict(zip(names, cond.std(axis=0, ddof=1) if cond.shape[0] > 1 else np.zeros(len(names))))
        else:
            cond_mean = {a: float("nan") for a in names}
            cond_std = {a: float("nan") for a in names}

        return {
            "n_sim": n,
            "n_breaking": int(breaking.sum()),
            "prob_break": float(breaking.mean()),
            "conditional_mean": {k: float(v) for k, v in cond_mean.items()},
            "conditional_std": {k: float(v) for k, v in cond_std.items()},
            "capped": capped,
            "note": (
                f"n_sim capped at {self.MAX_SIM:,} for runtime" if capped else ""
            ),
        }

    # --------------------------------------------------- probability_of_break
    def probability_of_break(
        self, portfolio: Portfolio, loss_threshold: float, model: str = "normal"
    ) -> float:
        """Analytic probability that the one-day loss exceeds a threshold.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.
        loss_threshold : float
            Loss as a positive fraction of notional.
        model : {'normal'}
            Distributional assumption for the portfolio return.

        Returns
        -------
        float
            ``P(loss > threshold)``.
        """
        if model != "normal":
            raise ValueError(f"Unsupported model {model!r}; only 'normal' is implemented")
        ret = portfolio.return_series()
        mu = float(ret.mean())
        sigma = float(ret.std(ddof=1))
        if sigma == 0.0:
            return 0.0
        # P(-r > L) = P(r < -L) = Phi((-L - mu)/sigma).
        return float(stats.norm.cdf((-loss_threshold - mu) / sigma))

    # ------------------------------------------------ break_scenario_distribution
    def break_scenario_distribution(
        self, portfolio: Portfolio, loss_threshold: float, n_sim: int = 100_000
    ) -> dict[str, object]:
        """Conditional asset-return distributions given a catastrophic loss.

        Simulates returns and conditions on portfolio loss exceeding the
        threshold, returning per-asset conditional moments and quantiles.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to evaluate.
        loss_threshold : float
            Loss as a positive fraction of notional.
        n_sim : int
            Number of simulated paths (capped at :attr:`MAX_SIM`).

        Returns
        -------
        dict
            ``{'n_breaking', 'conditional_mean', 'conditional_std',
            'q05', 'q50', 'q95'}`` mapping each asset to its statistics.
        """
        n = min(int(n_sim), self.MAX_SIM)
        rng = np.random.default_rng(self.seed)
        cov = portfolio.returns.cov().to_numpy()
        mean = portfolio.returns.mean().to_numpy()
        sims = rng.multivariate_normal(mean, cov, size=n)

        loss = -(sims @ portfolio.weights)
        mask = loss >= float(loss_threshold)
        names = portfolio.asset_names

        if mask.sum() == 0:
            nan_map = {a: float("nan") for a in names}
            return {
                "n_breaking": 0,
                "conditional_mean": dict(nan_map),
                "conditional_std": dict(nan_map),
                "q05": dict(nan_map),
                "q50": dict(nan_map),
                "q95": dict(nan_map),
            }

        cond = sims[mask]
        q05, q50, q95 = np.quantile(cond, [0.05, 0.5, 0.95], axis=0)
        std = cond.std(axis=0, ddof=1) if cond.shape[0] > 1 else np.zeros(len(names))
        return {
            "n_breaking": int(mask.sum()),
            "conditional_mean": {a: float(v) for a, v in zip(names, cond.mean(axis=0))},
            "conditional_std": {a: float(v) for a, v in zip(names, std)},
            "q05": {a: float(v) for a, v in zip(names, q05)},
            "q50": {a: float(v) for a, v in zip(names, q50)},
            "q95": {a: float(v) for a, v in zip(names, q95)},
        }
