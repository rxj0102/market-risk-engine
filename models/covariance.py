"""Covariance estimation: sample, EWMA, DCC-GARCH and Ledoit-Wolf shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import RISKMETRICS_LAMBDA, TRADING_DAYS
from models.garch import GARCHModel


class CovarianceEstimator:
    """Estimate return covariance matrices by several methods.

    All methods return *daily* covariance matrices unless ``annualize=True``.

    References
    ----------
    J.P. Morgan. "RiskMetrics Technical Document" (EWMA).
    Engle, R. (2002). "Dynamic Conditional Correlation." *JBES* 20(3).
    Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-
    dimensional covariance matrices." *J. Multivariate Analysis* 88(2).
    """

    def __init__(self) -> None:
        self._garch = GARCHModel()

    # ----------------------------------------------------------- sample
    def sample_covariance(self, returns: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
        """Equal-weighted sample covariance matrix."""
        cov = returns.cov()
        if annualize:
            cov = cov * TRADING_DAYS
        return cov

    # ------------------------------------------------------------- EWMA
    def ewma_covariance(
        self, returns: pd.DataFrame, lambda_: float = RISKMETRICS_LAMBDA, annualize: bool = False
    ) -> pd.DataFrame:
        """Current EWMA covariance via the RiskMetrics recursion.

        ``Sigma_t = lambda * Sigma_{t-1} + (1-lambda) r_t r_t^T`` computed
        recursively in a single pass (no full recomputation per step).
        """
        x = returns.to_numpy()
        x = x - x.mean(axis=0)
        n, k = x.shape
        sigma = np.cov(x, rowvar=False)  # seed with sample covariance
        one_minus = 1.0 - lambda_
        for t in range(n):
            r = x[t][:, None]
            sigma = lambda_ * sigma + one_minus * (r @ r.T)
        if annualize:
            sigma = sigma * TRADING_DAYS
        return pd.DataFrame(sigma, index=returns.columns, columns=returns.columns)

    def ewma_covariance_series(
        self, returns: pd.DataFrame, lambda_: float = RISKMETRICS_LAMBDA
    ) -> list[np.ndarray]:
        """Time series of EWMA covariance matrices (one per date)."""
        x = returns.to_numpy()
        x = x - x.mean(axis=0)
        n = x.shape[0]
        sigma = np.cov(x, rowvar=False)
        one_minus = 1.0 - lambda_
        out: list[np.ndarray] = []
        for t in range(n):
            r = x[t][:, None]
            sigma = lambda_ * sigma + one_minus * (r @ r.T)
            out.append(sigma.copy())
        return out

    # -------------------------------------------------------------- DCC
    def dcc_covariance(
        self, returns: pd.DataFrame, a: float = 0.05, b: float = 0.93
    ) -> tuple[pd.DataFrame, list[np.ndarray]]:
        """DCC-GARCH covariance (current matrix and correlation series).

        Step 1 fits univariate GARCH(1,1) to each asset; step 2 runs the DCC
        recursion on the standardised residuals; step 3 recombines with the
        GARCH volatilities.
        """
        cols = list(returns.columns)
        k = len(cols)
        # Step 1: univariate GARCH -> standardised residuals & vol paths.
        z = np.empty((len(returns), k))
        vol = np.empty((len(returns), k))
        for j, c in enumerate(cols):
            params = self._garch.fit(returns[c])
            vol[:, j] = self._garch.conditional_vols(returns[c], params)
            z[:, j] = self._garch.standardized_residuals(returns[c], params)

        # Step 2: DCC recursion.
        q_bar = np.cov(z, rowvar=False)
        q = q_bar.copy()
        corr_series: list[np.ndarray] = []
        for t in range(len(returns)):
            zt = z[t][:, None]
            q = (1 - a - b) * q_bar + a * (zt @ zt.T) + b * q
            d_inv = np.diag(1.0 / np.sqrt(np.diag(q)))
            r_t = d_inv @ q @ d_inv
            corr_series.append(r_t)

        # Step 3: Sigma_t = D_t R_t D_t with last-step GARCH vols.
        d_last = np.diag(vol[-1])
        sigma = d_last @ corr_series[-1] @ d_last
        cov_df = pd.DataFrame(sigma, index=cols, columns=cols)
        return cov_df, corr_series

    # ------------------------------------------------------ Ledoit-Wolf
    def ledoit_wolf_covariance(self, returns: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage covariance (via scikit-learn)."""
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(returns.to_numpy())
        cov = lw.covariance_
        if annualize:
            cov = cov * TRADING_DAYS
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

    # --------------------------------------------------------- dispatch
    def estimate(self, returns: pd.DataFrame, method: str = "sample", **kwargs) -> pd.DataFrame:
        """Dispatch to the requested covariance estimator.

        Parameters
        ----------
        method : {'sample', 'ewma', 'dcc', 'ledoit_wolf'}
        """
        method = method.lower()
        if method == "sample":
            return self.sample_covariance(returns, **kwargs)
        if method == "ewma":
            return self.ewma_covariance(returns, **kwargs)
        if method == "dcc":
            return self.dcc_covariance(returns, **kwargs)[0]
        if method in ("ledoit_wolf", "ledoitwolf", "lw"):
            return self.ledoit_wolf_covariance(returns, **kwargs)
        raise ValueError(f"Unknown covariance method: {method!r}")
