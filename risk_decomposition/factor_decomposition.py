"""Factor-based risk decomposition: PCA and explicit factor regression.

Two complementary views are provided:

* **Statistical factors** via principal-component analysis (PCA) of the daily
  covariance matrix, where eigenvectors are uncorrelated risk factors and
  eigenvalues their variances.
* **Fundamental factors** via OLS regression of asset returns on an explicit
  panel of factor returns, splitting portfolio risk into *systematic* and
  *idiosyncratic* parts.

References
----------
Connor, G. "The Three Types of Factor Models." Financial Analysts Journal,
1995.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import Portfolio
from models import CovarianceEstimator
from utils.statistics import z_score


def default_factor_returns(portfolio: Portfolio) -> pd.DataFrame:
    """Derive proxy factor returns from the portfolio's own asset returns.

    Builds a small set of intuitive, runnable factors from common ETF
    constituents so that the regression-based methods work out of the box on
    the sample equity portfolio.

    Factors
    -------
    ``market``
        Broad equity market (``SPY`` if present, else first asset).
    ``size``
        Small minus large (``IWM - SPY``).
    ``credit``
        Credit risk premium (``HYG - TLT``).
    ``commodity``
        Commodity factor (``USO``).
    ``rates``
        Rates / duration factor (``TLT``).

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio whose returns supply the proxy factors.

    Returns
    -------
    pd.DataFrame
        Factor-return panel indexed by the portfolio's dates. Only factors
        whose constituent assets are present are included.
    """
    r = portfolio.returns
    cols = set(r.columns)

    def col(name: str) -> pd.Series | None:
        return r[name] if name in cols else None

    factors: dict[str, pd.Series] = {}

    market = col("SPY")
    if market is None:
        market = r.iloc[:, 0]
    factors["market"] = market

    if {"IWM", "SPY"} <= cols:
        factors["size"] = r["IWM"] - r["SPY"]
    if {"HYG", "TLT"} <= cols:
        factors["credit"] = r["HYG"] - r["TLT"]
    if "USO" in cols:
        factors["commodity"] = r["USO"]
    if "TLT" in cols:
        factors["rates"] = r["TLT"]

    return pd.DataFrame(factors, index=r.index)


class FactorRiskDecomposition:
    """PCA and regression-based factor risk decomposition.

    Parameters
    ----------
    cov_estimator : CovarianceEstimator, optional
        Covariance estimator. A fresh instance is created if omitted.
    """

    def __init__(self, cov_estimator: CovarianceEstimator | None = None) -> None:
        self._cov = cov_estimator if cov_estimator is not None else CovarianceEstimator()

    # ------------------------------------------------------------------ utils
    def _covariance(self, portfolio: Portfolio, cov_method: str) -> np.ndarray:
        cov_df = self._cov.estimate(portfolio.returns, method=cov_method)
        cov_df = cov_df.reindex(index=portfolio.asset_names, columns=portfolio.asset_names)
        return cov_df.to_numpy()

    # ----------------------------------------------------------------- PCA
    def pca_decomposition(
        self,
        portfolio: Portfolio,
        n_factors: int = 5,
        cov_method: str = "sample",
        confidence: float = 0.99,
    ) -> dict:
        """Principal-component decomposition of the covariance matrix.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        n_factors : int, default 5
            Number of leading factors to report (eigenvalues are sorted
            descending). ``variance_explained`` and ``eigenvalues`` cover all
            components so that they obey the trace/variance identities.
        cov_method : str, default 'sample'
            Covariance estimation method (``'sample'`` or ``'ewma'``).
        confidence : float, default 0.99
            Confidence level for the per-factor VaR.

        Returns
        -------
        dict
            Keys: ``eigenvalues`` (all, descending), ``eigenvectors`` (all,
            columns = factors), ``factor_loadings`` (``w . q_k`` for the
            leading factors), ``variance_explained`` (fractions, all
            components, summing to 1), ``cumulative_variance``,
            ``factor_var`` (per leading factor,
            ``z * sqrt(lambda_k) * |w . q_k| * N``) and ``n_factors``.

        Notes
        -----
        The eigenvalues satisfy ``sum(eigenvalues) == trace(Sigma)`` because
        the covariance matrix is symmetric.
        """
        w = portfolio.weights
        cov = self._covariance(portfolio, cov_method)
        z = z_score(confidence)
        notional = portfolio.notional

        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending order; reverse to descending.
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        total = eigvals.sum()
        variance_explained = eigvals / total if total != 0 else np.zeros_like(eigvals)
        cumulative_variance = np.cumsum(variance_explained)

        k = min(n_factors, len(eigvals))
        leading_vecs = eigvecs[:, :k]
        factor_loadings = w @ leading_vecs  # w . q_k for each leading factor
        factor_var = z * np.sqrt(np.clip(eigvals[:k], 0.0, None)) * np.abs(factor_loadings) * notional

        return {
            "eigenvalues": eigvals,
            "eigenvectors": eigvecs,
            "factor_loadings": factor_loadings,
            "variance_explained": variance_explained,
            "cumulative_variance": cumulative_variance,
            "factor_var": factor_var,
            "n_factors": k,
        }

    # -------------------------------------------------------------- regression
    @staticmethod
    def _ols_betas(asset_returns: np.ndarray, factor_returns: np.ndarray) -> np.ndarray:
        """OLS coefficients (with intercept) for each asset column.

        Parameters
        ----------
        asset_returns : np.ndarray
            ``(T, n_assets)`` matrix of aligned asset returns.
        factor_returns : np.ndarray
            ``(T, n_factors)`` matrix of aligned factor returns.

        Returns
        -------
        np.ndarray
            ``(n_assets, n_factors + 1)`` matrix; column 0 is the intercept
            (alpha), the remaining columns are factor betas.
        """
        t = asset_returns.shape[0]
        design = np.column_stack([np.ones(t), factor_returns])
        coeffs, *_ = np.linalg.lstsq(design, asset_returns, rcond=None)
        return coeffs.T  # (n_assets, n_factors + 1)

    def factor_regression(
        self, portfolio: Portfolio, factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """OLS regression of each asset return on the factor panel.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio whose asset returns are regressed.
        factor_returns : pd.DataFrame
            Factor-return panel; aligned with asset returns on the date index.

        Returns
        -------
        pd.DataFrame
            Indexed by asset name with an ``alpha`` column followed by one
            column per factor holding the estimated betas.
        """
        asset_df = portfolio.returns[portfolio.asset_names]
        aligned = asset_df.join(factor_returns, how="inner", lsuffix="_a")
        asset_mat = aligned[asset_df.columns].to_numpy()
        factor_mat = aligned[factor_returns.columns].to_numpy()

        coeffs = self._ols_betas(asset_mat, factor_mat)
        columns = ["alpha", *factor_returns.columns]
        return pd.DataFrame(coeffs, index=list(portfolio.asset_names), columns=columns)

    def systematic_vs_idiosyncratic(
        self, portfolio: Portfolio, factor_returns: pd.DataFrame
    ) -> tuple[float, float]:
        """Split portfolio variance into systematic and idiosyncratic shares.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        factor_returns : pd.DataFrame
            Factor-return panel.

        Returns
        -------
        tuple[float, float]
            ``(systematic_var_pct, idiosyncratic_var_pct)`` summing to 1.

        Notes
        -----
        ``beta_p = sum_i w_i beta_i``;
        ``Var_factors = beta_p^T Cov(factors) beta_p``;
        ``Var_idio = sum_i w_i^2 sigma_eps_i^2`` with residual variances from
        the per-asset regressions.
        """
        w = portfolio.weights
        asset_df = portfolio.returns[portfolio.asset_names]
        aligned = asset_df.join(factor_returns, how="inner", lsuffix="_a")
        asset_mat = aligned[asset_df.columns].to_numpy()
        factor_mat = aligned[factor_returns.columns].to_numpy()

        coeffs = self._ols_betas(asset_mat, factor_mat)  # (n_assets, n_factors+1)
        alpha = coeffs[:, 0]
        betas = coeffs[:, 1:]  # (n_assets, n_factors)

        # Residual variance per asset.
        t = asset_mat.shape[0]
        fitted = alpha[None, :] + factor_mat @ betas.T  # (T, n_assets)
        residuals = asset_mat - fitted
        sigma_eps2 = residuals.var(axis=0, ddof=0)

        beta_p = w @ betas  # (n_factors,)
        cov_factors = np.cov(factor_mat, rowvar=False)
        var_systematic = float(beta_p @ cov_factors @ beta_p)
        var_idio = float(np.sum(w**2 * sigma_eps2))

        total = var_systematic + var_idio
        if total == 0.0:
            return 0.0, 0.0
        return var_systematic / total, var_idio / total

    def factor_var_contribution(
        self,
        portfolio: Portfolio,
        factor_returns: pd.DataFrame,
        confidence: float = 0.99,
    ) -> pd.DataFrame:
        """Per-factor VaR contribution from the regression decomposition.

        Each factor's standalone VaR is
        ``z * |beta_p_k| * sigma_factor_k * N`` and its *contribution* uses the
        Euler decomposition of the systematic variance so that the listed
        contributions are additive.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to analyse.
        factor_returns : pd.DataFrame
            Factor-return panel.
        confidence : float, default 0.99
            VaR confidence level.

        Returns
        -------
        pd.DataFrame
            One row per factor with columns ``Factor``, ``Beta_p``,
            ``FactorVol``, ``StandaloneVaR($)``, ``VarContribution``
            (fraction of systematic variance) and ``ContributionVaR($)``.
        """
        w = portfolio.weights
        asset_df = portfolio.returns[portfolio.asset_names]
        aligned = asset_df.join(factor_returns, how="inner", lsuffix="_a")
        asset_mat = aligned[asset_df.columns].to_numpy()
        factor_mat = aligned[factor_returns.columns].to_numpy()

        coeffs = self._ols_betas(asset_mat, factor_mat)
        betas = coeffs[:, 1:]
        beta_p = w @ betas

        cov_factors = np.cov(factor_mat, rowvar=False)
        z = z_score(confidence)
        notional = portfolio.notional

        factor_vol = np.sqrt(np.clip(np.diag(cov_factors), 0.0, None))
        standalone_var = z * np.abs(beta_p) * factor_vol * notional

        # Euler decomposition of systematic variance: contribution_k =
        # beta_p_k * (Cov_factors beta_p)_k.
        cov_beta = cov_factors @ beta_p
        var_contrib = beta_p * cov_beta
        var_systematic = float(beta_p @ cov_beta)
        var_share = var_contrib / var_systematic if var_systematic != 0 else np.zeros_like(var_contrib)

        systematic_var_dollars = z * np.sqrt(max(var_systematic, 0.0)) * notional
        contribution_var = var_share * systematic_var_dollars

        return pd.DataFrame(
            {
                "Factor": list(factor_returns.columns),
                "Beta_p": beta_p,
                "FactorVol": factor_vol,
                "StandaloneVaR($)": standalone_var,
                "VarContribution": var_share,
                "ContributionVaR($)": contribution_var,
            }
        )
