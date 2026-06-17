"""Berkowitz (2001) likelihood-ratio test on the probability-integral transform."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from utils.statistics import z_score


class BerkowitzTest:
    """Berkowitz density-forecast test based on the PIT transform.

    Notes
    -----
    Given a forecast predictive distribution, the probability-integral transform
    ``u_t = F_t(pnl_t)`` should be i.i.d. ``U(0,1)`` and ``z_t = Phi^{-1}(u_t)``
    i.i.d. ``N(0,1)``. The Berkowitz LR test fits an AR(1) Gaussian model

    ``z_t = mu + rho (z_{t-1} - mu) + e_t``

    and tests ``H0: mu = 0, rho = 0, sigma^2 = 1`` (``chi^2(3)``).

    When only a VaR series is available the predictive distribution is taken as
    Normal with the volatility implied by the VaR quantile,
    ``sigma_t = VaR_t / z_alpha``, which is the common practical specification.

    References
    ----------
    Berkowitz, J. (2001). "Testing density forecasts, with applications to risk
    management." *Journal of Business & Economic Statistics* 19(4), 465-474.
    """

    def _pit_z(
        self,
        var_series: pd.Series,
        pnl_series: pd.Series,
        confidence: float,
        mu_series: pd.Series | None,
        sigma_series: pd.Series | None,
    ) -> np.ndarray:
        pnl = np.asarray(pnl_series, dtype=float)
        if sigma_series is None:
            z_alpha = z_score(confidence)
            sigma = np.abs(np.asarray(var_series, dtype=float)) / z_alpha
        else:
            sigma = np.asarray(sigma_series, dtype=float)
        mu = np.zeros_like(pnl) if mu_series is None else np.asarray(mu_series, dtype=float)
        sigma = np.where(sigma <= 0, np.nan, sigma)
        u = stats.norm.cdf((pnl - mu) / sigma)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return stats.norm.ppf(u)

    def test(
        self,
        var_series: pd.Series,
        pnl_series: pd.Series,
        confidence: float = 0.99,
        mu_series: pd.Series | None = None,
        sigma_series: pd.Series | None = None,
    ) -> tuple[float, float, bool]:
        """Run the Berkowitz LR test.

        Returns
        -------
        tuple
            ``(LR statistic, p-value, reject_H0)`` against ``chi^2(3)``.
        """
        z = self._pit_z(var_series, pnl_series, confidence, mu_series, sigma_series)
        z = z[np.isfinite(z)]
        n = len(z)
        if n < 3:
            return 0.0, 1.0, False

        # Unrestricted AR(1) MLE.
        z_lag = z[:-1]
        z_cur = z[1:]
        rho = np.corrcoef(z_lag, z_cur)[0, 1]
        rho = 0.0 if not np.isfinite(rho) else rho
        mu = z.mean() * (1 - rho)
        resid = z_cur - mu - rho * (z_lag - z.mean())
        sigma2 = max(np.var(resid), 1e-12)

        def _ll(m: float, r: float, s2: float) -> float:
            res = z_cur - m - r * (z_lag - z.mean())
            return -0.5 * np.sum(np.log(2 * np.pi * s2) + res**2 / s2)

        ll_unrestricted = _ll(mu, rho, sigma2)
        ll_restricted = _ll(0.0, 0.0, 1.0)
        lr = float(max(-2.0 * (ll_restricted - ll_unrestricted), 0.0))
        pvalue = float(stats.chi2.sf(lr, df=3))
        return lr, pvalue, pvalue < 0.05
