"""Kupiec Proportion-of-Failures (POF) unconditional coverage test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class KupiecTest:
    """Kupiec (1995) POF likelihood-ratio test for VaR exception frequency.

    Notes
    -----
    Under ``H0`` the exception probability equals ``p0 = 1 - confidence``. The
    test statistic

    ``LR_POF = -2 ln[ (1-p0)^{T-x} p0^x / ((1-p_hat)^{T-x} p_hat^x) ]``

    is asymptotically ``chi^2(1)`` where ``x`` is the exception count, ``T`` the
    number of test days and ``p_hat = x/T``.

    References
    ----------
    Kupiec, P. (1995). "Techniques for verifying the accuracy of risk
    measurement models." *Journal of Derivatives* 3(2), 73-84.
    """

    @staticmethod
    def _exceptions_mask(var_series: pd.Series, pnl_series: pd.Series) -> pd.Series:
        """Boolean mask of breaches where loss exceeds VaR (``pnl < -VaR``)."""
        var = np.asarray(var_series, dtype=float)
        pnl = np.asarray(pnl_series, dtype=float)
        mask = pnl < -np.abs(var)
        return pd.Series(mask, index=getattr(pnl_series, "index", None))

    def exception_count(
        self, var_series: pd.Series, pnl_series: pd.Series
    ) -> tuple[int, int, list]:
        """Return ``(n_exceptions, n_total, exception_dates)``."""
        mask = self._exceptions_mask(var_series, pnl_series)
        n_exc = int(mask.sum())
        dates = list(mask.index[mask.to_numpy()]) if mask.index is not None else []
        return n_exc, int(len(mask)), dates

    def test(
        self, var_series: pd.Series, pnl_series: pd.Series, confidence: float = 0.99
    ) -> tuple[float, float, bool]:
        """Run the POF test.

        Returns
        -------
        tuple
            ``(LR_POF statistic, p-value, reject_H0)`` at the 5% level.
        """
        x, T, _ = self.exception_count(var_series, pnl_series)
        p0 = 1.0 - confidence
        if T == 0:
            return 0.0, 1.0, False
        p_hat = x / T

        # Log-likelihoods (guard against log(0)).
        def _ll(p: float) -> float:
            p = min(max(p, 1e-12), 1 - 1e-12)
            return (T - x) * np.log(1 - p) + x * np.log(p)

        if x == 0:
            lr = -2.0 * (_ll(p0) - (T * np.log(1 - 1e-12)))
        else:
            lr = -2.0 * (_ll(p0) - _ll(p_hat))
        lr = float(max(lr, 0.0))
        pvalue = float(stats.chi2.sf(lr, df=1))
        return lr, pvalue, pvalue < 0.05

    def critical_values(
        self, T: int, confidence: float, alpha_test: float = 0.05
    ) -> tuple[int, int]:
        """Acceptance region (min, max exceptions) for the POF test.

        Scans exception counts and returns the contiguous range whose POF
        statistic does not reject ``H0`` at ``alpha_test``.
        """
        p0 = 1.0 - confidence
        crit = stats.chi2.ppf(1 - alpha_test, df=1)
        accepted = []
        for x in range(T + 1):
            p_hat = x / T
            if x == 0:
                lr = -2.0 * ((T) * np.log(1 - p0))
            else:
                ll0 = (T - x) * np.log(1 - p0) + x * np.log(p0)
                ll1 = (T - x) * np.log(max(1 - p_hat, 1e-12)) + x * np.log(max(p_hat, 1e-12))
                lr = -2.0 * (ll0 - ll1)
            if lr <= crit:
                accepted.append(x)
        if not accepted:
            return 0, 0
        return min(accepted), max(accepted)
