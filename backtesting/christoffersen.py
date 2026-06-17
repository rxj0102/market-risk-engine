"""Christoffersen independence and conditional-coverage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from backtesting.kupiec import KupiecTest


class ChristoffersenTest:
    """Christoffersen (1998) independence and conditional-coverage LR tests.

    Notes
    -----
    The independence test compares a first-order Markov exception process
    against an i.i.d. process via the transition counts ``n_ij``. The
    conditional-coverage statistic ``LR_CC = LR_POF + LR_IND`` is asymptotically
    ``chi^2(2)`` and jointly tests exception frequency and independence.

    References
    ----------
    Christoffersen, P. (1998). "Evaluating interval forecasts."
    *International Economic Review* 39(4), 841-862.
    """

    def __init__(self) -> None:
        self._kupiec = KupiecTest()

    def independence_test(
        self, exceptions_series: pd.Series
    ) -> tuple[float, float, np.ndarray]:
        """First-order Markov independence test.

        Returns
        -------
        tuple
            ``(LR_IND statistic, p-value, 2x2 transition-count matrix)``.
        """
        e = np.asarray(exceptions_series, dtype=int)
        n = len(e)
        if n < 2:
            return 0.0, 1.0, np.zeros((2, 2))

        trans = np.zeros((2, 2))
        for i, j in zip(e[:-1], e[1:]):
            trans[i, j] += 1
        n00, n01 = trans[0, 0], trans[0, 1]
        n10, n11 = trans[1, 0], trans[1, 1]

        pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)

        def _safe(p: float) -> float:
            return min(max(p, 1e-12), 1 - 1e-12)

        ll_null = (n00 + n10) * np.log(_safe(1 - pi)) + (n01 + n11) * np.log(_safe(pi))
        ll_alt = (
            n00 * np.log(_safe(1 - pi01))
            + n01 * np.log(_safe(pi01))
            + n10 * np.log(_safe(1 - pi11))
            + n11 * np.log(_safe(pi11))
        )
        lr = float(max(-2.0 * (ll_null - ll_alt), 0.0))
        pvalue = float(stats.chi2.sf(lr, df=1))
        return lr, pvalue, trans

    def conditional_coverage_test(
        self, var_series: pd.Series, pnl_series: pd.Series, confidence: float = 0.99
    ) -> tuple[float, float, bool]:
        """Joint conditional-coverage test ``LR_CC = LR_POF + LR_IND ~ chi^2(2)``."""
        mask = self._kupiec._exceptions_mask(var_series, pnl_series).astype(int)
        lr_pof, _, _ = self._kupiec.test(var_series, pnl_series, confidence)
        lr_ind, _, _ = self.independence_test(mask)
        lr_cc = float(lr_pof + lr_ind)
        pvalue = float(stats.chi2.sf(lr_cc, df=2))
        return lr_cc, pvalue, pvalue < 0.05

    def exception_clustering_chart(self, exceptions_series: pd.Series) -> pd.DataFrame:
        """Timeline data for visualising exception clustering.

        Returns a DataFrame with the exception indicator and the running gap
        (days since the previous exception) to reveal serial dependence.
        """
        e = pd.Series(np.asarray(exceptions_series, dtype=int),
                      index=getattr(exceptions_series, "index", None))
        idx = np.arange(len(e))
        exc_positions = idx[e.to_numpy() == 1]
        gaps = np.full(len(e), np.nan)
        last = None
        for pos in exc_positions:
            if last is not None:
                gaps[pos] = pos - last
            last = pos
        return pd.DataFrame({"exception": e.to_numpy(), "gap_since_prev": gaps}, index=e.index)
