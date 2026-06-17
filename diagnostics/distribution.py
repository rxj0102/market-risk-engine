"""Return-distribution diagnostics: moments, normality and dependence structure.

The :class:`ReturnDistributionAnalyzer` operates on a portfolio's realised
return series and quantifies the stylised facts of financial returns -- fat
tails, skewness, volatility clustering (ARCH effects) and non-normality.

References
----------
Cont, R. (2001). "Empirical properties of asset returns: stylized facts and
statistical issues." *Quantitative Finance* 1(2), 223-236.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

from core import Portfolio
from utils.statistics import summary_statistics

#: scipy's Shapiro-Wilk implementation is unreliable / warns above this size.
_SHAPIRO_MAX_N: int = 5000


class ReturnDistributionAnalyzer:
    """Analyse the empirical distribution of portfolio returns.

    All public methods accept either a :class:`~core.Portfolio` (whose
    ``return_series`` is used) or, where noted, a raw return series.
    """

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _as_series(returns: Portfolio | pd.Series | np.ndarray) -> pd.Series:
        """Coerce the supported inputs into a clean (finite) return series."""
        if isinstance(returns, Portfolio):
            s = returns.return_series()
        elif isinstance(returns, pd.Series):
            s = returns
        else:
            s = pd.Series(np.asarray(returns, dtype=float))
        return s[np.isfinite(s)]

    # -------------------------------------------------------- summary stats
    def summary_stats(self, portfolio: Portfolio | pd.Series) -> dict[str, float]:
        """Descriptive moments and the Jarque-Bera normality statistic.

        Parameters
        ----------
        portfolio : Portfolio or pd.Series
            Portfolio (whose return series is used) or a return series.

        Returns
        -------
        dict
            ``mean``, ``std``, ``skewness``, ``kurtosis`` (excess), ``JB_stat``
            and ``JB_pvalue``.
        """
        s = self._as_series(portfolio)
        base = summary_statistics(s)
        return {
            "mean": base["mean"],
            "std": base["std"],
            "skewness": base["skewness"],
            "kurtosis": base["excess_kurtosis"],
            "JB_stat": base["jb_stat"],
            "JB_pvalue": base["jb_pvalue"],
        }

    # ----------------------------------------------------- normality tests
    def normality_tests(self, portfolio: Portfolio | pd.Series) -> dict[str, dict[str, float]]:
        """Battery of normality tests.

        Parameters
        ----------
        portfolio : Portfolio or pd.Series

        Returns
        -------
        dict
            Keys ``jarque_bera``, ``shapiro_wilk`` and ``ks`` each mapping to a
            ``{"statistic": ..., "pvalue": ...}`` sub-dictionary. The
            Kolmogorov-Smirnov test compares the sample against a Normal fitted
            by maximum likelihood (sample mean and standard deviation).
        """
        s = self._as_series(portfolio)
        x = s.to_numpy()

        jb_stat, jb_p = stats.jarque_bera(x)

        # Shapiro-Wilk: subsample if the series is large (scipy warns/errors).
        if x.size > _SHAPIRO_MAX_N:
            rng = np.random.default_rng(42)
            sample = rng.choice(x, size=_SHAPIRO_MAX_N, replace=False)
        else:
            sample = x
        sw_stat, sw_p = stats.shapiro(sample)

        mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
        ks_stat, ks_p = stats.kstest(x, "norm", args=(mu, sigma))

        return {
            "jarque_bera": {"statistic": float(jb_stat), "pvalue": float(jb_p)},
            "shapiro_wilk": {"statistic": float(sw_stat), "pvalue": float(sw_p)},
            "ks": {"statistic": float(ks_stat), "pvalue": float(ks_p)},
        }

    # ----------------------------------------------------------- QQ data
    def qq_plot_data(self, returns: Portfolio | pd.Series | np.ndarray) -> pd.DataFrame:
        """Normal quantile-quantile plot coordinates.

        Parameters
        ----------
        returns : Portfolio, pd.Series or np.ndarray

        Returns
        -------
        pd.DataFrame
            Columns ``theoretical`` (standard-normal quantiles) and
            ``empirical`` (sorted standardised sample quantiles).
        """
        s = self._as_series(returns)
        x = np.sort(s.to_numpy())
        n = x.size
        # Standardise the empirical sample so the reference line is y = x.
        mu, sigma = x.mean(), x.std(ddof=1)
        empirical = (x - mu) / sigma
        # Plotting positions (Blom).
        probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        theoretical = stats.norm.ppf(probs)
        return pd.DataFrame({"theoretical": theoretical, "empirical": empirical})

    # ------------------------------------------------------- histogram data
    def histogram_data(
        self, returns: Portfolio | pd.Series | np.ndarray, n_bins: int = 50
    ) -> dict[str, np.ndarray]:
        """Histogram densities with fitted Normal and Student-t overlays.

        Parameters
        ----------
        returns : Portfolio, pd.Series or np.ndarray
        n_bins : int, default 50

        Returns
        -------
        dict
            ``bin_centers`` (length ``n_bins``), ``counts`` (density-normalised
            histogram heights), ``normal_pdf`` and ``student_t_pdf`` evaluated
            at the bin centres.
        """
        s = self._as_series(returns)
        x = s.to_numpy()
        counts, edges = np.histogram(x, bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        mu, sigma = x.mean(), x.std(ddof=1)
        normal_pdf = stats.norm.pdf(centers, loc=mu, scale=sigma)

        df, t_loc, t_scale = stats.t.fit(x)
        student_t_pdf = stats.t.pdf(centers, df, loc=t_loc, scale=t_scale)

        return {
            "bin_centers": centers,
            "counts": counts,
            "normal_pdf": normal_pdf,
            "student_t_pdf": student_t_pdf,
        }

    # ----------------------------------------------------- autocorrelation
    def autocorrelation(
        self, returns: Portfolio | pd.Series | np.ndarray, lags: int = 20
    ) -> pd.DataFrame:
        """Autocorrelation of returns and squared returns (ARCH effects).

        Parameters
        ----------
        returns : Portfolio, pd.Series or np.ndarray
        lags : int, default 20
            Number of lags (excludes lag 0).

        Returns
        -------
        pd.DataFrame
            Indexed by lag (``1..lags``) with columns ``acf_returns`` and
            ``acf_squared`` (the latter diagnoses volatility clustering).
        """
        s = self._as_series(returns)
        x = s.to_numpy()
        acf_ret = acf(x, nlags=lags, fft=True)
        acf_sq = acf(x**2, nlags=lags, fft=True)
        return pd.DataFrame(
            {"acf_returns": acf_ret[1:], "acf_squared": acf_sq[1:]},
            index=pd.Index(np.arange(1, lags + 1), name="lag"),
        )

    # -------------------------------------------------------- rolling stats
    def rolling_stats(
        self, returns: Portfolio | pd.Series | np.ndarray, window: int = 60
    ) -> pd.DataFrame:
        """Rolling first four moments.

        Parameters
        ----------
        returns : Portfolio, pd.Series or np.ndarray
        window : int, default 60

        Returns
        -------
        pd.DataFrame
            Columns ``mean``, ``vol``, ``skew`` and ``kurtosis`` (excess), with
            leading ``NaN`` rows until the window is full.
        """
        s = self._as_series(returns)
        roll = s.rolling(window)
        return pd.DataFrame(
            {
                "mean": roll.mean(),
                "vol": roll.std(ddof=1),
                "skew": roll.skew(),
                "kurtosis": roll.kurt(),
            }
        )
