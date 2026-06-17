"""Risk-statistics helper functions shared across modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core import TRADING_DAYS


def z_score(confidence: float) -> float:
    """Lower-tail standard-normal quantile ``z_alpha`` as a positive number.

    For a 99% confidence level this returns ``2.326``.
    """
    return float(-stats.norm.ppf(1.0 - confidence))


def normal_pdf_at_quantile(confidence: float) -> float:
    """Standard-normal PDF evaluated at the confidence quantile ``phi(z_alpha)``."""
    return float(stats.norm.pdf(stats.norm.ppf(confidence)))


def annualize_vol(daily_vol: float, periods: int = TRADING_DAYS) -> float:
    """Annualise a daily volatility by ``sqrt(periods)``."""
    return daily_vol * np.sqrt(periods)


def empirical_quantile(losses: np.ndarray, confidence: float) -> float:
    """Loss quantile at ``confidence`` (e.g. 99th percentile of the loss series)."""
    return float(np.quantile(losses, confidence))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile of ``values`` at probability ``q`` (for age-weighted VaR)."""
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return float(np.interp(q, cum, v))


def summary_statistics(returns: pd.Series | np.ndarray) -> dict[str, float]:
    """Mean, vol, skewness, excess kurtosis and Jarque-Bera test."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    jb_stat, jb_p = stats.jarque_bera(r)
    return {
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_p),
    }


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a return series (positive fraction)."""
    r = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return float(-dd.min())


def sharpe_ratio(returns: pd.Series | np.ndarray, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio from periodic returns."""
    r = np.asarray(returns, dtype=float)
    excess = r.mean() * periods - rf
    vol = r.std(ddof=1) * np.sqrt(periods)
    return float(excess / vol) if vol > 0 else float("nan")


def cornish_fisher_quantile(confidence: float, skew: float, excess_kurt: float) -> float:
    """Cornish-Fisher adjusted lower-tail quantile (positive number).

    ``z_CF = z + (z^2-1)s/6 + (z^3-3z)k/24 - (2z^3-5z)s^2/36`` evaluated at the
    lower-tail quantile ``z = Phi^{-1}(1-confidence)``.
    """
    z = stats.norm.ppf(1.0 - confidence)
    z_cf = (
        z
        + (z**2 - 1) * skew / 6.0
        + (z**3 - 3 * z) * excess_kurt / 24.0
        - (2 * z**3 - 5 * z) * skew**2 / 36.0
    )
    return float(-z_cf)


def herfindahl_index(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman concentration index of a share vector."""
    s = np.asarray(shares, dtype=float)
    total = np.abs(s).sum()
    if total == 0:
        return float("nan")
    return float(np.sum((np.abs(s) / total) ** 2))
