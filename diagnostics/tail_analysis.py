"""Extreme-value tail analysis via the peaks-over-threshold (POT) method.

The :class:`TailAnalyzer` fits a Generalized Pareto Distribution (GPD) to the
losses exceeding a high threshold and derives EVT-based VaR and Expected
Shortfall, together with the Hill tail-index estimator.

Notes
-----
All routines operate in *loss space* (``loss = -return``). VaR / ES are returned
as positive return-space losses; multiply by the portfolio notional externally
to obtain dollar figures.

References
----------
McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management:
Concepts, Techniques and Tools.* Princeton University Press, ch. 5-7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core import Portfolio

#: GPD shape parameters near zero are treated by the exponential (Gumbel) limit.
_XI_TOL: float = 1e-6


class TailAnalyzer:
    """Peaks-over-threshold extreme-value analysis of portfolio losses."""

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _as_losses(returns: Portfolio | pd.Series | np.ndarray) -> np.ndarray:
        """Coerce input to a clean loss array (``loss = -return``)."""
        if isinstance(returns, Portfolio):
            r = returns.return_series().to_numpy()
        elif isinstance(returns, pd.Series):
            r = returns.to_numpy()
        else:
            r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        return -r

    # ---------------------------------------------------------- GPD fit
    def fit_gpd(
        self,
        returns: Portfolio | pd.Series | np.ndarray,
        threshold_quantile: float = 0.95,
    ) -> dict[str, float]:
        """Fit a Generalized Pareto Distribution to threshold exceedances.

        Parameters
        ----------
        returns : Portfolio, pd.Series or np.ndarray
            Return series; converted internally to losses.
        threshold_quantile : float, default 0.95
            Quantile of the loss distribution used as the POT threshold ``u``.

        Returns
        -------
        dict
            ``xi`` (shape), ``beta`` (scale), ``threshold`` (``u``),
            ``n_exceedances`` and ``n_total``. A heavy tail corresponds to
            ``xi > 0``.
        """
        losses = self._as_losses(returns)
        n_total = int(losses.size)
        u = float(np.quantile(losses, threshold_quantile))
        excess = losses[losses > u] - u
        n_exceed = int(excess.size)

        # scipy parameterises genpareto with shape c (== xi), loc, scale.
        # Fix loc=0 because the excesses are already measured above u.
        xi, _loc, beta = stats.genpareto.fit(excess, floc=0.0)

        return {
            "xi": float(xi),
            "beta": float(beta),
            "threshold": u,
            "n_exceedances": n_exceed,
            "n_total": n_total,
        }

    # ------------------------------------------------------------ EVT VaR
    def evt_var(
        self,
        xi: float,
        beta: float,
        threshold: float,
        n_total: int,
        n_exceed: int,
        confidence: float,
    ) -> float:
        """POT (GPD) Value-at-Risk.

        ``VaR = u + (beta/xi) * ( ((n/N_u)(1-conf))^(-xi) - 1 )``, with the
        ``xi -> 0`` exponential limit ``VaR = u - beta * log((n/N_u)(1-conf))``.

        Parameters
        ----------
        xi, beta, threshold : float
            Fitted GPD shape, scale and threshold ``u``.
        n_total, n_exceed : int
            Total observations and number of exceedances.
        confidence : float
            VaR confidence level, e.g. ``0.99``.

        Returns
        -------
        float
            VaR as a positive return-space loss.
        """
        ratio = (n_total / n_exceed) * (1.0 - confidence)
        if abs(xi) < _XI_TOL:
            var = threshold - beta * np.log(ratio)
        else:
            var = threshold + (beta / xi) * (ratio ** (-xi) - 1.0)
        return float(var)

    # ------------------------------------------------------------- EVT ES
    def evt_es(
        self,
        xi: float,
        beta: float,
        threshold: float,
        n_total: int,
        n_exceed: int,
        confidence: float,
    ) -> float:
        """POT (GPD) Expected Shortfall.

        ``ES = (VaR + beta - xi * u) / (1 - xi)`` for ``xi < 1``; the mean of the
        excess distribution is infinite for ``xi >= 1`` (returns ``inf``).

        Parameters
        ----------
        xi, beta, threshold : float
            Fitted GPD shape, scale and threshold ``u``.
        n_total, n_exceed : int
        confidence : float

        Returns
        -------
        float
            ES as a positive return-space loss.
        """
        var = self.evt_var(xi, beta, threshold, n_total, n_exceed, confidence)
        if xi >= 1.0:
            return float("inf")
        return float((var + beta - xi * threshold) / (1.0 - xi))

    # -------------------------------------------------------- Hill index
    def hill_estimator(
        self, returns: Portfolio | pd.Series | np.ndarray, k: int = 50
    ) -> float:
        """Hill estimator of the tail index from the ``k`` largest losses.

        Parameters
        ----------
        returns : Portfolio, pd.Series or np.ndarray
        k : int, default 50
            Number of upper order statistics used.

        Returns
        -------
        float
            Hill estimate of the tail index ``xi = 1/alpha`` (positive for a
            heavy, power-law tail). ``NaN`` if there are too few positive losses.
        """
        losses = self._as_losses(returns)
        losses = losses[losses > 0]
        if losses.size <= k:
            return float("nan")
        ordered = np.sort(losses)[::-1]  # descending
        top = ordered[: k + 1]
        # xi_hat = (1/k) sum_{i=1..k} ln(X_(i)) - ln(X_(k+1)).
        xi_hat = np.mean(np.log(top[:k])) - np.log(ordered[k])
        return float(xi_hat)

    # --------------------------------------------------------- tail plot
    def tail_plot_data(
        self,
        returns: Portfolio | pd.Series | np.ndarray,
        threshold_quantile: float = 0.95,
    ) -> pd.DataFrame:
        """Empirical vs. fitted-GPD tail exceedance probabilities.

        Parameters
        ----------
        returns : Portfolio, pd.Series or np.ndarray
        threshold_quantile : float, default 0.95

        Returns
        -------
        pd.DataFrame
            For each loss exceeding the threshold: columns ``loss``,
            ``empirical`` (empirical exceedance probability ``P(L > x)``) and
            ``gpd`` (fitted GPD tail probability). Sorted by ``loss``.
        """
        losses = self._as_losses(returns)
        n_total = losses.size
        fit = self.fit_gpd(returns, threshold_quantile)
        u, xi, beta = fit["threshold"], fit["xi"], fit["beta"]
        n_exceed = fit["n_exceedances"]

        exceed = np.sort(losses[losses > u])
        # Empirical exceedance probability P(L > x) via descending ranks.
        m = exceed.size
        empirical = (m - np.arange(m)) / n_total

        # Fitted tail: P(L > x) = (N_u/n) * (1 - GPD_cdf(x - u)).
        tail_gpd = (n_exceed / n_total) * stats.genpareto.sf(exceed - u, xi, loc=0.0, scale=beta)

        return pd.DataFrame({"loss": exceed, "empirical": empirical, "gpd": tail_gpd})
