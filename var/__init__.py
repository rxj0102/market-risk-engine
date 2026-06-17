"""Value-at-Risk estimation package.

Exposes historical, parametric, Monte-Carlo and filtered-historical VaR
estimators plus a unified :class:`VaREngine` aggregator.
"""

from var.filtered_historical import FilteredHistoricalSimulation
from var.historical import HistoricalVaR
from var.monte_carlo import MonteCarloVaR
from var.parametric import ParametricVaR
from var.var_engine import VaREngine

__all__ = [
    "HistoricalVaR",
    "ParametricVaR",
    "MonteCarloVaR",
    "FilteredHistoricalSimulation",
    "VaREngine",
]
