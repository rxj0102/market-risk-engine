"""Expected Shortfall (CVaR) estimation package.

Exposes historical, parametric and Monte-Carlo Expected Shortfall estimators
plus a unified :class:`ESEngine` aggregator.
"""

from expected_shortfall.es_engine import ESEngine
from expected_shortfall.es_historical import HistoricalES
from expected_shortfall.es_monte_carlo import MCExpectedShortfall
from expected_shortfall.es_parametric import ParametricES

__all__ = [
    "HistoricalES",
    "ParametricES",
    "MCExpectedShortfall",
    "ESEngine",
]
