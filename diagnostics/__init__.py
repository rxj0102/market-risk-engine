"""Risk diagnostics: distribution, tail (EVT), correlation and model comparison.

This package provides exploratory and validation diagnostics for the market
risk engine:

- :class:`ReturnDistributionAnalyzer` -- moments, normality tests and
  volatility-clustering (ARCH) diagnostics.
- :class:`TailAnalyzer` -- extreme-value (peaks-over-threshold) tail analysis,
  EVT VaR / ES and the Hill tail-index estimator.
- :class:`CorrelationDynamicsAnalyzer` -- rolling, DCC and regime-conditional
  correlation dynamics.
- :class:`VaRModelComparison` -- backtest-driven ranking and selection of VaR
  models.
"""

from diagnostics.correlation_dynamics import CorrelationDynamicsAnalyzer
from diagnostics.distribution import ReturnDistributionAnalyzer
from diagnostics.model_comparison import VaRModelComparison
from diagnostics.tail_analysis import TailAnalyzer

__all__ = [
    "ReturnDistributionAnalyzer",
    "TailAnalyzer",
    "CorrelationDynamicsAnalyzer",
    "VaRModelComparison",
]
