"""Risk decomposition: component/marginal VaR, factor and correlation risk.

This package decomposes total portfolio risk into interpretable building
blocks:

* :class:`ComponentVaR` -- additive per-asset VaR contributions plus
  diversification and concentration diagnostics.
* :class:`MarginalVaR` -- marginal and incremental VaR and hedging analytics.
* :class:`FactorRiskDecomposition` -- PCA and regression-based factor risk,
  with a :func:`default_factor_returns` helper for ready-to-run factors.
* :class:`CorrelationRiskAnalyzer` -- correlation contribution, rolling
  correlation risk and correlation stress testing.
"""

from risk_decomposition.component_var import ComponentVaR
from risk_decomposition.correlation_risk import CorrelationRiskAnalyzer
from risk_decomposition.factor_decomposition import (
    FactorRiskDecomposition,
    default_factor_returns,
)
from risk_decomposition.marginal_var import MarginalVaR

__all__ = [
    "ComponentVaR",
    "MarginalVaR",
    "FactorRiskDecomposition",
    "CorrelationRiskAnalyzer",
    "default_factor_returns",
]
