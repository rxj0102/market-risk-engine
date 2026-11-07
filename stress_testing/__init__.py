"""Stress-testing package.

Forward, reverse, historical and sensitivity stress-testing engines for the
market-risk library.

Public classes
--------------
HistoricalScenarioEngine
    Replay realised market crises onto a portfolio.
HypotheticalScenarioEngine
    Apply forward-looking, analyst-defined shocks.
ReverseStressTester
    Find scenarios that break a portfolio at a target loss.
SensitivityAnalyzer
    Single- and two-factor sensitivity / tornado / grid analysis.
StressTestEngine
    Unified facade aggregating all of the above.
"""

from stress_testing.historical_scenarios import (
    SCENARIOS,
    HistoricalScenarioEngine,
)
from stress_testing.hypothetical_scenarios import (
    HYPOTHETICAL_SCENARIOS,
    HypotheticalScenarioEngine,
)
from stress_testing.reverse_stress import ReverseStressTester
from stress_testing.sensitivity_analysis import SensitivityAnalyzer
from stress_testing.stress_engine import StressTestEngine

__all__ = [
    "SCENARIOS",
    "HYPOTHETICAL_SCENARIOS",
    "HistoricalScenarioEngine",
    "HypotheticalScenarioEngine",
    "ReverseStressTester",
    "SensitivityAnalyzer",
    "StressTestEngine",
]
