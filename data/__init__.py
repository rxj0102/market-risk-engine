"""Market data loading, return computation and sample portfolios."""

from data.loader import MarketDataLoader
from data.returns import ReturnCalculator
from data.sample_portfolios import (
    SAMPLE_PORTFOLIOS,
    equity_portfolio,
    fx_portfolio,
    generate_equity_returns,
    generate_fx_returns,
    load_equity_returns,
    load_fx_returns,
    write_sample_csvs,
)

__all__ = [
    "MarketDataLoader",
    "ReturnCalculator",
    "SAMPLE_PORTFOLIOS",
    "equity_portfolio",
    "fx_portfolio",
    "generate_equity_returns",
    "generate_fx_returns",
    "load_equity_returns",
    "load_fx_returns",
    "write_sample_csvs",
]
