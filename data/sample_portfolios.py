"""Built-in sample portfolios and a deterministic synthetic return generator.

The generator produces realistic daily return panels with:

* a common market factor driving cross-sectional correlation,
* Student-t innovations for fat tails,
* stochastic-volatility (GARCH-like) clustering, and
* embedded crisis windows (2015 China deval, 2018 Q4, 2020 COVID, 2022 bear)
  with elevated volatility and negative drift.

Data spans ``2015-01-02`` to ``2024-12-31`` on a business-day calendar so that
the historical stress-testing scenarios that fall inside that window can be
replayed on the sample portfolios.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import DEFAULT_SEED, Portfolio

DATA_DIR = Path(__file__).resolve().parent
EQUITY_CSV = DATA_DIR / "sample_equity_portfolio.csv"
FX_CSV = DATA_DIR / "sample_fx_portfolio.csv"

START_DATE = "2015-01-02"
END_DATE = "2024-12-31"

# Equity ETF universe: (name, annual drift, annual vol, market beta)
EQUITY_SPEC: list[tuple[str, float, float, float]] = [
    ("SPY", 0.10, 0.16, 1.00),   # S&P 500
    ("QQQ", 0.14, 0.22, 1.15),   # Nasdaq 100
    ("IWM", 0.08, 0.21, 1.10),   # Russell 2000
    ("GLD", 0.05, 0.14, -0.10),  # Gold (diversifier)
    ("TLT", 0.03, 0.13, -0.35),  # 20Y Treasuries (flight to quality)
    ("HYG", 0.05, 0.10, 0.55),   # High-yield credit
    ("EEM", 0.06, 0.20, 1.05),   # Emerging markets
    ("USO", 0.02, 0.35, 0.45),   # Crude oil
    ("VNQ", 0.08, 0.19, 0.90),   # REITs
    ("XLF", 0.10, 0.20, 1.20),   # Financials
]

# FX universe: (name, annual drift, annual vol, market beta) — returns of the pair
FX_SPEC: list[tuple[str, float, float, float]] = [
    ("EURUSD", 0.00, 0.08, 0.20),
    ("GBPUSD", -0.01, 0.09, 0.25),
    ("USDJPY", 0.01, 0.09, -0.15),
    ("AUDUSD", -0.01, 0.11, 0.45),
    ("USDMXN", 0.02, 0.14, -0.40),
]

# Crisis windows: (start, end, daily vol multiplier, extra daily drift)
CRISIS_WINDOWS: list[tuple[str, str, float, float]] = [
    ("2015-08-17", "2015-09-30", 2.2, -0.0030),  # China devaluation
    ("2018-10-01", "2018-12-24", 1.9, -0.0022),  # Q4 2018 selloff
    ("2020-02-19", "2020-03-23", 3.5, -0.0070),  # COVID crash
    ("2020-03-24", "2020-06-30", 1.8, 0.0030),   # COVID recovery
    ("2022-01-03", "2022-10-12", 1.6, -0.0016),  # 2022 rate-hike bear
]


def _trading_calendar() -> pd.DatetimeIndex:
    """Business-day index covering the sample period."""
    return pd.bdate_range(START_DATE, END_DATE)


def _crisis_multipliers(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Per-date volatility multiplier and extra drift from crisis windows."""
    vol_mult = np.ones(len(dates))
    extra_drift = np.zeros(len(dates))
    for start, end, mult, drift in CRISIS_WINDOWS:
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        vol_mult[mask] = np.maximum(vol_mult[mask], mult)
        extra_drift[mask] += drift
    return vol_mult, extra_drift


def _simulate(spec: list[tuple[str, float, float, float]], seed: int) -> pd.DataFrame:
    """Generate a synthetic daily return panel for the given asset spec."""
    rng = np.random.default_rng(seed)
    dates = _trading_calendar()
    n_days = len(dates)
    n_assets = len(spec)

    names = [s[0] for s in spec]
    mu = np.array([s[1] for s in spec]) / 252.0
    sigma = np.array([s[2] for s in spec]) / np.sqrt(252.0)
    beta = np.array([s[3] for s in spec])

    vol_mult, extra_drift = _crisis_multipliers(dates)

    # Stochastic volatility: AR(1) in log-vol shared structure + idiosyncratic.
    log_vol = np.zeros((n_days, n_assets))
    persistence = 0.94
    vol_of_vol = 0.30
    state = np.zeros(n_assets)
    for t in range(n_days):
        state = persistence * state + vol_of_vol * rng.standard_normal(n_assets)
        log_vol[t] = state
    sv = np.exp(log_vol - 0.5 * np.var(log_vol, axis=0))  # mean ~1 multiplicative SV

    # Common market factor (fat-tailed) plus idiosyncratic fat-tailed shocks.
    df_t = 5.0  # Student-t degrees of freedom -> heavy tails
    market = rng.standard_t(df_t, size=n_days) / np.sqrt(df_t / (df_t - 2))
    idio = rng.standard_t(df_t, size=(n_days, n_assets)) / np.sqrt(df_t / (df_t - 2))

    # Build returns: drift + beta * market shock + idiosyncratic, scaled by SV & crises.
    daily_vol = sigma[None, :] * sv * vol_mult[:, None]
    systematic = beta[None, :] * market[:, None]
    idio_weight = np.sqrt(np.maximum(1.0 - beta**2 * 0.0, 0.0))  # keep idio full-strength
    shocks = 0.6 * systematic + np.sqrt(1.0 - 0.6**2) * idio
    returns = mu[None, :] + extra_drift[:, None] * np.abs(beta)[None, :] + daily_vol * shocks

    return pd.DataFrame(returns, index=dates, columns=names)


def generate_equity_returns(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Deterministically generate the 10-asset equity return panel."""
    return _simulate(EQUITY_SPEC, seed=seed)


def generate_fx_returns(seed: int = DEFAULT_SEED + 1) -> pd.DataFrame:
    """Deterministically generate the 5-pair FX return panel."""
    return _simulate(FX_SPEC, seed=seed)


def write_sample_csvs(seed: int = DEFAULT_SEED) -> tuple[Path, Path]:
    """Generate and persist the sample CSVs. Returns the two paths."""
    eq = generate_equity_returns(seed)
    fx = generate_fx_returns(seed + 1)
    eq.to_csv(EQUITY_CSV, index_label="date", float_format="%.6f")
    fx.to_csv(FX_CSV, index_label="date", float_format="%.6f")
    return EQUITY_CSV, FX_CSV


def load_equity_returns() -> pd.DataFrame:
    """Load the equity return panel, regenerating the CSV if absent."""
    if not EQUITY_CSV.exists():
        write_sample_csvs()
    return pd.read_csv(EQUITY_CSV, index_col="date", parse_dates=True)


def load_fx_returns() -> pd.DataFrame:
    """Load the FX return panel, regenerating the CSV if absent."""
    if not FX_CSV.exists():
        write_sample_csvs()
    return pd.read_csv(FX_CSV, index_col="date", parse_dates=True)


def equity_portfolio(notional: float = 10_000_000.0, weights: np.ndarray | None = None) -> Portfolio:
    """Build the built-in sample equity portfolio.

    Parameters
    ----------
    notional : float
        Total portfolio value in USD.
    weights : np.ndarray, optional
        Asset weights; defaults to equal weight.

    Returns
    -------
    Portfolio
    """
    rets = load_equity_returns()
    names = list(rets.columns)
    if weights is None:
        weights = np.full(len(names), 1.0 / len(names))
    return Portfolio(weights=weights, asset_names=names, returns=rets, notional=notional)


def fx_portfolio(notional: float = 5_000_000.0, weights: np.ndarray | None = None) -> Portfolio:
    """Build the built-in sample multi-currency FX portfolio."""
    rets = load_fx_returns()
    names = list(rets.columns)
    if weights is None:
        weights = np.full(len(names), 1.0 / len(names))
    return Portfolio(weights=weights, asset_names=names, returns=rets, notional=notional)


SAMPLE_PORTFOLIOS = {
    "Sample Equity (10 ETFs)": equity_portfolio,
    "Sample FX (5 pairs)": fx_portfolio,
}


if __name__ == "__main__":
    paths = write_sample_csvs()
    print(f"Wrote sample data:\n  {paths[0]}\n  {paths[1]}")
