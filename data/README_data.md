# Sample Data

This directory contains two **synthetically generated** return panels used by
the dashboard, tests, notebook and research paper. The data is produced
deterministically by [`sample_portfolios.py`](sample_portfolios.py) and is
**not** real market data.

> ⚠️ **Simulated data.** Figures are illustrative. Upload your own returns CSV
> via the dashboard sidebar to analyse a real portfolio.

## Files

| File | Description |
|------|-------------|
| `sample_equity_portfolio.csv` | 10-asset equity-ETF daily simple returns |
| `sample_fx_portfolio.csv` | 5-pair FX daily returns |

### Format

```
date,SPY,QQQ,IWM,GLD,TLT,HYG,EEM,USO,VNQ,XLF
2015-01-02,0.001234,-0.002101,...
...
```

- **Index:** `date` — business days from `2015-01-02` to `2024-12-31`
  (~2,608 observations).
- **Columns:** one per asset; values are **daily simple returns** (not prices,
  not percentages — e.g. `0.0123` = +1.23%).

### Equity universe

| Ticker | Exposure |
|--------|----------|
| SPY | S&P 500 |
| QQQ | Nasdaq 100 |
| IWM | Russell 2000 (small cap) |
| GLD | Gold |
| TLT | 20Y+ US Treasuries |
| HYG | High-yield credit |
| EEM | Emerging-market equity |
| USO | Crude oil |
| VNQ | US REITs |
| XLF | Financials |

### FX universe

`EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `USDMXN` (returns of each pair).

## How the data is generated

`sample_portfolios.py` simulates correlated returns with realistic risk
features:

- a **common market factor** with Student-t (fat-tailed) innovations driving
  cross-sectional correlation, plus per-asset idiosyncratic shocks;
- **stochastic volatility** (AR(1) log-vol) producing volatility clustering;
- **asymmetric downward jumps** in the market factor → negative tail skew and
  crash risk;
- embedded **crisis windows** with elevated volatility and negative drift:
  - 2015-08 China devaluation
  - 2018 Q4 selloff
  - 2020-02/03 COVID crash (+ recovery)
  - 2022 rate-hike bear market

Crisis dates are chosen so the historical stress-testing scenarios that fall
inside the 2015–2024 window can be replayed on the sample portfolios. (Pre-2015
crises such as 1987, 1998, 2008 are reported as "outside the sample data
window".)

## Regenerating

```bash
python -m data.sample_portfolios      # rewrites both CSVs deterministically
```

Or from Python:

```python
from data.sample_portfolios import write_sample_csvs
write_sample_csvs()
```

The generator is seeded (`DEFAULT_SEED = 42`), so regeneration reproduces the
committed files exactly.
