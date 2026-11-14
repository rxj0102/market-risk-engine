# Market Risk Engine

[![CI](https://github.com/rxj0102/market-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/market-risk-engine/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-FF4B4B.svg)](app.py)

> A market-risk platform: **Historical, Parametric, and Monte Carlo VaR** with **Expected Shortfall**, **Basel III/IV backtesting**, **stress testing**, **risk decomposition**, and **economic-capital allocation**; packaged as a Python library, a nine-tab Streamlit dashboard, and an SSRN-ready research paper.

---

## Highlights

| Domain | What's inside |
|--------|---------------|
| **Value-at-Risk** | Historical simulation (plain, age-weighted BRW, vol-scaled), Parametric Delta-Normal & Student-t, Monte Carlo (Normal / t / copula, antithetic, path-dependent), Filtered Historical Simulation (Hull–White) |
| **Expected Shortfall** | Historical CVaR, Parametric (Normal, Student-t, Cornish–Fisher), Monte Carlo ES with bootstrap CIs |
| **Backtesting** | Kupiec POF, Christoffersen independence & conditional coverage, Berkowitz LR, Basel traffic light, vectorised rolling engine |
| **Stress testing** | 10-crisis historical replay (1987–2022), hypothetical macro shocks, correlation stress, reverse stress (factor-space + Monte Carlo), tornado sensitivities |
| **Risk decomposition** | Component & Marginal VaR (exactly additive), PCA factor attribution, systematic vs idiosyncratic split, correlation-risk contribution |
| **Capital** | Economic capital (99.9% / 1Y), Basel III IMA vs Basel IV FRTB ES with liquidity horizons, RAROC / RAPM, capital-efficient allocation |
| **Models** | GARCH(1,1) MLE, EWMA & DCC-GARCH covariance, Ledoit–Wolf shrinkage, Gaussian & Student-t copulas, two-regime Markov switching |
| **Diagnostics** | Distribution & normality tests, EVT (POT/GPD) tail fitting, rolling/DCC correlation dynamics, VaR model ranking |

---

## Quickstart

```bash
git clone https://github.com/rxj0102/market-risk-engine.git
cd market-risk-engine
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run app.py
```

### Use the library

```python
from data import equity_portfolio
from var import VaREngine
from expected_shortfall import ESEngine
from backtesting import BacktestEngine

portfolio = equity_portfolio(notional=10_000_000)

# Run every VaR method at 99% / 1-day
results = VaREngine().compute_all(portfolio, confidence=0.99, horizon_days=1)
for name, r in results.items():
    print(f"{r.method:24s} {r.var:>14,.0f}  ({r.var_pct:.2%})")

# Side-by-side VaR & ES table
print(VaREngine().comparison_table(portfolio, confidence=0.99))

# Basel backtest with traffic-light classification
result, frame = BacktestEngine().run(portfolio, "historical", confidence=0.99)
print(result)   # -> Backtest[GREEN] 5/500 exceptions ...
```

### Generate the research paper

```bash
python paper/generate_paper.py      # -> paper/market_risk_engine.pdf
```

### Run the tests

```bash
pytest -q                            # 35 tests
```

---

## Architecture

```
market-risk-engine/
├── core.py                 # Portfolio / VaRResult / ESResult / BacktestResult / ScenarioResult + constants
├── app.py                  # 9-tab Streamlit dashboard
├── var/                    # Historical · Parametric · Monte Carlo · FHS · VaREngine
├── expected_shortfall/     # Historical · Parametric · Monte Carlo ES · ESEngine
├── backtesting/            # Kupiec · Christoffersen · Berkowitz · Traffic light · Engine
├── stress_testing/         # Historical · Hypothetical · Reverse · Sensitivity · Engine
├── risk_decomposition/     # Component · Marginal · Factor (PCA) · Correlation risk
├── capital/                # Economic capital · Basel III/IV · RAPM · Engine
├── models/                 # Covariance (sample/EWMA/DCC/LW) · GARCH · Copulas · Regime
├── data/                   # Loader · Returns · Sample portfolios + CSVs
├── diagnostics/            # Distribution · Tail (EVT) · Correlation dynamics · Model comparison
├── utils/                  # Plotly chart builders · statistics helpers
├── tests/                  # pytest suite
├── notebooks/              # market_risk_deep_dive.ipynb
└── paper/                  # generate_paper.py (ReportLab)
```

Every module shares the lightweight dataclasses in `core.py`. The `Portfolio`
object carries `weights`, `asset_names`, a `returns` panel (dates × assets),
and `notional`; its `.pnl()` and `.return_series()` helpers feed the entire
engine.

---

## Methodology notes

- **Sign convention.** VaR and ES are reported as **positive dollar losses**;
  `var_pct = var / notional`. Multi-day horizons use √-time scaling unless a
  method simulates the horizon directly.
- **Vectorised & seeded.** VaR computations avoid per-day Python loops; EWMA
  covariance uses the recursive RiskMetrics update; all Monte Carlo is seeded
  and offers antithetic variance reduction.
- **Coherence.** Expected Shortfall is subadditive (a coherent risk measure),
  motivating the Basel IV shift from 99% VaR to 97.5% ES.
- **Basel traffic light** (250-day, 99%): green 0–4 (×3.00), yellow 5–9
  (×3.40–3.85), red 10+ (×4.00).

See **Tab 9: Documentation** in the dashboard and `paper/` for the full
mathematical treatment and references (Kupiec 1995, Christoffersen 1998,
Hull–White 1998, Engle 2002, Acerbi–Tasche 2002, BCBS FRTB 2019).

---

## Sample data

Two deterministic, synthetically-generated panels ship with the repo (see
[`data/README_data.md`](data/README_data.md)):

- **`sample_equity_portfolio.csv`**: 10 ETFs (SPY, QQQ, IWM, GLD, TLT, HYG,
  EEM, USO, VNQ, XLF), business-daily 2015–2024, with embedded crisis windows
  (2015 China devaluation, 2018 Q4, 2020 COVID, 2022 bear) and fat-tailed,
  jump-driven returns.
- **`sample_fx_portfolio.csv`**: 5 currency pairs (EURUSD, GBPUSD, USDJPY,
  AUDUSD, USDMXN).

> The sample data is **simulated for demonstration** and is not real market
> data. Upload your own returns CSV in the dashboard's sidebar to analyse a
> live portfolio.

---

## Reproducibility

### Data availability

Inputs ship in the `data/` folder of this repository. No external downloads are needed.

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `requirements.lock` (compiled from `requirements.txt` with uv).
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock   # exact pinned environment
pip install -e . --no-deps         # this package
pytest -q                          # runs the test suite
```

### Runtime

The test suite (37 tests) finishes in about 9 seconds on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## Disclaimer

This project is for **research and educational purposes**. It is not
investment advice and the bundled data is synthetic. Validate independently
before any production or regulatory use.

## License

[MIT](LICENSE)

## Testing and Basel zones

- CI runs `pytest` on every push and pull request.
- The Basel traffic-light zones are now computed from the cumulative binomial probability of the exception count (green below 95%, red from 99.99%), which reproduces the published 0-4 / 5-9 / 10+ table exactly for 250 days at 99%. Previously other window lengths scaled the 250-day counts, and `backtest_report` reported a plus-factor from the unscaled count, which could disagree with the zone and multiplier it returned for the same backtest.
