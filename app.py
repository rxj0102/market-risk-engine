"""Market Risk Engine — Streamlit dashboard.

A nine-tab institutional market-risk cockpit covering portfolio setup, VaR,
Expected Shortfall, Basel backtesting, stress testing, risk decomposition,
capital allocation, model analytics and documentation.

Run with::

    streamlit run app.py

All heavy computations are cached via ``st.cache_data``. Because
:class:`core.Portfolio` is not hashable, cached helpers accept the raw returns
DataFrame, a weights tuple and the notional, and rebuild the portfolio
internally.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from core import BASEL_CONFIDENCE, TRADING_DAYS, Portfolio
from data import (
    MarketDataLoader,
    SAMPLE_PORTFOLIOS,
    equity_portfolio,
    fx_portfolio,
    load_equity_returns,
    load_fx_returns,
)
from data.loader import MarketDataLoader as _Loader
from var import (
    FilteredHistoricalSimulation,
    HistoricalVaR,
    MonteCarloVaR,
    ParametricVaR,
    VaREngine,
)
from expected_shortfall import ESEngine, HistoricalES, MCExpectedShortfall, ParametricES
from backtesting import (
    SUPPORTED_METHODS,
    BacktestEngine,
    BerkowitzTest,
    ChristoffersenTest,
    KupiecTest,
)
from stress_testing import (
    HistoricalScenarioEngine,
    HypotheticalScenarioEngine,
    ReverseStressTester,
    SensitivityAnalyzer,
)
from risk_decomposition import (
    ComponentVaR,
    CorrelationRiskAnalyzer,
    FactorRiskDecomposition,
    MarginalVaR,
    default_factor_returns,
)
from capital import (
    BaselCapitalEngine,
    CapitalEngine,
    EconomicCapitalEngine,
    RiskAdjustedPerformance,
)
from diagnostics import (
    CorrelationDynamicsAnalyzer,
    ReturnDistributionAnalyzer,
    TailAnalyzer,
    VaRModelComparison,
)
from models import GARCHModel, RegimeSwitchingModel
from utils import plotting
from utils.statistics import max_drawdown, sharpe_ratio, summary_statistics

# ---------------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Market Risk Engine",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1b2735 0%, #111722 100%);
        border: 1px solid #2a3441; border-radius: 10px; padding: 14px 16px;
    }
    .tl-badge { font-size: 26px; font-weight: 700; padding: 10px 22px;
        border-radius: 8px; display: inline-block; color: #0E1117; }
    h1, h2, h3 { color: #FAFAFA; }
    </style>
    """,
    unsafe_allow_html=True,
)

TEMPLATE = "plotly_dark"
ACCENT = "#00CC96"
DANGER = "#EF553B"
WARN = "#FFA15A"


# ---------------------------------------------------------------------------
# Cached compute helpers (accept hashable primitives, rebuild Portfolio)
# ---------------------------------------------------------------------------
def _portfolio(returns: pd.DataFrame, weights: tuple, notional: float, currency: str = "USD") -> Portfolio:
    return Portfolio(np.asarray(weights, dtype=float), list(returns.columns), returns, float(notional), currency)


@st.cache_data(show_spinner=False)
def cached_var_all(returns, weights, notional, confidence, horizon):
    p = _portfolio(returns, weights, notional)
    return VaREngine().compute_all(p, confidence=confidence, horizon_days=horizon)


@st.cache_data(show_spinner=False)
def cached_var_table(returns, weights, notional, confidence):
    p = _portfolio(returns, weights, notional)
    return VaREngine().comparison_table(p, confidence=confidence)


@st.cache_data(show_spinner=False)
def cached_es_all(returns, weights, notional, confidence):
    p = _portfolio(returns, weights, notional)
    return ESEngine().compute_all(p, confidence=confidence)


@st.cache_data(show_spinner=False)
def cached_rolling_var(returns, weights, notional, confidence, window):
    p = _portfolio(returns, weights, notional)
    return HistoricalVaR().rolling_var(p, confidence=confidence, window=window)


@st.cache_data(show_spinner=False)
def cached_backtest(returns, weights, notional, method, confidence, test_window, est_window):
    p = _portfolio(returns, weights, notional)
    res, frame = BacktestEngine().run(p, method, confidence, test_window, est_window)
    return res, frame


@st.cache_data(show_spinner=False)
def cached_backtest_all(returns, weights, notional, confidence, test_window, est_window):
    p = _portfolio(returns, weights, notional)
    return BacktestEngine().run_all_methods(p, confidence, test_window, est_window)


@st.cache_data(show_spinner=False)
def cached_component_var(returns, weights, notional, confidence, cov_method):
    p = _portfolio(returns, weights, notional)
    return ComponentVaR().compute(p, confidence=confidence, cov_method=cov_method)


@st.cache_data(show_spinner=False)
def cached_capital_summary(returns, weights, notional):
    p = _portfolio(returns, weights, notional)
    return CapitalEngine().summary_table(p)


def fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


# ---------------------------------------------------------------------------
# Sidebar — portfolio builder
# ---------------------------------------------------------------------------
st.sidebar.title("📉 Market Risk Engine")
st.sidebar.caption("Historical · Parametric · Monte Carlo VaR · ES · Basel III/IV")

source = st.sidebar.radio("Portfolio source", ["Sample Equity (10 ETFs)", "Sample FX (5 pairs)", "Upload CSV"])

returns_df = None
if source == "Upload CSV":
    up = st.sidebar.file_uploader("Returns CSV (date index + asset columns)", type=["csv"])
    if up is not None:
        try:
            returns_df = _Loader.load_returns_csv(up)
            st.sidebar.success(f"Loaded {returns_df.shape[1]} assets, {returns_df.shape[0]} rows")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Could not parse CSV: {exc}")
    if returns_df is None:
        returns_df = load_equity_returns()
        st.sidebar.info("Using sample equity data until a valid CSV is uploaded.")
elif source == "Sample FX (5 pairs)":
    returns_df = load_fx_returns()
else:
    returns_df = load_equity_returns()

asset_names = list(returns_df.columns)
n_assets = len(asset_names)

notional = st.sidebar.number_input(
    "Notional (USD)", min_value=100_000.0, max_value=1e10, value=10_000_000.0, step=1_000_000.0, format="%.0f"
)

weight_mode = st.sidebar.radio("Weighting", ["Equal weight", "Minimum variance", "Manual"])
if weight_mode == "Equal weight":
    weights = np.full(n_assets, 1.0 / n_assets)
elif weight_mode == "Minimum variance":
    try:
        weights = _Loader.minimum_variance_weights(returns_df)
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum() if weights.sum() else np.full(n_assets, 1.0 / n_assets)
    except Exception:  # noqa: BLE001
        weights = np.full(n_assets, 1.0 / n_assets)
else:
    st.sidebar.caption("Set raw weights (normalised automatically)")
    raw = [st.sidebar.slider(a, 0.0, 1.0, 1.0 / n_assets, 0.01) for a in asset_names]
    raw = np.array(raw)
    weights = raw / raw.sum() if raw.sum() else np.full(n_assets, 1.0 / n_assets)

global_conf = st.sidebar.selectbox("Confidence", [0.95, 0.99, 0.999], index=1, format_func=lambda x: f"{x:.1%}")
global_horizon = st.sidebar.selectbox("Horizon (days)", [1, 5, 10, 20], index=0)

weights_t = tuple(float(w) for w in weights)
portfolio = _portfolio(returns_df, weights_t, notional)

st.sidebar.metric("Assets", n_assets)
st.sidebar.metric("Largest weight", f"{weights.max():.1%}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(
    [
        "🧱 Portfolio",
        "📊 VaR",
        "🪙 Expected Shortfall",
        "🚦 Backtesting",
        "💥 Stress Testing",
        "🧩 Risk Decomposition",
        "🏦 Capital",
        "🔬 Model Analytics",
        "📚 Documentation",
    ]
)

# ======================================================================= Tab 1
with tabs[0]:
    st.header("Portfolio Setup & Overview")
    port_ret = portfolio.return_series()
    bench_ret = returns_df.mean(axis=1)

    c1, c2, c3, c4 = st.columns(4)
    stats = summary_statistics(port_ret)
    c1.metric("Ann. return", f"{port_ret.mean() * TRADING_DAYS:.2%}")
    c2.metric("Ann. volatility", f"{stats['std'] * np.sqrt(TRADING_DAYS):.2%}")
    c3.metric("Sharpe", f"{sharpe_ratio(port_ret):.2f}")
    c4.metric("Max drawdown", f"{max_drawdown(port_ret):.2%}")

    left, right = st.columns(2)
    with left:
        cum = pd.DataFrame(
            {"Portfolio": (1 + port_ret).cumprod(), "Equal-weight benchmark": (1 + bench_ret).cumprod()}
        )
        fig = plotting.line_chart(cum, "Cumulative Growth of $1", "Wealth")
        st.plotly_chart(fig, use_container_width=True)

        roll_vol = port_ret.rolling(30).std() * np.sqrt(TRADING_DAYS)
        st.plotly_chart(
            plotting.line_chart(roll_vol.to_frame("30d ann. vol"), "Rolling 30-Day Volatility", "Vol"),
            use_container_width=True,
        )
    with right:
        corr = returns_df.corr()
        st.plotly_chart(plotting.heatmap(corr, "Correlation Matrix", zmid=0), use_container_width=True)

        hist = go.Figure()
        hist.add_trace(go.Histogram(x=port_ret, nbinsx=80, histnorm="probability density",
                                    marker_color=ACCENT, opacity=0.7, name="Returns"))
        xs = np.linspace(port_ret.min(), port_ret.max(), 200)
        pdf = (1 / (stats["std"] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - stats["mean"]) / stats["std"]) ** 2)
        hist.add_trace(go.Scatter(x=xs, y=pdf, mode="lines", line_color=WARN, name="Normal fit"))
        hist.update_layout(template=TEMPLATE, title="Return Distribution vs Normal", height=420)
        st.plotly_chart(hist, use_container_width=True)

    st.subheader("Summary Statistics")
    summ = pd.DataFrame(
        {
            "Mean (daily)": [stats["mean"]],
            "Vol (daily)": [stats["std"]],
            "Skewness": [stats["skewness"]],
            "Excess kurtosis": [stats["excess_kurtosis"]],
            "Sharpe (ann.)": [sharpe_ratio(port_ret)],
            "Max drawdown": [max_drawdown(port_ret)],
            "Jarque-Bera p": [stats["jb_pvalue"]],
        }
    ).T.rename(columns={0: "Value"})
    st.dataframe(summ.style.format("{:.4f}"), use_container_width=True)

# ======================================================================= Tab 2
with tabs[1]:
    st.header("Value-at-Risk Dashboard")
    with st.expander("Parameters", expanded=True):
        cc1, cc2 = st.columns(2)
        conf = cc1.selectbox("Confidence", [0.95, 0.99, 0.999],
                             index=[0.95, 0.99, 0.999].index(global_conf), format_func=lambda x: f"{x:.1%}", key="var_conf")
        horizon = cc2.selectbox("Horizon (days)", [1, 5, 10, 20], index=[1, 5, 10, 20].index(global_horizon), key="var_h")

    if st.button("▶ Run all VaR methods", type="primary"):
        st.session_state["run_var"] = True

    if st.session_state.get("run_var"):
        with st.spinner("Computing VaR across all methods…"):
            results = cached_var_all(returns_df, weights_t, notional, conf, horizon)

        labels = {
            "historical": "Historical", "age_weighted": "Age-Weighted", "fhs": "FHS",
            "parametric_normal": "Parametric-Normal", "parametric_t": "Parametric-t",
            "monte_carlo_normal": "Monte Carlo",
        }
        cols = st.columns(3)
        for i, (key, lab) in enumerate(labels.items()):
            r = results.get(key)
            if r is not None:
                cols[i % 3].metric(lab, fmt_usd(r.var), f"{r.var_pct:.2%} of notional")

        names = [results[k].method for k in results]
        vals = [results[k].var for k in results]
        order = np.argsort(vals)
        st.plotly_chart(
            plotting.bar_chart([names[i] for i in order], [vals[i] for i in order], "VaR by Method (sorted)"),
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            try:
                ts = HistoricalVaR().compute_multi_horizon(portfolio, confidence=conf, horizons=[1, 5, 10, 20])
                st.plotly_chart(plotting.line_chart(ts, "VaR Term Structure (Historical)", "VaR ($)"),
                                use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.info(f"Term structure unavailable: {exc}")
        with col_b:
            confs = np.array([0.90, 0.95, 0.975, 0.99, 0.995, 0.999])
            pv = ParametricVaR()
            sweep = pd.DataFrame(
                {"Parametric-Normal": [pv.compute(portfolio, confidence=float(c)).var for c in confs]},
                index=[f"{c:.1%}" for c in confs],
            )
            st.plotly_chart(plotting.line_chart(sweep, "VaR vs Confidence", "VaR ($)"), use_container_width=True)

        # P&L distribution with VaR/ES lines
        pnl = portfolio.pnl().to_numpy()
        var_hist = results["historical"].var
        try:
            es_hist = HistoricalES().compute(portfolio, confidence=conf).es
        except Exception:  # noqa: BLE001
            es_hist = None
        st.plotly_chart(plotting.histogram_with_var(pnl, var_hist, es_hist, "Daily P&L Distribution"),
                        use_container_width=True)

        with st.spinner("Rolling VaR…"):
            rv = cached_rolling_var(returns_df, weights_t, notional, conf, 250)
        st.plotly_chart(plotting.line_chart(rv.to_frame("Rolling Historical VaR"), "Rolling 250-Day VaR", "VaR ($)"),
                        use_container_width=True)

        st.subheader("Comparison Table (VaR & ES)")
        st.dataframe(cached_var_table(returns_df, weights_t, notional, conf), use_container_width=True)
    else:
        st.info("Set parameters and click **Run all VaR methods**.")

# ======================================================================= Tab 3
with tabs[2]:
    st.header("Expected Shortfall (CVaR)")
    conf_es = st.selectbox("Confidence", [0.95, 0.99, 0.999],
                           index=[0.95, 0.99, 0.999].index(global_conf), format_func=lambda x: f"{x:.1%}", key="es_conf")
    with st.spinner("Computing Expected Shortfall…"):
        es_results = cached_es_all(returns_df, weights_t, notional, conf_es)

    cols = st.columns(len(es_results))
    for col, (key, r) in zip(cols, es_results.items()):
        col.metric(r.method, fmt_usd(r.es), f"ES/VaR {r.es_var_ratio:.3f}")

    ratio_rows = [{"Method": r.method, "VaR ($)": r.var, "ES ($)": r.es, "ES/VaR": r.es_var_ratio}
                  for r in es_results.values()]
    st.subheader("ES / VaR Ratio")
    st.dataframe(pd.DataFrame(ratio_rows).style.format(
        {"VaR ($)": "{:,.0f}", "ES ($)": "{:,.0f}", "ES/VaR": "{:.3f}"}), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        n_tail = st.slider("Worst N days", 5, 50, 15)
        try:
            tail = HistoricalES().tail_scenarios(portfolio, confidence=conf_es, lookback=len(returns_df))
            st.dataframe(tail.head(n_tail), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Tail scenarios unavailable: {exc}")
    with c2:
        confs = np.array([0.90, 0.95, 0.975, 0.99, 0.995, 0.999])
        pe = ParametricES()
        es_term = pd.DataFrame(
            {"Parametric-Normal ES": [pe.compute(portfolio, confidence=float(c), distribution="normal").es for c in confs]},
            index=[f"{c:.1%}" for c in confs],
        )
        st.plotly_chart(plotting.line_chart(es_term, "ES vs Confidence", "ES ($)"), use_container_width=True)

    st.subheader("Cornish-Fisher vs Normal ES")
    try:
        cf = ParametricES().cornish_fisher_es(portfolio, confidence=conf_es).es
        nm = ParametricES().compute(portfolio, confidence=conf_es, distribution="normal").es
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Normal ES", fmt_usd(nm))
        cc2.metric("Cornish-Fisher ES", fmt_usd(cf))
        cc3.metric("Skew/kurtosis uplift", f"{(cf / nm - 1):.1%}" if nm else "n/a")
    except Exception as exc:  # noqa: BLE001
        st.info(f"Cornish-Fisher ES unavailable: {exc}")

# ======================================================================= Tab 4
with tabs[3]:
    st.header("Basel Backtesting")
    bc1, bc2 = st.columns(2)
    method = bc1.selectbox("VaR method", list(SUPPORTED_METHODS))
    test_window = bc2.selectbox("Backtest window", [250, 500, 1000], index=1)
    conf_bt = st.selectbox("Confidence", [0.95, 0.99], index=1, format_func=lambda x: f"{x:.1%}", key="bt_conf")
    est_window = 500

    with st.spinner("Running rolling backtest…"):
        res, frame = cached_backtest(returns_df, weights_t, notional, method, conf_bt, test_window, est_window)

    zone_color = {"green": "#00CC96", "yellow": "#FFD166", "red": "#EF553B"}[res.traffic_light]
    st.markdown(
        f'<span class="tl-badge" style="background:{zone_color};">{res.traffic_light.upper()} ZONE</span>',
        unsafe_allow_html=True,
    )
    m = st.columns(6)
    m[0].metric("Exceptions", res.exceptions)
    m[1].metric("Expected", f"{res.expected_exceptions:.1f}")
    m[2].metric("Exception rate", f"{res.exception_rate:.2%}")
    m[3].metric("Kupiec p", f"{res.kupiec_pvalue:.3f}")
    m[4].metric("CC p", f"{res.christoffersen_pvalue:.3f}")
    m[5].metric("Capital mult.", f"{res.capital_multiplier:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["pnl"], mode="lines", name="P&L", line_color="#636EFA"))
    fig.add_trace(go.Scatter(x=frame.index, y=-frame["var"], mode="lines", name="-VaR", line_color=WARN))
    exc = frame[frame["exception"] == 1]
    fig.add_trace(go.Scatter(x=exc.index, y=exc["pnl"], mode="markers", name="Exceptions",
                             marker=dict(color=DANGER, size=7, symbol="x")))
    fig.update_layout(template=TEMPLATE, title="P&L vs VaR with Exceptions", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical Tests")
    kup = KupiecTest().test(frame["var"], frame["pnl"], conf_bt)
    cc = ChristoffersenTest().conditional_coverage_test(frame["var"], frame["pnl"], conf_bt)
    bz = BerkowitzTest().test(frame["var"], frame["pnl"], conf_bt)
    tests = pd.DataFrame(
        {
            "Test": ["Kupiec POF", "Christoffersen CC", "Berkowitz LR"],
            "Statistic": [kup[0], cc[0], bz[0]],
            "p-value": [kup[1], cc[1], bz[1]],
            "Result": ["REJECT" if kup[2] else "PASS", "REJECT" if cc[2] else "PASS", "REJECT" if bz[2] else "PASS"],
        }
    )
    st.dataframe(tests.style.format({"Statistic": "{:.3f}", "p-value": "{:.4f}"}), use_container_width=True)

    st.subheader("Method Comparison")
    st.dataframe(cached_backtest_all(returns_df, weights_t, notional, conf_bt, test_window, est_window),
                 use_container_width=True)

# ======================================================================= Tab 5
with tabs[4]:
    st.header("Stress Testing")
    sub = st.tabs(["Historical Crisis", "Hypothetical", "Reverse Stress", "Sensitivity"])

    with sub[0]:
        hist_eng = HistoricalScenarioEngine()
        try:
            tbl = hist_eng.run_all(portfolio)
            st.dataframe(tbl, use_container_width=True)
            loss_col = next((c for c in tbl.columns if "Loss" in c and "%" in c), None)
            name_col = tbl.columns[0]
            if loss_col:
                sub_t = tbl.dropna(subset=[loss_col])
                st.plotly_chart(
                    plotting.bar_chart(list(sub_t[name_col]), sub_t[loss_col].to_numpy(),
                                       "Scenario Loss (% of notional)", color=DANGER),
                    use_container_width=True,
                )
        except Exception as exc:  # noqa: BLE001
            st.info(f"Historical scenarios unavailable for this portfolio: {exc}")

    with sub[1]:
        hyp = HypotheticalScenarioEngine()
        try:
            st.dataframe(hyp.run_all_hypothetical(portfolio), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Pre-built scenarios not applicable: {exc}")
        st.subheader("Custom shock builder")
        cols = st.columns(min(5, n_assets))
        shocks = {}
        for i, a in enumerate(asset_names):
            shocks[a] = cols[i % len(cols)].slider(a, -0.5, 0.5, 0.0, 0.01, key=f"shock_{a}")
        try:
            sc = hyp.apply_custom_shock(portfolio, shocks)
            st.metric("Custom shock P&L", fmt_usd(sc.pnl), f"{sc.pnl_pct:.2%}")
        except Exception as exc:  # noqa: BLE001
            st.info(f"Custom shock failed: {exc}")
        try:
            cs = hyp.apply_correlation_shock(portfolio, correlation_floor=0.7)
            st.metric("Correlation-shock impact (floor 0.7)", fmt_usd(cs.pnl), f"{cs.pnl_pct:.2%}")
        except Exception as exc:  # noqa: BLE001
            st.info(f"Correlation shock unavailable: {exc}")

    with sub[2]:
        target = st.slider("Target loss threshold (% of notional)", 1, 50, 10) / 100.0
        rst = ReverseStressTester()
        try:
            out = rst.minimum_stress_scenario(portfolio, target)
            st.write("**Minimum-norm breaking scenario (factor shocks):**")
            fshocks = out.get("factor_shocks", {})
            st.dataframe(pd.DataFrame({"Factor": list(fshocks.keys()), "Shock": list(fshocks.values())}),
                         use_container_width=True)
            prob = rst.probability_of_break(portfolio, target, model="normal")
            st.metric("Daily probability of breach", f"{prob:.4%}")
        except Exception as exc:  # noqa: BLE001
            st.info(f"Reverse stress unavailable: {exc}")

    with sub[3]:
        sa = SensitivityAnalyzer()
        try:
            tor = sa.tornado_chart_data(portfolio)
            st.dataframe(tor, use_container_width=True)
            cols = list(tor.columns)
            fac = cols[0]
            down = next((c for c in cols if c.lower() in ("down", "-1sd", "low")), cols[1])
            up = next((c for c in cols if c.lower() in ("up", "+1sd", "high")), cols[2])
            st.plotly_chart(plotting.tornado(list(tor[fac]), tor[down].to_numpy(), tor[up].to_numpy(),
                                             "Risk Factor Tornado"), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Sensitivity analysis unavailable: {exc}")
        try:
            eq = sa.equity_sensitivity(portfolio)
            xcol, ycol = eq.columns[0], eq.columns[1]
            st.plotly_chart(plotting.line_chart(eq.set_index(xcol), "Equity Shock Sweep", "P&L ($)"),
                            use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Equity sweep unavailable: {exc}")

# ======================================================================= Tab 6
with tabs[5]:
    st.header("Risk Decomposition")
    cov_method = st.selectbox("Covariance estimator", ["ewma", "sample", "ledoit_wolf"], index=0)
    cvar = cached_component_var(returns_df, weights_t, notional, global_conf, cov_method)
    comp_col = next((c for c in cvar.columns if "ComponentVaR" in c and "$" in c), None)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(cvar, use_container_width=True)
    with c2:
        try:
            div = ComponentVaR().diversification_ratio(portfolio, cov_method=cov_method)
        except Exception:  # noqa: BLE001
            div = ComponentVaR().diversification_ratio(portfolio)
        st.metric("Diversification ratio", f"{div:.2%}")
        try:
            hhi = ComponentVaR().risk_concentration(portfolio, confidence=global_conf)
            st.metric("Risk concentration (HHI)", f"{hhi:.3f}")
        except Exception:  # noqa: BLE001
            pass

    if comp_col:
        labels = list(cvar[cvar.columns[0]]) + ["Total"]
        vals = list(cvar[comp_col]) + [0]
        st.plotly_chart(plotting.waterfall(labels, vals, "Component VaR Waterfall"), use_container_width=True)

    mv = MarginalVaR().marginal_var(portfolio, confidence=global_conf, cov_method=cov_method)
    st.plotly_chart(plotting.bar_chart(asset_names, mv, "Marginal VaR (positive = risk-adder)"),
                    use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        pca = FactorRiskDecomposition().pca_decomposition(portfolio, n_factors=min(5, n_assets))
        st.plotly_chart(plotting.scree_plot(np.asarray(pca["variance_explained"]), "PCA Scree Plot"),
                        use_container_width=True)
    with c4:
        try:
            fac_ret = default_factor_returns(portfolio)
            sysv, idiov = FactorRiskDecomposition().systematic_vs_idiosyncratic(portfolio, fac_ret)
            pie = go.Figure(go.Pie(labels=["Systematic", "Idiosyncratic"], values=[sysv, idiov], hole=0.5))
            pie.update_layout(template=TEMPLATE, title="Systematic vs Idiosyncratic Risk", height=380)
            st.plotly_chart(pie, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Factor split unavailable: {exc}")
    try:
        cr = CorrelationRiskAnalyzer().correlation_contribution(portfolio)
        st.metric("Risk from correlations", f"{cr:.1%}")
    except Exception:  # noqa: BLE001
        pass

# ======================================================================= Tab 7
with tabs[6]:
    st.header("Capital Allocation")
    eco = EconomicCapitalEngine()
    with st.spinner("Computing economic & regulatory capital…"):
        ec = eco.economic_capital(portfolio)
        divb = eco.diversification_benefit(portfolio)
    c = st.columns(3)
    c[0].metric("Economic Capital (99.9%, 1Y)", fmt_usd(ec))
    c[1].metric("EC / notional", f"{ec / notional:.1%}")
    c[2].metric("Diversification benefit", f"{divb:.1%}")

    st.subheader("Basel III vs Basel IV")
    try:
        st.dataframe(BaselCapitalEngine().capital_comparison(portfolio), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Capital comparison unavailable: {exc}")

    c5, c6 = st.columns(2)
    with c5:
        try:
            eca = eco.ec_by_asset(portfolio)
            val_col = next((c for c in eca.columns if "$" in c or "EC" in c), eca.columns[-1])
            st.plotly_chart(plotting.waterfall(list(eca[eca.columns[0]]), eca[val_col].to_numpy(),
                                               "Economic Capital by Asset"), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"EC by asset unavailable: {exc}")
    with c6:
        try:
            er = np.full(n_assets, 0.08)
            rap = RiskAdjustedPerformance()
            mr = rap.marginal_raroc(portfolio, er, confidence=0.999)
            fig = plotting.bar_chart(asset_names, np.asarray(mr) * 100, "Marginal RAROC by Asset (%)")
            fig.add_hline(y=12, line_color=WARN, annotation_text="12% hurdle")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"RAROC unavailable: {exc}")

    st.subheader("Optimal capital-efficient allocation")
    tr = st.slider("Target annual return", 0.0, 0.20, 0.06, 0.01)
    try:
        opt = RiskAdjustedPerformance().optimal_allocation(portfolio, np.full(n_assets, 0.08), tr, confidence=0.999)
        st.dataframe(pd.DataFrame({"Asset": asset_names, "Optimal weight": opt}).style.format({"Optimal weight": "{:.2%}"}),
                     use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Optimisation unavailable: {exc}")

# ======================================================================= Tab 8
with tabs[7]:
    st.header("Model Analytics")
    port_ret = portfolio.return_series()
    rda = ReturnDistributionAnalyzer()

    c1, c2 = st.columns(2)
    with c1:
        try:
            qq = rda.qq_plot_data(port_ret)
            tcol, ecol = qq.columns[0], qq.columns[1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=qq[tcol], y=qq[ecol], mode="markers", marker_color=ACCENT, name="Q-Q"))
            lim = [qq[tcol].min(), qq[tcol].max()]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line_color=WARN, name="45°"))
            fig.update_layout(template=TEMPLATE, title="Normal Q-Q Plot", height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Q-Q plot unavailable: {exc}")
    with c2:
        try:
            nt = rda.normality_tests(portfolio)
            st.dataframe(pd.DataFrame(nt).T if isinstance(nt, dict) else nt, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Normality tests unavailable: {exc}")

    st.subheader("GARCH conditional volatility")
    try:
        g = GARCHModel()
        params = g.fit(port_ret)
        cvols = g.conditional_vols(port_ret, params)
        df = pd.DataFrame(
            {"GARCH vol": cvols, "Rolling 30d realized": port_ret.rolling(30).std().to_numpy()},
            index=port_ret.index,
        )
        st.plotly_chart(plotting.line_chart(df, f"GARCH(1,1) — persistence {params.persistence:.3f}", "Daily vol"),
                        use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"GARCH unavailable: {exc}")

    st.subheader("Volatility regime probability")
    try:
        rs = RegimeSwitchingModel(max_iter=40)
        rp = rs.fit(port_ret)
        reg = pd.Series(rp.smoothed_probabilities, index=port_ret.index, name="P(stressed)")
        st.plotly_chart(plotting.line_chart(reg.to_frame(), "Smoothed Stressed-Regime Probability", "Probability"),
                        use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Regime model unavailable: {exc}")

    st.subheader("Extreme Value Theory (POT / GPD)")
    try:
        ta = TailAnalyzer()
        gpd = ta.fit_gpd(port_ret, threshold_quantile=0.95)
        evt_var = ta.evt_var(gpd["xi"], gpd["beta"], gpd["threshold"],
                             int(gpd["n_total"]), int(gpd["n_exceedances"]), 0.99)
        cc = st.columns(3)
        cc[0].metric("Tail index ξ", f"{gpd['xi']:.3f}")
        cc[1].metric("Scale β", f"{gpd['beta']:.4f}")
        cc[2].metric("EVT VaR(99%)", f"{evt_var:.2%}")
    except Exception as exc:  # noqa: BLE001
        st.info(f"EVT unavailable: {exc}")

    st.subheader("VaR model ranking")
    try:
        st.dataframe(VaRModelComparison().rank_models(portfolio, confidence=global_conf, window=250),
                     use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Model ranking unavailable: {exc}")

# ======================================================================= Tab 9
with tabs[8]:
    st.header("Documentation & Methodology")
    st.markdown(
        """
        ### From VaR to Expected Shortfall
        Basel II/2.5 relied on **99% 10-day VaR**. The 2008 crisis exposed VaR's
        blindness to tail severity and its lack of **subadditivity**. Basel IV /
        FRTB therefore adopts **97.5% Expected Shortfall**, a coherent risk
        measure that averages losses beyond the VaR quantile.

        ### Traffic-light backtesting (250-day window, 99%)
        | Zone | Exceptions | Action | Multiplier |
        |------|-----------|--------|-----------|
        | 🟢 Green | 0–4 | Model accepted | 3.00 |
        | 🟡 Yellow | 5–9 | Scrutiny + add-on | 3.40–3.85 |
        | 🔴 Red | 10+ | Model rejected | 4.00 |
        """
    )

    st.subheader("VaR Methods")
    st.dataframe(
        pd.DataFrame(
            {
                "Method": ["Historical", "Age-weighted", "Parametric-Normal", "Parametric-t", "Monte Carlo", "FHS"],
                "Assumption": ["None (empirical)", "Decay-weighted empirical", "Multivariate normal",
                               "Multivariate t", "Parametric / copula", "GARCH + empirical residuals"],
                "Captures fat tails": ["Yes", "Yes", "No", "Yes", "Depends", "Yes"],
                "Vol regime aware": ["No", "Partly", "EWMA/DCC", "EWMA/DCC", "Yes", "Yes"],
            }
        ),
        use_container_width=True,
    )

    st.subheader("Key formulae")
    st.latex(r"\text{VaR}_\alpha = z_\alpha \,\sigma_p\,\sqrt{h}\,N \qquad \sigma_p=\sqrt{w^\top\Sigma w}")
    st.latex(r"\text{ES}_\alpha = \frac{\phi(z_\alpha)}{1-\alpha}\,\sigma_p\,\sqrt{h}\,N")
    st.latex(r"LR_{POF} = -2\ln\frac{(1-p_0)^{T-x}p_0^{x}}{(1-\hat p)^{T-x}\hat p^{x}}\sim\chi^2_1")
    st.latex(r"LR_{CC} = LR_{POF} + LR_{IND}\sim\chi^2_2")
    st.latex(r"\text{CVaR}_i = w_i\,\frac{(\Sigma w)_i}{\sigma_p}\,z_\alpha\,N,\qquad \sum_i \text{CVaR}_i = \text{VaR}")
    st.latex(r"\text{RAROC} = \frac{\mathbb{E}[\text{P\&L}]}{\text{Economic Capital}}")

    st.subheader("References")
    st.markdown(
        """
        - Kupiec, P. (1995). *Techniques for verifying the accuracy of risk measurement models.* J. Derivatives.
        - Christoffersen, P. (1998). *Evaluating interval forecasts.* International Economic Review.
        - Hull, J. & White, A. (1998). *Incorporating volatility updating into the historical simulation method for VaR.* J. Risk.
        - Boudoukh, Richardson & Whitelaw (1998). *The best of both worlds.* Risk.
        - Engle, R. (2002). *Dynamic Conditional Correlation.* JBES.
        - Acerbi, C. & Tasche, D. (2002). *On the coherence of Expected Shortfall.* J. Banking & Finance.
        - BCBS (2019). *Minimum capital requirements for market risk* (FRTB).
        """
    )
