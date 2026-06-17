"""Reusable Plotly chart builders for the Streamlit dashboard.

All builders return ``plotly.graph_objects.Figure`` objects styled for a dark
theme. Plotly is imported lazily so the core library has no hard dependency on
it for headless / test usage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DARK_TEMPLATE = "plotly_dark"
ACCENT = "#00CC96"
DANGER = "#EF553B"
WARN = "#FFA15A"


def _go() -> Any:
    import plotly.graph_objects as go

    return go


def _apply_layout(fig: Any, title: str = "", height: int = 420) -> Any:
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def line_chart(df: pd.DataFrame, title: str = "", yaxis: str = "") -> Any:
    """Multi-series line chart from a DataFrame indexed by x."""
    go = _go()
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)))
    _apply_layout(fig, title)
    fig.update_yaxes(title_text=yaxis)
    return fig


def bar_chart(labels: list[str], values: np.ndarray, title: str = "", color: str = ACCENT) -> Any:
    """Simple bar chart."""
    go = _go()
    fig = go.Figure(go.Bar(x=list(labels), y=np.asarray(values), marker_color=color))
    _apply_layout(fig, title)
    return fig


def heatmap(matrix: pd.DataFrame, title: str = "", zmid: float | None = None) -> Any:
    """Correlation / covariance heatmap."""
    go = _go()
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="RdBu",
            zmid=zmid,
            colorbar=dict(title=""),
        )
    )
    _apply_layout(fig, title)
    return fig


def histogram_with_var(
    pnl: np.ndarray, var: float, es: float | None = None, title: str = "P&L Distribution"
) -> Any:
    """Histogram of P&L with VaR (and optional ES) lines marked."""
    go = _go()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl, nbinsx=80, marker_color=ACCENT, opacity=0.75, name="P&L"))
    fig.add_vline(x=-var, line_color=WARN, line_width=2, annotation_text="VaR")
    if es is not None:
        fig.add_vline(x=-es, line_color=DANGER, line_width=2, annotation_text="ES")
    _apply_layout(fig, title)
    return fig


def waterfall(labels: list[str], values: np.ndarray, title: str = "") -> Any:
    """Waterfall chart (e.g. component VaR build-up)."""
    go = _go()
    fig = go.Figure(
        go.Waterfall(
            x=list(labels),
            y=list(np.asarray(values)),
            connector={"line": {"color": "rgb(120,120,120)"}},
        )
    )
    _apply_layout(fig, title)
    return fig


def tornado(labels: list[str], low: np.ndarray, high: np.ndarray, title: str = "") -> Any:
    """Tornado chart of factor sensitivities sorted by impact."""
    go = _go()
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=np.asarray(low), orientation="h", name="-1 sd", marker_color=DANGER))
    fig.add_trace(go.Bar(y=labels, x=np.asarray(high), orientation="h", name="+1 sd", marker_color=ACCENT))
    fig.update_layout(barmode="overlay")
    _apply_layout(fig, title)
    return fig


def scree_plot(explained: np.ndarray, title: str = "PCA Scree Plot") -> Any:
    """Scree plot with cumulative variance explained."""
    go = _go()
    x = [f"PC{i+1}" for i in range(len(explained))]
    cum = np.cumsum(explained)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=explained, name="Explained", marker_color=ACCENT))
    fig.add_trace(go.Scatter(x=x, y=cum, name="Cumulative", mode="lines+markers", line_color=WARN))
    _apply_layout(fig, title)
    return fig
