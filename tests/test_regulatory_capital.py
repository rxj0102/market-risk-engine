"""Basel 2.5 IMA and FRTB ES capital formulas."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from capital.regulatory_capital import BaselCapitalEngine
from core import Portfolio


def _portfolio(names, weights, n=900, seed=0):
    rng = np.random.default_rng(seed)
    k = len(names)
    corr = 0.5 * np.ones((k, k)) + 0.5 * np.eye(k)
    vols = np.linspace(0.008, 0.015, k)
    cov = corr * np.outer(vols, vols)
    rets = rng.multivariate_normal(np.zeros(k), cov, size=n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return Portfolio(
        weights=np.asarray(weights, dtype=float),
        asset_names=list(names),
        returns=pd.DataFrame(rets, index=idx, columns=list(names)),
        notional=1_000_000.0,
    )


def _normal_es_10d(p, mask, c=0.975):
    w = np.where(mask, p.weights, 0.0)
    s = np.sqrt(w @ p.returns.cov().to_numpy() @ w) * np.sqrt(10)
    return stats.norm.pdf(stats.norm.ppf(c)) / (1 - c) * s * p.notional


def test_frtb_all_ten_day_horizons_equal_base_es():
    p = _portfolio(["SPY", "QQQ", "TLT"], [0.4, 0.4, 0.2])
    eng = BaselCapitalEngine()
    base = eng.frtb_es_capital(p)
    assert base == pytest.approx(_normal_es_10d(p, np.ones(3, bool)), rel=1e-10)
    assert eng.frtb_capital(p)["total_charge"] == pytest.approx(base, rel=1e-10)


def test_frtb_cascade_matches_mar33_formula():
    p = _portfolio(["SPY", "IWM", "HYG", "XYZ"], [0.4, 0.3, 0.2, 0.1])
    eng = BaselCapitalEngine()
    lhs = np.array([eng._asset_lh(n) for n in p.asset_names])
    terms = [_normal_es_10d(p, np.ones(4, bool))]
    prev = 10
    for lh in (20, 40, 60, 120):
        terms.append(_normal_es_10d(p, lhs >= lh) * np.sqrt((lh - prev) / 10))
        prev = lh
    expected = np.sqrt(np.sum(np.square(terms)))
    got = eng.frtb_capital(p)["total_charge"]
    assert got == pytest.approx(expected, rel=1e-10)
    assert got > eng.frtb_es_capital(p)


def test_basel_charge_is_ten_day_and_multiplies_svar():
    p = _portfolio(["SPY", "QQQ"], [0.5, 0.5])
    eng = BaselCapitalEngine()
    out = eng.var_capital_charge(p)
    m = out["multiplier"]
    assert m >= 3.0
    svar_10d = eng._stressed_var(p, 0.99) * np.sqrt(10)
    assert out["svar_charge"] == pytest.approx(m * svar_10d, rel=1e-10)
    assert out["total"] == pytest.approx(out["var_charge"] + out["svar_charge"])


def test_rolling_var_uses_only_past_returns():
    p = _portfolio(["SPY", "QQQ"], [0.5, 0.5])
    eng = BaselCapitalEngine()
    v1 = eng._rolling_parametric_var(p, 0.99, window=60)
    shocked = p.returns.copy()
    shocked.iloc[-1] = -0.2
    p2 = Portfolio(weights=p.weights, asset_names=p.asset_names, returns=shocked, notional=p.notional)
    v2 = eng._rolling_parametric_var(p2, 0.99, window=60)
    assert v1.iloc[-1] == pytest.approx(v2.iloc[-1], rel=1e-12)
