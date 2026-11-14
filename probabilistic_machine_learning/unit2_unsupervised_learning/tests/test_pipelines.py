"""Tests for the Unit 2 pipelines and the helpers they add to hfp_tools.
Run from the unit folder:  pytest tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "historical_lab"))
sys.path.insert(0, str(ROOT / "pipelines"))
from hfp_tools import hfp_es, ks_2samp_hfp            # noqa: E402
from data import clean_prices                          # noqa: E402
import var_backtest as vb                              # noqa: E402


# ---------- hfp_tools additions ----------
def test_ks_matches_scipy_with_equal_weights():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=300), rng.normal(0.3, 1.2, size=200)
    d, p = ks_2samp_hfp(x, y)
    ref = ks_2samp(x, y, method="asymp")
    assert d == pytest.approx(ref.statistic)
    assert p == pytest.approx(ref.pvalue, rel=1e-6)


def test_ks_weights_reduce_evidence():
    """Same data, but decay weights -> smaller ens -> larger p-value."""
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=400), rng.normal(0.25, 1, size=400)
    p_decay = np.exp(np.linspace(-3, 0, 400)); p_decay /= p_decay.sum()
    assert ks_2samp_hfp(x, y, px=p_decay)[1] > ks_2samp_hfp(x, y)[1]


def test_hfp_es_small_example():
    eps = np.arange(1.0, 11.0)              # 1..10, equal weights
    p = np.full(10, 0.1)
    assert hfp_es(eps, p, 0.2) == pytest.approx(1.5)   # average of the worst 20%: 1 and 2


# ---------- data cleaning ----------
def test_clean_prices_removes_bad_rows():
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    df = pd.DataFrame({"close": [101, 100, 102, -5, np.nan], "volume": [1, 1, 1, 1, 0]}, index=idx)
    out = clean_prices(df)
    assert list(out.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert out.loc["2024-01-03", "close"] == 102          # duplicate date: last print kept


# ---------- VaR backtest ----------
def test_kupiec_accepts_exact_rate_and_rejects_bad_rate():
    hits = np.zeros(1000, bool); hits[::100] = True       # exactly 1%
    assert vb.kupiec_pof(hits, 0.01)[1] > 0.99
    hits[::20] = True                                      # 5%
    assert vb.kupiec_pof(hits, 0.01)[1] < 1e-6


def test_christoffersen_detects_clustering():
    scattered = np.zeros(1000, bool); scattered[::50] = True
    clustered = np.zeros(1000, bool)
    for s in range(0, 1000, 100):
        clustered[s:s + 2] = True                         # breaches in pairs
    assert vb.christoffersen_ind(scattered)[1] > 0.05
    assert vb.christoffersen_ind(clustered)[1] < 0.01


def test_basel_zones():
    assert [vb.basel_zone(b) for b in (0, 4, 5, 9, 10)] == ["green"] * 2 + ["yellow"] * 2 + ["red"]


def test_var_is_out_of_sample(monkeypatch):
    """Day t's VaR must use only returns before t: a crash ON day t can't be in its own VaR."""
    monkeypatch.setattr(vb, "WINDOW", 100)
    monkeypatch.setattr(vb, "METHODS", {"hist_equal": None})
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 101)
    r[100] = -0.5                                          # the crash, on the first forecast day
    res = vb.rolling_var_es(pd.Series(r, index=pd.bdate_range("2024-01-01", periods=101)))
    assert res["VaR_hist_equal"].iloc[0] < 0.05            # VaR built from calm days only
    assert bool(res["hit_hist_equal"].iloc[0])             # and the crash is a breach
