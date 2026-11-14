"""
var_backtest.py
Pipeline 2: historical Value-at-Risk with flexible probabilities, plus a backtest.

Real-world scenario
-------------------
A risk desk reports, every evening, tomorrow's 1-day 99% VaR (and ES) for an S&P 500
position. Regulators then ask: how often were you breached, and did breaches come in
clusters? This pipeline runs that process day by day, out of sample, and compares:
    * equal-weight historical VaR   (classic "historical simulation", eq. 17.24)
    * exponential-decay HFP VaR     (recent days weigh more, eq. 17.29 with 17.6)

Theory used (Unit 2, Section 17.2)
----------------------------------
* Invariants: estimate on log returns, not prices (Glivenko-Cantelli needs i.i.d.).
* Plug-in principle (17.35): VaR is the c-quantile of the HFP distribution, ES is the
  probability-weighted average of the scenarios beyond it.
* ens(p) reports how many days each method effectively relies on: the price of
  reacting faster is a smaller effective sample.

Backtests
---------
* Kupiec POF test: is the breach RATE consistent with 1%?
* Christoffersen independence test: do breaches CLUSTER (a breach today makes one
  tomorrow more likely)? Clustering means the model reacts too slowly to new volatility.
* Basel traffic light: breaches over the latest 250 days (0-4 green, 5-9 yellow, 10+ red).

Stages: load -> clean -> invariants -> rolling out-of-sample VaR/ES -> breaches
        -> backtests -> report + plot
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "historical_lab"))
from hfp_tools import ens, exp_decay_probs, hfp_es, hfp_quantile  # noqa: E402
from data import OUTPUT_DIR, clean_prices, load_prices, log_returns  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "^GSPC"
START = "2005-01-01"
WINDOW = 500           # look-back window (about 2 years)
CONF = 0.99            # VaR confidence: the c = 1% left tail of returns
METHODS = {            # name -> half-life in days (None = equal weights)
    "hist_equal": None,
    "hfp_hl250": 250,
    "hfp_hl60": 60,
}
BASEL_DAYS = 250

OUT = OUTPUT_DIR / "var"
log = logging.getLogger("var_backtest")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def method_weights(half_life: int | None) -> np.ndarray:
    """Flexible probabilities for one look-back window, oldest first, most recent last."""
    if half_life is None:
        return np.full(WINDOW, 1 / WINDOW)
    return exp_decay_probs(WINDOW, half_life)


def rolling_var_es(ret: pd.Series) -> pd.DataFrame:
    """For each day t, estimate VaR/ES from the WINDOW returns BEFORE t only
    (out of sample: no peeking at day t), then record day t's actual return."""
    c = 1 - CONF
    windows = sliding_window_view(ret.to_numpy()[:-1], WINDOW)  # window k predicts ret[k + WINDOW]
    out = pd.DataFrame(index=ret.index[WINDOW:])
    out["ret"] = ret.to_numpy()[WINDOW:]
    for name, hl in METHODS.items():
        p = method_weights(hl)
        q = np.array([hfp_quantile(w, p, c) for w in windows])
        es = np.array([hfp_es(w, p, c) for w in windows])
        out[f"VaR_{name}"] = -q          # reported as positive losses
        out[f"ES_{name}"] = -es
        out[f"hit_{name}"] = out["ret"] < q
    return out


def kupiec_pof(hits: np.ndarray, c: float) -> tuple[float, float]:
    """Likelihood ratio for 'breach probability = c' (unconditional coverage)."""
    n, x = len(hits), int(hits.sum())
    pi = x / n
    ll0 = (n - x) * np.log(1 - c) + x * np.log(c)
    ll1 = (n - x) * np.log(1 - pi) + x * np.log(pi) if 0 < x < n else 0.0
    lr = -2 * (ll0 - ll1)
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Likelihood ratio for 'breaches are independent' (no clustering)."""
    h0, h1 = hits[:-1].astype(int), hits[1:].astype(int)
    n00 = np.sum((h0 == 0) & (h1 == 0)); n01 = np.sum((h0 == 0) & (h1 == 1))
    n10 = np.sum((h0 == 1) & (h1 == 0)); n11 = np.sum((h0 == 1) & (h1 == 1))

    def ll(k0, k1):  # log-likelihood of k1 ones and k0 zeros at their MLE rate
        n = k0 + k1
        if n == 0 or k0 == 0 or k1 == 0:
            return 0.0
        return k0 * np.log(k0 / n) + k1 * np.log(k1 / n)

    lr = -2 * (ll(n00 + n10, n01 + n11) - (ll(n00, n01) + ll(n10, n11)))
    return float(lr), float(chi2.sf(lr, 1))


def basel_zone(breaches: int) -> str:
    return "green" if breaches <= 4 else ("yellow" if breaches <= 9 else "red")


def backtest(res: pd.DataFrame) -> pd.DataFrame:
    c = 1 - CONF
    rows = []
    for name, hl in METHODS.items():
        hits = res[f"hit_{name}"].to_numpy()
        lr_pof, p_pof = kupiec_pof(hits, c)
        lr_ind, p_ind = christoffersen_ind(hits)
        last = int(hits[-BASEL_DAYS:].sum())
        rows.append({
            "method": name,
            "half_life": hl if hl else "equal",
            "ens(p)": round(ens(method_weights(hl))),
            "breaches": int(hits.sum()),
            "expected": round(c * len(hits), 1),
            "breach_rate_%": 100 * hits.mean(),
            "kupiec_p": p_pof,
            "christoffersen_p": p_ind,
            "conditional_cov_p": float(chi2.sf(lr_pof + lr_ind, 2)),
            f"breaches_last{BASEL_DAYS}": last,
            "basel_zone": basel_zone(last),
            "avg_VaR_%": 100 * res[f"VaR_{name}"].mean(),
            "avg_ES_%": 100 * res[f"ES_{name}"].mean(),
        })
    return pd.DataFrame(rows).set_index("method")


def conclusion(bt: pd.DataFrame) -> str:
    lines = []
    for name, r in bt.iterrows():
        verdict = []
        verdict.append("breach rate OK" if r["kupiec_p"] > 0.05 else "breach rate REJECTED")
        verdict.append("no clustering" if r["christoffersen_p"] > 0.05 else "breaches CLUSTER")
        lines.append(f"  {name:<11} ens={r['ens(p)']:<4} {r['breaches']} breaches "
                     f"(expected {r['expected']}) -> {', '.join(verdict)}; "
                     f"Basel {r['basel_zone']}")
    passing = bt[(bt["kupiec_p"] > 0.05) & (bt["christoffersen_p"] > 0.05)]
    best = (passing["avg_VaR_%"].idxmin() if len(passing) else
            bt["conditional_cov_p"].idxmax())
    lines.append(f"\n  Preferred: {best} "
                 + ("(passes both tests with the lowest average capital)"
                    if len(passing) else "(no method passes both; best conditional coverage)"))
    return "\n".join(lines)


def plot(res: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    loss_days = res["ret"] < 0
    ax.scatter(res.index[loss_days], -100 * res.loc[loss_days, "ret"], s=3,
               color="lightgrey", label="daily loss")
    colours = {"hist_equal": "tab:blue", "hfp_hl250": "tab:green", "hfp_hl60": "tab:red"}
    for name in METHODS:
        col = colours.get(name)
        ax.plot(res.index, 100 * res[f"VaR_{name}"], lw=1, color=col, label=f"VaR {name}")
        hit = res[f"hit_{name}"]
        ax.scatter(res.index[hit], -100 * res.loc[hit, "ret"], s=12, color=col, marker="x")
    ax.set_ylabel("loss (%)")
    ax.set_title(f"{TICKER} 1-day {CONF:.0%} VaR, {WINDOW}-day window (x = breach)")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "var_backtest.png", dpi=120)


def main(show: bool = True) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    ret = log_returns(clean_prices(load_prices(TICKER, START))["close"])
    res = rolling_var_es(ret)
    bt = backtest(res)

    res.to_csv(OUT / "var_series.csv")
    bt.to_csv(OUT / "var_backtest_summary.csv")
    pd.set_option("display.width", 160)
    print(f"\n=== Backtest: {len(res)} out-of-sample days, "
          f"{res.index[0].date()} to {res.index[-1].date()} ===")
    print(bt.round(3).to_string())
    print("\n=== Conclusion ===")
    print(conclusion(bt))
    print(f"\nOutputs in {OUT}")

    plot(res)
    if show:
        plt.show()
    return bt


if __name__ == "__main__":
    main()
