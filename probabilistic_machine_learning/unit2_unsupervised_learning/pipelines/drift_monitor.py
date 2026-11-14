"""
drift_monitor.py
Pipeline 1: data-drift monitor for a model's input features.

Real-world scenario
-------------------
A model (trading signal, risk model, credit model...) was trained on features from a
REFERENCE period. In production you must check, every few days, whether the features
it now sees still come from the same distribution. If they don't, its predictions
can no longer be trusted, so it gets flagged for review or retraining.

Theory used (Unit 2, Section 17.2)
----------------------------------
* Glivenko-Cantelli: the historical cdf of the reference sample is a good estimate of
  the feature's true training-time distribution.
* Two-sample KS statistic D = sup_z |F_ref(z) - F_live(z)|: exactly the GC distance,
  but between two historical cdfs. Big D means the live data looks different.
* Optional flexible probabilities on the live window (exponential decay), so the
  newest days count more; ens(p) then sets the effective sample size of the test.

Practical point: GC and the KS p-value assume i.i.d. data. Rolling features such as
20-day volatility are strongly autocorrelated, so textbook p-values would raise far
too many false alarms. The alert threshold is therefore CALIBRATED on held-out
reference data (what D normally looks like when nothing has changed) instead of
trusting p < 0.01.

Stages: load -> clean -> features -> calibrate threshold -> monitor -> alerts -> report
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "historical_lab"))
from hfp_tools import exp_decay_probs, hfp_cdf, ks_2samp_hfp  # noqa: E402
from data import OUTPUT_DIR, clean_prices, load_prices          # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "^GSPC"
START = "2005-01-01"
REF_START, REF_END = "2010-01-01", "2016-12-31"   # the model's "training" period
CALIB_FRACTION = 0.30     # last 30% of the reference period calibrates the threshold
LIVE_WINDOW = 60          # days of live data compared with the reference each check
STEP = 5                  # run the check every 5 trading days (weekly)
NULL_QUANTILE = 0.99      # alert if D beats 99% of the "no drift" values
PERSIST = 3               # declare DRIFT only after 3 consecutive alerts (cuts noise)
LIVE_HALF_LIFE = None     # e.g. 20 to weight recent live days more (HFP); None = equal

OUT = OUTPUT_DIR / "drift"
log = logging.getLogger("drift_monitor")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """The model's input features, all built from invariants (returns), not prices."""
    ret = np.log(prices["close"]).diff()
    feats = pd.DataFrame({
        "ret_1d": ret,                                      # daily log return
        "vol_20d": ret.rolling(20).std() * np.sqrt(252),    # annualised 20d volatility
        "mom_20d": ret.rolling(20).sum(),                   # 20d momentum
    })
    if "volume" in prices and prices["volume"].notna().mean() > 0.95:
        feats["volume_chg"] = np.log(prices["volume"]).diff()  # daily volume change
    return feats.dropna()


def live_weights(n: int) -> np.ndarray | None:
    return None if LIVE_HALF_LIFE is None else exp_decay_probs(n, LIVE_HALF_LIFE)


def calibrate_thresholds(ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the reference period: the first part is the reference distribution, the
    last part supplies 'normal' live windows. The NULL_QUANTILE of their KS distances
    is the alert threshold for each feature."""
    n_fit = int(len(ref) * (1 - CALIB_FRACTION))
    fit, calib = ref.iloc[:n_fit], ref.iloc[n_fit:]
    p = live_weights(LIVE_WINDOW)
    thresholds = {}
    for col in ref.columns:
        null_d = [ks_2samp_hfp(calib[col].iloc[i - LIVE_WINDOW:i], fit[col], px=p)[0]
                  for i in range(LIVE_WINDOW, len(calib) + 1)]
        thresholds[col] = float(np.quantile(null_d, NULL_QUANTILE))
    return fit, pd.Series(thresholds, name="threshold")


def monitor(feats: pd.DataFrame, fit: pd.DataFrame, thresholds: pd.Series) -> pd.DataFrame:
    """Every STEP days after the reference period, compare the last LIVE_WINDOW days
    of each feature with the reference distribution."""
    live = feats.loc[feats.index > pd.Timestamp(REF_END)]
    p = live_weights(LIVE_WINDOW)
    rows = []
    for end in range(LIVE_WINDOW, len(live) + 1, STEP):
        window = live.iloc[end - LIVE_WINDOW:end]
        for col in feats.columns:
            d, pval = ks_2samp_hfp(window[col], fit[col], px=p)
            rows.append({"date": window.index[-1], "feature": col, "ks_stat": d,
                         "p_value": pval, "threshold": thresholds[col]})
    res = pd.DataFrame(rows)
    res["alert"] = res["ks_stat"] > res["threshold"]
    # DRIFT = PERSIST consecutive alerts for the same feature
    res["drift"] = (res.groupby("feature")["alert"]
                       .transform(lambda a: a.astype(int).rolling(PERSIST).sum() == PERSIST))
    return res


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for col, g in res.groupby("feature", sort=False):
        drift_dates = g.loc[g["drift"], "date"]
        last = g.iloc[-1]
        out.append({
            "feature": col,
            "threshold": last["threshold"],
            "latest_ks": last["ks_stat"],
            "pct_checks_alert": 100 * g["alert"].mean(),
            "first_drift": drift_dates.iloc[0].date() if len(drift_dates) else None,
            "naive_pct_p<0.01": 100 * (g["p_value"] < 0.01).mean(),
            "status_now": "DRIFT" if last["drift"] else ("alert" if last["alert"] else "ok"),
        })
    return pd.DataFrame(out).set_index("feature")


def plot(res: pd.DataFrame, feats: pd.DataFrame, fit: pd.DataFrame) -> None:
    cols = list(feats.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.4 * len(cols)), sharex=True)
    for ax, col in zip(np.atleast_1d(axes), cols):
        g = res[res["feature"] == col]
        ax.plot(g["date"], g["ks_stat"], lw=1, label="KS distance D")
        ax.axhline(g["threshold"].iloc[0], color="red", ls="--", lw=1, label="threshold")
        ax.scatter(g.loc[g["drift"], "date"], g.loc[g["drift"], "ks_stat"],
                   color="red", s=8, zorder=3, label="DRIFT")
        ax.set_ylabel(col)
    np.atleast_1d(axes)[0].legend(loc="upper left", ncol=3, fontsize=8)
    fig.suptitle(f"{TICKER} feature drift vs reference {REF_START[:4]}-{REF_END[:4]} "
                 f"(window {LIVE_WINDOW}d)")
    fig.tight_layout()
    fig.savefig(OUT / "drift_timeline.png", dpi=120)

    # Glivenko-Cantelli picture: reference cdf vs the latest live window's cdf,
    # for the feature that is currently furthest past its threshold
    last = res.groupby("feature").tail(1).set_index("feature")
    worst = (last["ks_stat"] / last["threshold"]).idxmax()
    live_win = feats[worst].iloc[-LIVE_WINDOW:]
    grid = np.linspace(min(fit[worst].min(), live_win.min()),
                       max(fit[worst].max(), live_win.max()), 800)
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.step(grid, hfp_cdf(grid, fit[worst].to_numpy()), where="post", label="reference (training)")
    ax2.step(grid, hfp_cdf(grid, live_win.to_numpy(), live_weights(LIVE_WINDOW)),
             where="post", label=f"latest {LIVE_WINDOW} days")
    ax2.set_title(f"{worst}: D = {last.loc[worst, 'ks_stat']:.3f} "
                  f"(threshold {last.loc[worst, 'threshold']:.3f})")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT / "drift_latest_cdf.png", dpi=120)


def main(show: bool = True) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    prices = clean_prices(load_prices(TICKER, START))
    feats = build_features(prices)
    ref = feats.loc[REF_START:REF_END]
    fit, thresholds = calibrate_thresholds(ref)
    log.info("Reference: %d days (%d fit, %d calibration)", len(ref), len(fit), len(ref) - len(fit))

    res = monitor(feats, fit, thresholds)
    summary = summarise(res)

    res.to_csv(OUT / "drift_checks.csv", index=False)
    summary.to_csv(OUT / "drift_summary.csv")
    pd.set_option("display.width", 140)
    print("\n=== Drift summary ===")
    print(summary.round(3).to_string())
    print(f"\nChecks run: {res['date'].nunique()}  |  outputs in {OUT}")

    plot(res, feats, fit)
    if show:
        plt.show()
    return summary


if __name__ == "__main__":
    main()
