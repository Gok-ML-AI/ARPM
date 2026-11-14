# Unit 2 – Historical estimation pipelines

Two real-world uses of Glivenko–Cantelli and historical-with-flexible-probabilities (HFP)
estimation, built on `historical_lab/hfp_tools.py`.

| Pipeline | Question it answers | Theory used |
|---|---|---|
| `drift_monitor.py` | Do a model's live input features still look like its training data? | Two-sample KS = Glivenko–Cantelli sup distance between two historical cdfs; ens(p) as effective sample size |
| `var_backtest.py` | Is our 1-day 99% historical VaR well calibrated? Equal weights or exponential decay? | VaR/ES as plug-in HFP quantile/tail mean; Kupiec, Christoffersen and Basel traffic-light backtests |

## Run

```bash
pip install numpy pandas scipy matplotlib yfinance pytest
python pipelines/drift_monitor.py
python pipelines/var_backtest.py
pytest tests
```

The first run downloads S&P 500 (`^GSPC`) from Yahoo Finance and caches it in `data/`.
Results (CSV + PNG) go to `outputs/drift/` and `outputs/var/`. All settings are
constants at the top of each script.

## Stages

**drift_monitor:** load → clean → features (1d return, 20d vol, 20d momentum, volume
change) → calibrate alert thresholds on held-out reference data → weekly KS checks on a
rolling 60-day live window → alert / DRIFT (3 consecutive alerts) → summary, timeline
plot, reference-vs-live cdf plot.

**var_backtest:** load → clean → log returns (invariants) → rolling out-of-sample VaR and
ES (500-day window: equal weights vs half-lives 250 and 60) → breaches → Kupiec,
Christoffersen, Basel → conclusion, VaR plot.

## Key lessons

* **KS p-values assume i.i.d. data.** Rolling features such as 20d volatility are
  autocorrelated, so naive p < 0.01 alarms fire almost constantly. The monitor calibrates
  its threshold empirically instead.
* **Recency vs sample size.** Shorter half-lives react faster to volatility (fewer
  clustered breaches) but lower ens(p). At 99%, the VaR then rests on only a couple
  of effective tail scenarios, which makes it jumpy.
