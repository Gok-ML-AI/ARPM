# Shared ARPM data

| File | Contents | Coverage |
|---|---|---|
| db_stocks_sp.csv | S&P 500 stock values, 2-row header (sector, ticker) | 2006-07-31 → 2012-12-05 |
| SPX.csv | S&P 500 index | (fill in) |

## Open issue: out-of-sample data (affects every unit's pipelines)
ARPM databases end around 2012. Honest pipelines need a later testing set.

To decide (once, reused by all units):
- [ ] Source for newer prices (newer ARPM database? yfinance? other)
- [ ] Survivorship: tickers delisted or acquired since 2012 (e.g. FDO → bought by DLTR, 2015).
      Drop them, or keep them until delisting? Document the choice.
- [ ] Sector labels: GICS changed in 2016 (Real Estate split out) and 2018 (Communication Services).
      Use the labels as they were at the time, or today's?
- [ ] One loader in a shared module, so every unit gets the same cleaned data

Used by: unit3 pipeline (sector vs k-means clustering OOS test)