# pipelines/

End-to-end implementations of Unit 3, built after the homeworks.
Each pipeline: load/clean data → estimate → predict/decide → write results to `outputs/`, with tests in `tests/`.

Planned (to be agreed):
- correlation clustering of S&P 500 sectors with regime/stability checks
- graphical model (precision-matrix network) of returns

## Planned

1. **Sector vs correlation clustering: out-of-sample test** *(from s_shrink_corr_clusters)*
   - Fit k-means clusters on 2006–2012 (learning set).
   - On later data (testing set), compare the within-minus-between correlation gap for clusters vs official sectors.
   - In-sample result to beat: sectors 0.098, clusters 0.110.
   - Extension: k_bar + 1 clusters, to see whether a REIT cluster splits out of Financials.
2. Graphical model (precision-matrix network) of returns *(after the Markov network homework)*