# Unit 3 — Autoencoders, Graphical Models & Clustering for the Stock Market

ARPM Probabilistic Machine Learning, Unit 3.

## Layout

```
unit3_autoencoders_graphical_clustering/
├── clustering_lab/            reusable tools (imported by notebooks, pipelines, tests)
│   ├── paths.py               the ONE place data/output paths are defined
│   ├── kmeans_tools.py        k-means as an autoencoder, law of total variance
│   └── corr_cluster_tools.py  returns → correlation → distance → clusters / MST / homogenisation
├── data/                      db_stocks_sp.csv goes here (not committed; see data/README.md)
├── notebooks/
│   ├── autoencoders_kmeans.ipynb                 ← s_autoencoders_kmeans
│   ├── sector_vs_correlation_clustering.ipynb    ← s_shrink_corr_clusters
│   ├── homogeneous_correlation_clustering.ipynb  ← s_shrink_corr_hom_clusters
│   └── homeworks/             (next: k-means 50pt, Markov random network 20pt, sector clustering 20pt)
├── outputs/                   figures / csvs written by pipelines
├── pipelines/                 end-to-end implementations (after homeworks)
└── tests/                     pytest; runs on synthetic data, no database needed
```

## PyCharm setup

1. Copy this folder into `probabilistic_machine_learning/` next to `unit2_unsupervised_learning/`.
2. Right-click the unit folder → **Mark Directory as → Sources Root** (so `import clustering_lab` resolves without red squiggles).
3. Put `db_stocks_sp.csv` in `data/` (or set env var `ARPM_DB_STOCKS_SP` to its full path).
4. Install deps into the repo's interpreter: `pip install -r requirements.txt`.
5. Tests: right-click `tests/` → **Run 'pytest in tests'** (or `pytest` from the unit folder). Expect 10 passed.
6. Notebooks: open any `.ipynb`, pick the project interpreter as kernel, run all. The first code cell finds the unit
   root itself, so the working directory doesn't matter.

## Changes vs ARPM's originals

| Notebook | Change | Why |
|---|---|---|
| all | data path from `clustering_lab.paths`, `seed(42)` | runs anywhere, reproducible |
| autoencoders_kmeans | `normal.rvs(mu, sqrt(sigma2), …)` | scipy's `scale` is the **std dev**; ARPM passed the variance |
| homogeneous_correlation_clustering | cross-check cell only (ARPM code untouched) | ARPM's inner loop overwrites the within-block average with one that includes the diagonal 1s |
| — | `corr_distance` forces `d_ii = 0` | `corrcoef` diagonal is 1−1e-16 → `d_ii ≈ 1e-8` → tiny self-loops in the MST graph |

   - Data for the testing set: see ARPM/data/README.md → "Open issue: out-of-sample data"