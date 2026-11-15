"""
Clustering stocks by correlation (ARPM Unit 3, Labs 22-25).

The pipeline in plain words
---------------------------
1. prices  -> daily log-returns           eps_t = ln v_t - ln v_{t-1}
2. returns -> correlation matrix          c2 (n_bar x n_bar)
3. correlation -> distance                d_ij = sqrt(2 (1 - c2_ij))
       c2 =  1  -> d = 0   (move together, same place)
       c2 =  0  -> d = 1.41
       c2 = -1  -> d = 2   (opposite moves, far apart)
4. group stocks: by sector label, by k-means on correlation, or by the
   minimum spanning tree of d (the "skeleton" of the market)
5. optionally *homogenise*: replace every correlation inside / between
   clusters by its block average - a shrinkage estimator.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ----------------------------------------------------------------- data
def load_db_stocks_sp(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ARPM's db_stocks_sp.csv (2-row header: sector, ticker).

    Returns
    -------
    db      : raw DataFrame (dates x stocks, MultiIndex columns)
    v       : (t_bar, n_bar) stock values
    sectors : (k_bar,) sector names (alphabetical, as pandas stores levels)
    labels  : (n_bar,) sector index of each column, 0..k_bar-1
    """
    db = pd.read_csv(path, index_col=0, header=[0, 1])
    v = db.to_numpy(dtype=float)
    sectors = np.array(db.columns.levels[0])
    labels = np.array(db.columns.codes)[0, :]
    return db, v, sectors, labels


def log_returns(v: np.ndarray) -> np.ndarray:
    """Daily log-returns, shape (t_bar - 1, n_bar)."""
    return np.diff(np.log(v), axis=0)


# ------------------------------------------------- correlation & distance
def correlation(eps: np.ndarray) -> np.ndarray:
    """Sample correlation of the columns of eps (stocks as columns)."""
    return np.corrcoef(eps.T)


def corr_distance(c2: np.ndarray) -> np.ndarray:
    """
    Correlation distance d_ij = sqrt(2 (1 - c2_ij)).

    This is the Euclidean distance between two standardised return series
    (divided by sqrt(t_bar)), so it is a genuine metric: it can be used by
    any distance-based method (MST, hierarchical clustering...).
    """
    d = np.sqrt(np.clip(2.0 * (1.0 - c2), 0.0, None))
    # corrcoef's diagonal is 1 - 1e-16, which would give d_ii ~ 1e-8 and
    # tiny self-loops in the graph; a stock is at distance 0 from itself.
    np.fill_diagonal(d, 0.0)
    return d


def sort_by_labels(c2: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder rows/cols of c2 so stocks with the same label sit together."""
    idx = np.argsort(labels, kind="stable")
    return c2[np.ix_(idx, idx)], idx


def block_edges(sorted_labels: np.ndarray) -> np.ndarray:
    """Cumulative block sizes, used to draw cluster boundaries on heatmaps."""
    return np.cumsum(np.bincount(sorted_labels))


# ------------------------------------------------------------- clustering
def kmeans_correlation_clusters(
    c2: np.ndarray, labels: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    ARPM's 'statistical clustering' trick (s_shrink_corr_clusters):

    simulate 2*n_bar scenarios from N(0, c2) so each *stock* becomes a point
    in R^{2 n_bar}; run k-means on those points, initialised at the sector
    averages so the result is comparable to the sector partition.

    Returns cluster labels (n_bar,).
    """
    rng = np.random.default_rng() if rng is None else rng
    n_bar = c2.shape[0]
    k_bar = int(labels.max()) + 1
    x = rng.multivariate_normal(np.zeros(n_bar), c2, 2 * n_bar)
    c0 = np.column_stack([x[:, labels == k].mean(axis=1) for k in range(k_bar)])
    km = KMeans(n_clusters=k_bar, init=c0.T, n_init=1).fit(x.T)
    return km.labels_


def minimum_spanning_tree(d: np.ndarray) -> nx.Graph:
    """MST of the complete graph whose edge weights are the distances d."""
    return nx.minimum_spanning_tree(nx.from_numpy_array(d))


# ------------------------------------------------------- homogenisation
def homogenize_correlation(
    c2: np.ndarray,
    l_clus: list[int],
    threshold: float = 1e-5,
    step: float = 0.005,
) -> tuple[np.ndarray, float]:
    """
    Homogeneous cluster shrinkage (s_shrink_corr_hom_clusters).

    c2 must already be ordered by cluster. l_clus are the block boundaries,
    e.g. [0, 2, 5] -> cluster 0 = entries 0..1, cluster 1 = entries 2..4.

    Every off-diagonal correlation inside a block is replaced by the block's
    average correlation, every entry between two blocks by the cross-block
    average. If that matrix is not positive definite, the off-diagonals are
    shrunk towards 0 by gamma until the smallest eigenvalue > threshold.

    (Cleaner than ARPM's loop, which compounds gamma each iteration;
    both produce a valid correlation matrix.)

    Returns (c2_hom, gamma).
    """
    n = c2.shape[0]
    k_bar = len(l_clus) - 1
    c2_clus = np.zeros((n, n))
    for k1 in range(k_bar):
        d1 = slice(l_clus[k1], l_clus[k1 + 1])
        b1 = l_clus[k1 + 1] - l_clus[k1]
        if b1 > 1:  # average of off-diagonal entries within the block
            block = c2[d1, d1]
            c2_clus[d1, d1] = (block.sum() - np.trace(block)) / (b1 * (b1 - 1))
        for k2 in range(k1 + 1, k_bar):
            d2 = slice(l_clus[k2], l_clus[k2 + 1])
            avg = c2[d1, d2].mean()
            c2_clus[d1, d2] = avg
            c2_clus[d2, d1] = avg
    np.fill_diagonal(c2_clus, 1.0)

    gamma = 1.0
    c2_hom = c2_clus.copy()
    while np.linalg.eigvalsh(c2_hom).min() <= threshold:
        gamma -= step
        if gamma <= 0:
            raise RuntimeError("homogenisation did not reach positive definiteness")
        c2_hom = gamma * c2_clus
        np.fill_diagonal(c2_hom, 1.0)
    return c2_hom, gamma
