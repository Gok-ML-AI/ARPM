"""Checks the correlation-clustering building blocks on synthetic data."""
import numpy as np
import pytest

from clustering_lab.corr_cluster_tools import (
    block_edges, corr_distance, correlation, homogenize_correlation,
    kmeans_correlation_clusters, load_db_stocks_sp, log_returns,
    minimum_spanning_tree, sort_by_labels,
)


@pytest.fixture
def loaded(fake_db_csv):
    db, v, sectors, labels = load_db_stocks_sp(fake_db_csv)
    c2 = correlation(log_returns(v))
    return v, sectors, labels, c2


def test_loader_shapes(loaded):
    v, sectors, labels, c2 = loaded
    assert v.shape == (300, 20)
    assert len(sectors) == 4 and labels.max() == 3
    assert c2.shape == (20, 20)


def test_distance_is_a_proper_metric_matrix(loaded):
    *_, c2 = loaded
    d = corr_distance(c2)
    assert np.allclose(np.diag(d), 0)
    assert np.allclose(d, d.T)
    assert d.min() >= 0 and d.max() <= 2


def test_sorting_puts_sectors_in_blocks(loaded):
    _, _, labels, c2 = loaded
    c2_sorted, idx = sort_by_labels(c2, labels)
    assert np.all(np.diff(labels[idx]) >= 0)
    assert list(block_edges(labels[idx])) == [5, 10, 15, 20]
    # same-sector correlations should beat cross-sector ones on factor data
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(len(labels), dtype=bool)
    assert c2[same & off].mean() > c2[~same].mean()


def test_mst_connects_everything(loaded):
    *_, c2 = loaded
    mst = minimum_spanning_tree(corr_distance(c2))
    assert mst.number_of_nodes() == 20
    assert mst.number_of_edges() == 19


def test_kmeans_recovers_sectors_on_clean_data(loaded):
    _, _, labels, c2 = loaded
    found = kmeans_correlation_clusters(c2, labels, np.random.default_rng(0))
    # each true sector should land (mostly) in a single found cluster
    for k in range(4):
        counts = np.bincount(found[labels == k], minlength=4)
        assert counts.max() >= 4


def test_homogenized_matrix_is_valid_correlation(loaded):
    _, _, labels, c2 = loaded
    c2_sorted, _ = sort_by_labels(c2, labels)
    c2_hom, gamma = homogenize_correlation(c2_sorted, [0, 5, 10, 15, 20])
    assert np.allclose(np.diag(c2_hom), 1)
    assert np.allclose(c2_hom, c2_hom.T)
    assert np.linalg.eigvalsh(c2_hom).min() > 0
    assert 0 < gamma <= 1
    # constant off-diagonal value inside the first block
    block = c2_hom[:5, :5][~np.eye(5, dtype=bool)]
    assert np.allclose(block, block[0])
