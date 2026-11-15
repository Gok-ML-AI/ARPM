"""Checks the maths identities behind k-means-as-autoencoder."""
import numpy as np
import pytest

from clustering_lab.kmeans_tools import (
    encode, simulate_binary_gaussian_mixture, two_means_boundary, variance_split,
)


@pytest.fixture
def mixture():
    rng = np.random.default_rng(1)
    return simulate_binary_gaussian_mixture(0.4, 0.0, 2.0, 0.5, 0.2, 50_000, rng)


def test_simulation_uses_standard_deviation(mixture):
    x, h = mixture
    # variance of each component should match sigma2, not sigma2**2
    assert x[h == 0].var() == pytest.approx(0.5, rel=0.03)
    assert x[h == 1].var() == pytest.approx(0.2, rel=0.03)
    assert h.mean() == pytest.approx(0.4, abs=0.01)


def test_boundary_is_midpoint_of_sorted_centroids(mixture):
    x, _ = mixture
    x_c, c, labels = two_means_boundary(x)
    assert c[0] < c[1]
    assert x_c == pytest.approx(c.mean())
    assert x_c == pytest.approx(1.0, abs=0.2)   # between the two bells
    assert x[labels == 1].mean() > x[labels == 0].mean()


def test_law_of_total_variance(mixture):
    x, _ = mixture
    s = variance_split(x, encode(x, 0.7))
    assert s.within + s.between == pytest.approx(s.total)
    assert 0 < s.r2_between < 1


def test_kmeans_cutoff_beats_arbitrary_cutoff(mixture):
    x, _ = mixture
    x_c, _, _ = two_means_boundary(x)
    best = variance_split(x, encode(x, x_c)).within
    for arbitrary in (0.0, 0.5, 1.6):
        assert best <= variance_split(x, encode(x, arbitrary)).within + 1e-9
