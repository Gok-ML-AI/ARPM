"""
k-means viewed as an autoencoder (ARPM Unit 3, Labs 1-3, 6, 8).

The big idea in one line
------------------------
    encoder  zeta(x) : squash x into a cluster label z in {0, 1}
    decoder  chi(z)  : rebuild x as the average of its cluster, E{X | Z = z}

The reconstruction error of that autoencoder is exactly the within-cluster
variance, and the law of total variance splits what's left:

    V{X}  =  E{V{X|Z}}   +   V{E{X|Z}}
   total     within          between
             (error)         (what the code explains)

k-means picks the cutoff that makes "within" as small as possible.

Functions here are reusable building blocks for the notebooks, the
homework and the pipelines. They mirror ARPM's notation in the names.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import k_means


def simulate_binary_gaussian_mixture(
    p_h: float,
    mu_x0: float,
    mu_x1: float,
    sigma2_x0: float,
    sigma2_x1: float,
    j_bar: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo scenarios of X = (1-H) * X|{H=0} + H * X|{H=1},
    with H ~ Bernoulli(p_h) and X|{H=h} ~ N(mu_xh, sigma2_xh).

    NOTE: numpy/scipy take the *standard deviation* as the scale argument,
    so we pass sqrt(sigma2). (ARPM's s_autoencoders_kmeans passes the
    variance directly, which silently simulates a narrower mixture.)

    Returns
    -------
    x : (j_bar,) observable scenarios
    h : (j_bar,) hidden labels (0/1) - the "truth" k-means is trying to recover
    """
    rng = np.random.default_rng() if rng is None else rng
    h = rng.binomial(1, p_h, j_bar)
    x_h0 = rng.normal(mu_x0, np.sqrt(sigma2_x0), j_bar)
    x_h1 = rng.normal(mu_x1, np.sqrt(sigma2_x1), j_bar)
    x = (1 - h) * x_h0 + h * x_h1
    return x, h


def two_means_boundary(
    x: np.ndarray, n_init: int = 10, random_state: int | None = 0
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Run 2-means on univariate scenarios and return the optimal cutoff.

    In 1-D with two clusters the boundary x_c* sits exactly halfway between
    the two centroids (any point is assigned to the nearer centroid).

    Returns
    -------
    x_c_star  : boundary point of the optimal clustering
    centroids : (2,) centroids sorted so centroids[0] < centroids[1]
    labels    : (j_bar,) labels relabelled so 1 = upper cluster
    """
    centroid, label, _ = k_means(
        np.asarray(x).reshape(-1, 1), 2, n_init=n_init, random_state=random_state
    )
    c = centroid.ravel()
    if c[1] < c[0]:  # k-means labels are arbitrary; force 1 = upper
        c = c[::-1]
        label = 1 - label
    x_c_star = float(c.mean())
    return x_c_star, c, label


def encode(x: np.ndarray, x_c: float) -> np.ndarray:
    """Encoder zeta(x) = 1{x > x_c}: the code is just which side of the cutoff."""
    return (np.asarray(x) > x_c).astype(int)


@dataclass(frozen=True)
class VarianceSplit:
    """Law-of-total-variance decomposition for a binary code Z."""

    p: float                 # P{Z = 1}
    chi_0: float             # best decoder chi*(0) = E{X | Z=0}
    chi_1: float             # best decoder chi*(1) = E{X | Z=1}
    total: float             # V{X}
    within: float            # E{V{X|Z}}  = reconstruction error
    between: float           # V{E{X|Z}}

    @property
    def r2_between(self) -> float:
        """Share of total variance explained by the code (like R^2 / PCA's explained variance)."""
        return self.between / self.total


def variance_split(x: np.ndarray, z: np.ndarray) -> VarianceSplit:
    """
    Decompose V{X} into within- and between-cluster variance for code z.

    Uses population (ddof=0) moments throughout so the identity
    total = within + between holds exactly on the sample.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z)
    x0, x1 = x[z == 0], x[z == 1]
    p = x1.size / x.size
    chi_0, chi_1 = x0.mean(), x1.mean()
    within = (1 - p) * x0.var() + p * x1.var()
    total = x.var()
    return VarianceSplit(
        p=p, chi_0=chi_0, chi_1=chi_1,
        total=total, within=within, between=total - within,
    )
