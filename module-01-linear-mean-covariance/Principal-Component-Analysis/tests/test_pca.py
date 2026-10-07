"""
Theory checks for the PCA toolbox: each test is one result from ARPM
turned into a t=yes/no check that the computer runs every time.
"""
from numpy import array, cov, allclose, eye, diag
from numpy.random import default_rng
from pca_lab.pca import sorted_eig, r_squared

N_STOCKS = 2
N_DAYS = 5000
# σ² (ARPM: s2) - the 'true' covariance we choose, so we know the right answer.
# Stock 1 variance 4, stock 2 variance 9, covariance 3 (they tend to move together).
TRUE_COVARIANCE = array([[4.0, 3.0], [3.0, 9.0]])

"""
Simulate daily returns for 2 stocks drawn from TRUE_COVARIANCE.

Returns - stock_returns : (N_STOCKS, N_DAYS) = (2, 5000) array (ARPM: x)
ARPM convention: variables down the rows, time across the columns.
"""
def simulate_stock_returns():
    # '0' fixed the seed, so the same random data is output every run
    random_generator = default_rng(0)
    # multivariate_normal() returns (days, stocks)
    # .T transposes to (stocks, days)
    stock_returns = random_generator.multivariate_normal(mean=[0, 0], cov=TRUE_COVARIANCE, size=5000).T
    assert stock_returns.shape == (N_STOCKS, N_DAYS), f"expected {N_STOCKS, N_DAYS}, got {stock_returns.shape}"
    return stock_returns

"""
Eigenvectors are perpendicular (ortho) and of length 1 (normal): Eᵀ E = I.
Diagonal of Eᵀ E = each directions length² -> 1
Off-diagonal = dot product between different directions -> 0.
"""
def test_principal_directions_are_orthonormal():
    sample_covariance = cov(simulate_stock_returns())
    _, principal_directions = sorted_eig(sample_covariance)
    assert allclose(principal_directions.T @ principal_directions, eye(N_STOCKS))

"""
Principal factors z = Eᵀ x are uncorrelated, and Var(z_k) = λ_k.
So Cov(z) is a diagonal matrix with the eigenvalues on the diagonal.
"""
def test_principal_factors_uncorrelated_with_variance_lambda():
    stock_returns = simulate_stock_returns()
    # σ̂² (ARPM: s2_hat)
    sample_covariance = cov(stock_returns)
    variances_along_directions, principal_directions = sorted_eig(sample_covariance)

    # z (ARPM: z) - each day's returns re-expressed in the principal directions:
    # row 1 = 'market' factor (stocks move together), row 2 = 'spread' actor.
    z = principal_directions.T @ stock_returns

    assert allclose(cov(z), diag(variances_along_directions))

"""
Using every component explains 100% of the variance: r²(n) = 1.
"""
def test_r_squared_ends_at_one():
    sample_covariance = cov(simulate_stock_returns())
    variances_along_directions, _ = sorted_eig(sample_covariance)
    assert allclose(r_squared(variances_along_directions)[-1], 1.0)