"""
Pipeline: can PCA recover a structure we planted ourselves?

2 simulated stocks with a known ("true") covariance.
We estimate the covariance from simulated data, run PCA on both the truth and the estimate, and compare.
The gap between them is estimation error from finite data.
Try N_DAYS = 50 to see it grow.
"""
from numpy import array, cov
from numpy.random import default_rng
from pca_lab.pca import sorted_eig, r_squared

# 1. sigma^2 (ARPM: s2) -> the true covariance of the 2 stock's returns
TRUE_COVARIANCE = array([[4.0, 3.0], [3.0, 9.0]])

# t̄ (ARPM: t_bar) — number of daily observations
N_DAYS = 5000

# 2. load (here: simulate) data
random_generator = default_rng(0)
# x (ARPM : x) - shape (2 stocks, N_DAYS days)
stock_returns = random_generator.multivariate_normal(mean=[0,0], cov=TRUE_COVARIANCE, size=N_DAYS).T

# 3. estimate
# σ̂² (ARPM: s2_hat) — covariance estimated from the simulated data
sample_covariance = cov(stock_returns)

# 4. decompose: the estimate AND the truth
# λ̂, ê (ARPM: lam_hat, e_hat)
estimated_variances, estimated_directions = sorted_eig(sample_covariance)
# λ, e (ARPM: lam, e)
true_variances, true_directions = sorted_eig(TRUE_COVARIANCE)

# 5. diagnose
print("True Eigenvalues : ", true_variances.round(3))
print()
print("Estimated Eigenvalues : ", estimated_variances.round(3))
print()
print("True 1st Eigenvectors : ", true_directions[:, 0].round(3))
print()
print("Estimated 1st Eigenvectors : ", estimated_directions[:, 0].round(3))
print()
print("R-squared (1 comp, 2 comps) : ", r_squared(estimated_variances).round(3))
print()