# -*- coding: utf-8 -*-
import numpy as np

def factor_analysis_paf(s2, k_bar, epsilon=1e-6, xi=0.005, max_iter=1000):
    """This function implements the Principal Axis Factorization (PAF) algorithm 
    to solve the factor analysis optimization.

    Parameters
    ----------
        s2 : array, shape (n_bar, n_bar)
            Sample covariance matrix.
        k_bar : int
            Number of factors.
        epsilon : float, optional
            Convergence threshold.
        xi : float, optional
            Rescaling parameter to ensure idiosyncratic variances are positive.
        max_iter : int, optional
            Maximum number of iterations.

    Returns
    -------
        sigma2 : array, shape (n_bar, n_bar)
            Low-rank plus diagonal covariance matrix estimate.
        beta : array, shape (n_bar, k_bar)
            Factor loadings.
        delta : array, shape (n_bar,)
            Idiosyncratic variances.
    """

    n_bar = s2.shape[0]

    # initial guess for delta
    delta = 1.0/np.diag(np.linalg.inv(s2))
    delta_prev = np.zeros_like(delta)

    for t in range(max_iter):
        # convergence check
        dist = np.max(np.abs(delta - delta_prev))/np.max(np.abs(delta))
        if t > 0 and dist < epsilon:
            break

        delta_prev = delta.copy()

        # 1. spectral decomposition
        s2_delta = s2 - np.diag(delta)
        lambda_delta, e_delta = np.linalg.eigh(s2_delta)
        idx = np.argsort(lambda_delta)[::-1]
        lambda_delta_k_bar = lambda_delta[idx][:k_bar]
        e_delta_k_bar = e_delta[:, idx][:, :k_bar]

        # 2. fix eigenvalues
        lambda_delta_k_bar = np.maximum(lambda_delta_k_bar, 0.0)

        # 3. update loadings
        beta = e_delta_k_bar@np.diag(np.sqrt(lambda_delta_k_bar))

        # 4. rescale loadings
        gamma = np.zeros(n_bar)
        for n_idx in range(n_bar):
            s_nn = s2[n_idx, n_idx]
            beta_norm = np.linalg.norm(beta[n_idx, :])
            if beta_norm < 1e-10:
                gamma[n_idx] = 1.0
            else:
                limit = ((1.0 - xi)*np.sqrt(s_nn))/beta_norm
                gamma[n_idx] = min(1.0, limit)
        beta = np.diag(gamma)@beta

        # 5. update idiosyncratic variances
        delta = np.diag(s2 - beta@beta.T).copy()
        delta[delta < 1e-6] = 1e-6

    sigma2 = beta@beta.T + np.diag(delta)

    return sigma2, beta, delta
