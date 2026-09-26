# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats

from arpym.twist_prob_mom_match import twist_prob_mom_match


def simulate_quadn(alpha, beta, gamma, mu, sigma2, j_bar):
    """This function simulates a quadratic form with normal distribution, 
       generating scenarios and twisting probabilities to match desired moments

    Parameters
    ----------
        alpha : float
        beta : array, shape(n_bar,)
        gamma : array, shape(n_bar, n_bar)
        mu : array, shape(n_bar,)
        sigma2 : array, shape(n_bar, n_bar)
        j_bar : float

    Returns
    -------
        y : array, shape(j_bar,)
        p_bar : array, shape(j_bar,)
    """
    
    if np.ndim(beta) == 1:
        n_bar = len(beta)
    else:
        n_bar = 1
        
    beta, gamma = np.reshape(beta, (n_bar, 1)), np.reshape(gamma, (n_bar, n_bar))
    mu, sigma2 = np.reshape(mu, (n_bar, 1)), np.reshape(sigma2, (n_bar, n_bar))

    # step 1: perform cholesky decomposition of sigma2
    l = np.linalg.cholesky(sigma2)

    # step 2: compute parameter lambda
    lambd, e = np.linalg.eig(l.T@gamma@l)
    lambd = lambd.reshape(-1, 1)

    # step 3: compute new parameters beta_tilde and gamma_tilde
    beta_tilde = beta + 2*gamma@mu
    gamma_tilde = e.T@l.T@beta_tilde

    # step 4: generate Monte Carlo scenarios
    z = stats.multivariate_normal.rvs(np.zeros(n_bar), np.eye(n_bar), j_bar)
    y = alpha + beta.T@mu + mu.T@gamma@mu + z@gamma_tilde + z**2@lambd

    # step 5: match expectation and variance by twisting probabilities
    e_y = alpha + beta.T@mu + mu.T@gamma@mu + np.trace(sigma2@gamma)
    v_y = 2*np.trace(np.linalg.matrix_power(gamma@sigma2, 2)) + beta_tilde.T@sigma2@beta_tilde
    p_bar = twist_prob_mom_match(y.reshape(-1, 1), e_y[0], v_y).reshape(-1)

    return np.squeeze(y), np.squeeze(p_bar)
