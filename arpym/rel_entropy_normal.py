# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import lu


def rel_entropy_normal(mu_x_updated, sigma2_x_updated, mu_x_base, sigma2_x_base):
    """This function computes the relative entropy between two multivariate normal distributions 
       specified by their means and covariance matrices

    Parameters
    ----------
        mu_x_updated : array, shape (n_bar,)
        sigma2_x_updated : array, shape (n_bar, n_bar)
        mu_x_base : array, shape (n_bar,)
        sigma2_x_base : array, shape (n_bar, n_bar)

    Returns
    -------
        relative_entropy : scalar

    """

    # fast logarithm-determinant computation
    if (np.ndim(mu_x_updated) == 0) or (np.ndim(mu_x_base) == 0):
        mu_x_updated, sigma2_x_updated, mu_x_base, sigma2_x_base = mu_x_updated.reshape(-1).copy(),\
                                                                   sigma2_x_updated.reshape(1, 1).copy(),\
                                                                   mu_x_base.reshape(-1).copy(),\
                                                                   sigma2_x_base.reshape(1, 1).copy()

    def logdet(a):
        p, l, u = lu(a)
        v = np.log(abs(np.prod(np.r_[np.diag(l), np.diag(u)])))
        return v

    n_bar = sigma2_x_base.shape[0]
    inv_sigma2_x_base = np.linalg.solve(sigma2_x_base, np.eye(n_bar))
    sigma2_x_updated_inv_sigma2_x_base = sigma2_x_updated@inv_sigma2_x_base
    mu_x_updated = mu_x_updated.reshape(-1, 1)
    mu_x_base = mu_x_base.reshape(-1, 1)
    mu_diff = mu_x_updated - mu_x_base
    relative_entropy = 0.5*(np.trace(sigma2_x_updated_inv_sigma2_x_base) - logdet(sigma2_x_updated_inv_sigma2_x_base) +\
                            mu_diff.T@inv_sigma2_x_base@mu_diff - n_bar)

    return np.squeeze(relative_entropy)
