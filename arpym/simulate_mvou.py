# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scipy import stats

from arpym.moments_mvou import moments_mvou


def simulate_mvou(x_tnow, delta_t_m, theta, mu, sigma2, j_bar=1000, init_value=True):

    """This function simulates multivariate Ornstein-Uhlenbeck (MV-OU) processes, generating future scenarios
       with an option to include the initial value as a starting node

    Parameters
    ----------
        x_tnow : array, shape(n_bar, )
        delta_t_m : array, shape(m_bar, )
        theta: array, shape(n_bar, n_bar)
        mu : array, shape(n_bar, )
        sigma2 : array, shape(n_bar,n_bar)
        j_bar : scalar
        init_value : boolean

    Returns
    -------
        x_tnow_thor : array, shape(j_bar, m_bar, n_bar)

    """

    if len(x_tnow.shape) != 1:
        x_tnow = x_tnow.reshape(-1, 1)
    if len(mu.shape) != 1:
        mu = mu.reshape(-1, 1)

    n_bar = x_tnow.shape[0]
    m_bar = delta_t_m.shape[0]

    # moments for each time interval
    x_tnow_thor = np.zeros((j_bar, m_bar, n_bar))
    for m in range(m_bar):
        # deltat_m-steps first and second moments
        mu_delta_t, _, sigma2_delta_t = moments_mvou(np.zeros(n_bar), delta_t_m[m], theta, mu, sigma2)
        # projected invariants
        eps = stats.multivariate_normal.rvs(mu_delta_t, sigma2_delta_t, j_bar)
        # risk drivers
        if m > 0:
            x_prec = x_tnow_thor[:, m-1, :]
        else:
            x_prec = np.tile(x_tnow, (j_bar, 1))
        # Monte Carlo scenarios of projected paths
        x_tnow_thor[:, m, :] = x_prec@linalg.expm(-theta*delta_t_m[m]).T + eps

    # include the initial value as starting node, if selected
    if init_value:
        x_tnow = np.tile(x_tnow, (j_bar, 1))
        x_tnow = np.expand_dims(x_tnow, axis=1)
        x_tnow_thor = np.concatenate((x_tnow, x_tnow_thor), axis=1)

    return x_tnow_thor.squeeze()
