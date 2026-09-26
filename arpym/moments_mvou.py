# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg


def moments_mvou(x_tnow, deltat_m, theta, mu, sigma2):
    """This function computes the moments of a multivariate Ornstein-Uhlenbeck process at different time intervals

    Parameters
    ----------
        x_tnow : array, shape(n_bar, )
        deltat_m : array, shape(m_bar, )
        theta : array, shape(n_bar, n_bar)
        mu : array, shape(n_bar, )
        sigma2 : array, shape(n_bar, n_bar)

    Returns
    -------
        mu_dt_m : array, shape(m_bar, n_bar)
        mu_deltat_m : array, shape(m_bar, n_bar)
        sigma2_deltat_m : array, shape(m_bar, n_bar, n_bar)

    """

    if len(x_tnow.shape) != 1:
        x_tnow = x_tnow.reshape(-1, 1).copy()
    if len(mu.shape) != 1:
        mu = mu.reshape(-1, 1).copy()

    n_bar = x_tnow.shape[0]

    if isinstance(deltat_m, float) or isinstance(deltat_m, np.int64):
        m_bar = 1
        deltat_m = np.array([deltat_m])
    else:
        m_bar = len(deltat_m)

    mu_dt_m = np.zeros((m_bar, n_bar))
    mu_deltat_m = np.zeros((m_bar, n_bar))
    sigma2_deltat_m = np.zeros((m_bar, n_bar, n_bar))

    for m, tm in np.ndenumerate(deltat_m):
        # drift of shocks
        mu_dt_m[[m], :] = (np.eye(n_bar) - linalg.expm(-theta*tm))@(np.linalg.solve(theta, mu))
        # drift of process
        mu_deltat_m[[m], :] = linalg.expm(-theta*tm)@x_tnow + mu_dt_m[[m], :]
        # covariance of process
        th_sum_th = linalg.kron(theta, np.eye(n_bar)) + linalg.kron(np.eye(n_bar), theta)
        vecsigma2 = np.reshape(sigma2, (n_bar**2, 1), 'F')
        vecsigma2_m = np.linalg.solve(th_sum_th, (np.eye(n_bar**2) - linalg.expm(-th_sum_th*tm)))@vecsigma2
        sig2_m = np.reshape(vecsigma2_m, (n_bar, n_bar), 'F')
        # grant numerical symmetry
        sigma2_deltat_m[[m], :, :] = (sig2_m + sig2_m.T)/2

    # resize
    if n_bar != 1:
        mu_dt_m = mu_dt_m.squeeze()
        mu_deltat_m = mu_deltat_m.squeeze()
    sigma2_deltat_m = np.atleast_2d(sigma2_deltat_m.squeeze())

    return mu_dt_m, mu_deltat_m, sigma2_deltat_m
