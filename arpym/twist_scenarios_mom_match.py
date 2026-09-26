# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg


def twist_scenarios_mom_match(x, m_bar, s2_bar, p=None, method='PCA', d=None):
    """This function twists scenarios to match specified moments using a linear transformation method

    Parameters
    ----------
        x : array, shape (j_bar, n_bar) if n_bar > 1 or (j_bar,) for n_bar = 1
        m_bar : array, shape (n_bar,)
        s2_bar : array, shape (n_bar, n_bar)
        p : array, optional, shape (j_bar,)
        method : string, optional
        d : array, shape (k_bar, n_bar), optional

    Returns
    -------
        x : array, shape (j_bar, n_bar) if n_bar > 1 or (j_bar,) for n_bar = 1

    """

    if np.ndim(m_bar) == 0:
        m_bar = np.reshape(m_bar, 1).copy()
    else:
        m_bar = np.array(m_bar).copy()
    if np.ndim(s2_bar) == 0:
        s2_bar = np.reshape(s2_bar, (1, 1))
    else:
        s2_bar = np.array(s2_bar).copy()
    if len(x.shape) == 1:
        x = x.reshape(-1, 1).copy()

    if p is None:
        j_bar = x.shape[0]
        p = np.ones(j_bar)/j_bar  # uniform probabilities as default value

    # step 1: original moments
    m_x = np.average(x, axis=0, weights=p)
    s2_x = np.cov(x.T, aweights=p)

    # step 2: transpose-square-root of s2_x
    lam, e = linalg.eigh(s2_x, subset_by_index=(0, s2_x.shape[0] - 1))
    lam = lam[::-1]
    e = e[:, ::-1]
    # enforce a sign convention on coefficients
    # largest element in each eigenvector will have a positive sign
    ind = np.argmax(abs(e), axis=0)
    ind = np.diag(e[ind, :]) < 0
    e[:, ind] = -e[:, ind]

    r_x = e@np.diag(np.sqrt(lam))

    # step 3: transpose-square-root of s2_bar
    lam, e = linalg.eigh(s2_bar, subset_by_index=(0, s2_bar.shape[0] - 1))
    lam = lam[::-1]
    e = e[:, ::-1]
    # enforce a sign convention on the coefficients
    # largest element in each eigenvector will have a positive sign
    ind = np.argmax(abs(e), axis=0)
    ind = np.diag(e[ind, :]) < 0
    e[:, ind] = -e[:, ind]

    r_bar = e@np.diag(np.sqrt(lam))

    # step 4: twist matrix
    b = r_bar@np.linalg.inv(r_x)

    # step 5: shift vector
    a = m_bar.reshape(-1, 1) - b@m_x.reshape(-1, 1)

    # step 6: twisted scenarios
    x_bar = (a + b@x.T).T

    return np.squeeze(x_bar)
