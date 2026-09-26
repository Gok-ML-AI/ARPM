# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


def fit_dcc_t(dx, p=None, *, rho2=None, param0=None, g=0.99):
    
    """This function estimates a Dynamic Conditional Correlation (DCC) model for
    multivariate time series data by minimizing the negative log-likelihood of a GARCH model.

    Parameters
    ----------
        dx : array, shape(t_bar, i_bar)
        p : array, optional, shape(t_bar)
        rho2 : array, shape(i_bar, i_bar)
        param0 : list or array, shape(2,)
        g : scalar, optional

    Returns
    -------
        params : list, shape(3,)
        r2_t : array, shape(t_bar, i_bar, i_bar)
        eps : array, shape(t_bar, i_bar)
        q2_t_bar : array, shape(i_bar, i_bar)

    """

    # setup default values
    t_bar, i_bar = dx.shape
    
    # flexible probabilities
    if p is None:
        p = np.ones(t_bar)/t_bar
    # target correlation
    if rho2 is None:
        rho2 = np.cov(dx.T, aweights=p)
        rho2 = np.diag(1/np.sqrt(np.diag(rho2)))@rho2@np.diag(1/np.sqrt(np.diag(rho2)))
    # initial parameters
    if param0 is None:
        param0 = [0.01, g - 0.01]

    # compute negative log-likelihood of GARCH
    def llh(params):
        a, b = params
        mu = np.zeros(i_bar)
        q2_t = rho2.copy()
        r2_t = np.diag(1/np.sqrt(np.diag(q2_t)))@q2_t@np.diag(1/np.sqrt(np.diag(q2_t)))
        llh = 0.0
        for t in range(t_bar):
            llh = llh - p[t]*multivariate_normal.logpdf(dx[t, :], mu, r2_t)
            q2_t = rho2*(1 - a - b) + a*np.outer(dx[t, :], dx[t, :]) + b*q2_t
            r2_t = np.diag(1/np.sqrt(np.diag(q2_t)))@q2_t@np.diag(1/np.sqrt(np.diag(q2_t)))
        return llh

    # minimize the negative log-likelihood
    # parameter boundaries
    bnds = ((1e-20, 1.), (1e-20, 1.))
    # stationary constraints
    cons = {'type': 'ineq', 'fun': lambda param: g - param[0] - param[1]}
    a, b = minimize(llh, param0, bounds=bnds, constraints=cons)['x']

    # compute realized correlations and residuals
    q2_t = rho2.copy()
    r2_t = np.zeros((t_bar, i_bar, i_bar))
    r2_t[0, :, :] = np.diag(1/np.sqrt(np.diag(q2_t)))@q2_t@np.diag(1/np.sqrt(np.diag(q2_t)))
    for t in range(t_bar - 1):
        q2_t = rho2*(1 - a - b) + a*np.outer(dx[t, :], dx[t, :]) + b*q2_t
        r2_t[t + 1, :, :] = np.diag(1/np.sqrt(np.diag(q2_t)))@q2_t@np.diag(1/np.sqrt(np.diag(q2_t)))
    l_t = np.linalg.cholesky(r2_t)
    eps = np.linalg.solve(l_t, dx)

    return [1. - a - b, a, b], r2_t, eps, q2_t
