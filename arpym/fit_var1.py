import numpy as np
import warnings
from arpym.cointegration_fp import cointegration_fp
from arpym.fit_lfm_mlfp import fit_lfm_mlfp


def fit_var1(x, p=None, *, nu=10**9, tol=1e-2, maxiter=500, shrink=False, lam=0.):
    
    """This function fits a VAR(1) model to the input time series data, estimating coefficients for autoregressive relationships, handling both stationary and cointegrated processes, 
    and providing parameter estimates such as coefficients, mean errors, and covariance matrices.

    Parameters
    ----------
        x : array, shape (t_bar, n_bar)
        p : array, shape(t_bar, )
        nu : scalar, optional
        tol : scalar, optional
        maxiter: scalar, optional
        shrink : bool, optional
        lam : float, optional

    Returns
    -------
        b_hat : array, shape(n_bar, n_bar)
        mu_epsi_hat : array, shape(n_bar, )
        sigma2_hat : array, shape(n_bar, n_bar)

    """

    t_bar = x.shape[0]

    if len(x.shape) == 1:
        x = x.reshape((t_bar, 1))
        n_bar = 1
    else:
        n_bar = x.shape[1]

    if p is None:
        p = np.ones(t_bar)/t_bar

    # step 0: find cointegrated relationships
    c_hat, _ = cointegration_fp(x, p, b_threshold=1)
    l_bar = c_hat.shape[1]

    # step 1: fit VAR(1)
    if l_bar == n_bar:
        # step 1a: fit stationary process
        x_t = x[1:, :]
        x_tm1 = x[:-1, :]
        mu_epsi_hat, b_hat, sig2_hat, _ = fit_lfm_mlfp(x_t, x_tm1, p[1:]/np.sum(p[1:]), nu, tol=tol,
                                          maxiter=maxiter, shrink=shrink, lam=lam)
    if l_bar < n_bar and l_bar > 0:
        # step 1b: fit cointegrated process
        x_t = np.diff(x, axis=0)
        x_tm1 = (x@c_hat)[:-1, :]
        mu_epsi_hat, d_hat, sig2_hat, _ = fit_lfm_mlfp(x_t, x_tm1, p[1:]/np.sum(p[1:]), nu, tol=tol,
                                          maxiter=maxiter, shrink=shrink, lam=lam)
        if np.ndim(d_hat) < 2:
            d_hat = d_hat.reshape((-1, 1))
        b_hat = np.eye(n_bar) + d_hat@c_hat.T

    if l_bar == 0:
        # step 1c: fit VAR(1) to differences
        warnings.warn('Warning: non-cointegrated series detected. ' +
                      'Fit performed on differences')
        delta_x_t = np.diff(x, axis=0)[1:]
        delta_x_tm1 = np.diff(x, axis=0)[:-1]
        mu_epsi_hat, b_hat, sig2_hat, _ = fit_lfm_mlfp(delta_x_t, delta_x_tm1, p[2:]/np.sum(p[2:]), nu,
                                          tol=tol, maxiter=maxiter, shrink=shrink, lam=lam)
        # b_hat is close to zero: detect random walk
        if np.linalg.norm(b_hat.reshape(-1, 1)) < tol:
            # random walk
            warnings.warn('Warning: a random walk has been fitted')
            b_hat = np.eye(n_bar)

    return np.squeeze(b_hat), np.squeeze(mu_epsi_hat), np.squeeze(sig2_hat)
