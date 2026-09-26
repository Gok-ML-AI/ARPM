# -*- coding: utf-8 -*-
import numpy as np
import warnings


def var2mvou(b_dt, mu_dt, sig2_dt, dt):
    """This function computes the parameters of the Multivariate Ornstein-Uhlenbeck (MVOU) 
       process given the parameters of a discrete-time dynamical system

    Parameters
    ----------
        b_dt : array, shape(d_bar, d_bar)
        mu_dt : array, shape(d_bar,)
        sig2_dt : array, shape(d_bar, d_bar)
        dt : scalar

    Returns
    -------
        theta : array, shape(d_bar, d_bar)
        mu_mvou : array, shape(d_bar)
        sigma2_mvou : array, shape(d_bar, d_bar)
    """

    b_dt = np.atleast_2d(b_dt).copy()
    mu_dt = np.atleast_1d(mu_dt).copy()
    sig2_dt = np.atleast_2d(sig2_dt).copy()
    d_bar = mu_dt.shape[0]
    
    # eigenvalues and eigenvectors of b_dt
    lambda2, e = np.linalg.eig(b_dt)  
    # warning: eigvals with negative real part, explosive MVOU
    if any(np.real(np.log(lambda2)) > 0):
        warnings.warn(('Warning: eigenvalues with ' + 'negative part: the MVOU is explosive'))
        
    # theta parameter of embedding MVOU 
    theta = -(1/dt)*e@np.diag(np.log(lambda2))@np.linalg.pinv(e)
    theta = np.real(theta)
    # drift of embedding MVOU
    mu_mvou = theta@np.linalg.pinv(np.eye(d_bar) - b_dt)@mu_dt
    # covariance matrix of embedding MVOU
    vec_sig2 = np.reshape(sig2_dt, (d_bar**2, 1), 'F')
    l_kprod, e_kprod = np.linalg.eig(np.kron(b_dt, b_dt))
    ln_bkronb = e_kprod@np.diag(np.log(l_kprod))@np.linalg.pinv(e_kprod)
    sigma2_mvou = -(1/dt)*np.linalg.pinv(np.eye(d_bar**2) - np.kron(b_dt, b_dt))@ln_bkronb@vec_sig2
    sigma2_mvou = np.real(np.reshape(sigma2_mvou, (d_bar, d_bar), 'F'))

    return theta, mu_mvou.reshape(-1), sigma2_mvou
