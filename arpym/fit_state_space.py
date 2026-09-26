#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import statsmodels.multivariate.factor as fa
from statsmodels.api import add_constant, WLS
from arpym.kalman_filter import kalman_filter


def fit_state_space(x, k_bar, p=None):
    
    """This function estimates state space models using observed time series data, 
    involving steps such as computing target mean and covariance, performing factor analysis, 
    estimating regression factors, and employing the Kalman filter to extract hidden factors.

    Parameters
    ----------
        x : array, shape (t_bar, n_bar) if n_bar>1 or (t_bar, ) for n_bar=1
        p : array, shape (t_bar,)
        k_bar : scalar

    Returns
    -------
        h : array, shape (t_bar, k_bar) if k_bar>1 or (t_bar, ) for k_bar=1
        alpha_hat : array, shape (n_bar,)
        beta_hat : array, shape (n_bar, k_bar) if k_bar>1 or (n_bar, ) for k_bar=1
        delta_hat : array, shape(n_bar, n_bar)
        alpha_hat_h : array, shape(k_bar,)
        beta_hat_h : array, shape(k_bar, k_bar)
        sigma2_hat_h : array, shape(k_bar, k_bar)

    """
    
    t_bar = x.shape[0]
    if len(x.shape) == 1:
        x = x.reshape((t_bar, 1))
    
    # equal probabilities as default value
    if p is None:
        p = np.ones(t_bar)/t_bar

    # step 1: compute target HFP mean and covariance
    m_x_hat_hfp = np.average(x, axis=0, weights=p)
    s2_x_hat_hfp = np.cov(x.T, aweights=p)
    
    # step 2: compute shift
    alpha_hat = m_x_hat_hfp
    
    # step 3: perform factor analysis
    s_vol = np.sqrt(np.diag(s2_x_hat_hfp))  # target volatility vector
    corr_x = np.diag(1/s_vol)@s2_x_hat_hfp@np.diag(1/s_vol)  # target correlation
    mlf = fa.Factor(n_factor=k_bar, corr=corr_x, method='ml').fit()  # MLF fitted model
    beta_hat = mlf.loadings  # MLF loadings
    delta_hat = mlf.uniqueness  # MLF variances
    beta_hat = np.diag(np.sqrt(np.diag(s2_x_hat_hfp)))@beta_hat  # re-scaled loadings
    delta_hat = np.diag(s2_x_hat_hfp)*delta_hat  # re-scaled variances
    
    if len(beta_hat.shape) == 0:
        beta_hat = beta_hat.reshape(-1, 1)

    # step 4: compute regression factors
    h_fa = beta_hat.T@np.linalg.inv(s2_x_hat_hfp)@(x - m_x_hat_hfp).T
        
    # step 5: estimate regression LFM for h_fa
    model = WLS(h_fa.T[1:, ], add_constant(h_fa.T[:-1, ]), weights=p[:-1]).fit()  # regression model
    # parameters of transition equation
    alpha_hat_h = np.atleast_1d(model.params[0])  # shifts
    beta_hat_h = np.atleast_2d(model.params[1:])  # loadings
    sigma2_hat_h = np.atleast_2d(np.cov(model.resid.T, aweights=p[:-1]))  # dispersion
    # step 6: unearth factors
    h = kalman_filter(x, alpha_hat, beta_hat, np.diagflat(delta_hat),
                      alpha_hat_h, beta_hat_h, sigma2_hat_h)  # hidden factors
                      
    return h, np.squeeze(alpha_hat), np.squeeze(beta_hat), np.squeeze(delta_hat),\
        np.squeeze(alpha_hat_h), np.squeeze(beta_hat_h), np.squeeze(sigma2_hat_h)
