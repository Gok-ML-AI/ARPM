# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy import special
from sklearn.linear_model import Lasso
from sklearn.covariance import graphical_lasso
from statsmodels.api import add_constant, WLS


def fit_lfm_roblasso(x, z, p=None, nu=1e9, lambda_beta=0., lambda_phi=0., tol=1e-3, 
                     fit_intercept=True, maxiter=500, print_iter=False, rescale=False):
    """This function estimates linear factor models robustly using Lasso regularization,
    adjusting model parameters iteratively and stopping when changes are within a tolerance or 
    after a maximum number of iterations.

    Parameters
    ----------
        x : array, shape(t_bar, n_bar)
        z : array, shape(t_bar, k_bar)
        p : array, optional, shape(t_bar)
        nu : scalar, optional
        lambda_beta : scalar, optional
        lambda_phil : scalar, optional
        tol : float, optional
        fit_intercept: bool, optional
        maxiter : scalar, optional
        print_iter : bool, optional
        rescale : bool, optional

    Returns
    -------
       alpha_rmlfp : array, shape(n_bar,)
       beta_rmlfp : array, shape(n_bar, k_bar)
       sigma2_rmlfp : array, shape(n_bar, n_bar)
    """

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(z.shape) == 1:
        z = z.reshape(-1, 1)
    
    t_bar, n_bar = x.shape
    k_bar = z.shape[1]
    
    if p is None:
        p = np.ones(t_bar)/t_bar

    # rescale variables
    if rescale is True:
        sigma2_x = np.cov(x.T, aweights=p)
        sigma_x = np.sqrt(np.diag(sigma2_x))
        x = x/sigma_x

        sigma2_z = np.cov(z.T, aweights=p)
        sigma_z = np.sqrt(np.diag(sigma2_z))
        z = z/sigma_z

    # step 0: set initial values using method of moments
    if lambda_beta == 0:
        # weighted least squares regression model
        model = WLS(x, add_constant(z), weights=p).fit(fit_intercept=fit_intercept)  
        # parameters of transition equation
        alpha = np.atleast_1d(model.params[0])  # shifts
        beta = np.atleast_2d(model.params[1:])  # loadings
        eps = model.resid  # residuals
        sigma2 = np.atleast_2d(np.cov(eps.T, aweights=p))  # dispersion
    else:
        if fit_intercept is True:
            m_x = p@x
            m_z = p@z
        else:
            m_x = np.zeros(n_bar,)
            m_z = np.zeros(k_bar,)
        x_p = ((x - m_x).T*np.sqrt(p)).T
        z_p = ((z - m_z).T*np.sqrt(p)).T
        clf = Lasso(alpha=lambda_beta/(2.*t_bar), fit_intercept=False)
        clf.fit(z_p, x_p)
        beta = clf.coef_
        if k_bar == 1:
            alpha = m_x - beta*m_z
            eps = x - alpha - z*beta
        else:
            alpha = m_x - beta@m_z
            eps = x - alpha - z@np.atleast_2d(beta).T
        sigma2 = np.cov(eps.T, aweights=p)
    mu_eps = np.zeros(n_bar)
    
    for i in range(maxiter):
        # step 1: update weights
        if nu >= 1e3 and np.linalg.det(sigma2) < 1e-13:
            w = np.ones(t_bar)
        else:
            w = (nu + n_bar)/(nu + np.sum((eps - mu_eps).T*np.linalg.solve(sigma2, (eps - mu_eps).T), axis=0))
        q = w*p
        q = q/np.sum(q)

        # step 2: update location and dispersion parameters
        alpha_old, beta_old = alpha, beta
        if lambda_beta == 0:
            # weighted least squares regression model
            model = WLS(x, add_constant(z), weights=q).fit(fit_intercept=fit_intercept)  
            # parameters of transition equation
            alpha = np.atleast_1d(model.params[0])  # shifts
            beta = np.atleast_2d(model.params[1:])  # loadings
            eps = model.resid  # residuals
            sigma2 = np.atleast_2d(np.cov(eps.T, aweights=q))  # dispersion
        else:
            if fit_intercept is True:
                m_x = q@x
                m_z = q@z
            else:
                m_x = np.zeros(n_bar,)
                m_z = np.zeros(k_bar,)
            x_q = ((x - m_x).T * np.sqrt(q)).T
            z_q = ((z - m_z).T * np.sqrt(q)).T
            clf = Lasso(alpha=lambda_beta/(2.*t_bar), fit_intercept=False)
            clf.fit(z_q, x_q)
            beta = clf.coef_
            if k_bar == 1:
                alpha = m_x - beta*m_z
                eps = x - alpha - z*beta
            else:
                alpha = m_x - beta@m_z
                eps = x - alpha - z@np.atleast_2d(beta).T
            sigma2 = np.cov(eps.T, aweights=q)
        sigma2, _ = graphical_lasso((w@q)*sigma2, lambda_phi)

        # step 3: check convergence
        errors = [np.linalg.norm(alpha - alpha_old, ord=np.inf)/max(np.linalg.norm(alpha_old, ord=np.inf), 1e-20),
                  np.linalg.norm(beta - beta_old, ord=np.inf)/max(np.linalg.norm(beta_old, ord=np.inf), 1e-20)]
        # print loglikelihood and error
        if print_iter is True:
            if np.shape(sigma2) == 0:
                # univaraite student t
                lf = stats.t.logpdf(eps, nu, mu_eps, sigma2)
            else:
                # multivariate student t
                d2 = np.sum((eps - mu_eps).T*np.linalg.solve(sigma2, (eps - mu_eps).T), axis=0)
                lf = -((nu + sigma2.shape[0])/2.)*np.log(1. + d2/nu) + special.gammaln((nu + sigma2.shape[0])/2.) -\
                     special.gammaln(nu/2.) - (sigma2.shape[0]/2.)*np.log(nu*np.pi) - 0.5*np.linalg.slogdet(sigma2)[1]
            print('Iter: %i; Loglikelihood: %.5f; Errors: %.5f' %
                  (i, q@lf - lambda_beta*np.linalg.norm(beta, ord=1), max(errors)))
        if max(errors) <= tol:
            break
            
    if rescale is True:
        alpha = alpha*sigma_x
        beta = ((beta/sigma_z).T*sigma_x).T
        sigma2 = (sigma2.T*sigma_x).T*sigma_x

    return alpha, beta, sigma2
