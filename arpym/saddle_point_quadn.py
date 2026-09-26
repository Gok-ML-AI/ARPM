# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def saddle_point_quadn(y, alpha, beta, gamma, mu, sigma2):
    """This function computes the cumulative distribution function (CDF) and probability density function (PDF) 
       of a saddle point approximation for a quadratic form

    Parameters
    ----------
        y : array, shape(j_bar,)
        alpha : scalar
        beta : array, shape(n_bar,)
        gamma : array, shape(n_bar, n_bar)
        mu : array, shape(n_bar,)
        sigma2 : array, shape(n_bar, n_bar)

    Returns
    -------
        cdf : array, shape(j_bar,)
        pdf : array, shape(j_bar,)
    """

    y = np.asarray(y).copy().reshape(-1)
    beta = np.asarray(beta).copy().reshape(-1, 1)
    mu = np.asarray(mu).copy().reshape(-1, 1)
    j_bar = len(y)

    # eigenvalues and eigenvectors of l.T@gamma@l
    l = np.linalg.cholesky(sigma2)
    lam, e = np.linalg.eig(l.T@gamma@l)
    lam = lam.reshape(-1, 1)

    # transformed parameters
    alpha_tilde = alpha + beta.T@mu + mu.T@gamma@mu
    beta_tilde = beta + 2*gamma@mu
    gamma_tilde = e.T@l.T@beta_tilde

    # log-characteristic function
    def c_y(w):
        return alpha_tilde*w - 0.5*np.sum(np.log(1 - 2.*w*lam) - w**2*gamma_tilde**2/(1 - 2.*w*lam))

    # first derivative
    def c_y_prime(w):
        return alpha_tilde + np.sum(lam/(1 - 2.*w*lam) + gamma_tilde**2*(w - w**2*lam)/(1 - 2.*w*lam)**2)

    # second derivative
    def c_y_second(w):
        return np.array([np.sum(2.*(lam/(1 - 2.*w*lam))**2 + gamma_tilde**2/(1 - 2.*w*lam)**3)])

    # find w_hat numerically using Brent's method
    lam_max = np.max(lam)
    lam_min = np.min(lam)
    if lam_max > 0:
        w_max = (1 - 1e-5)/(2*lam_max)
    else:
        w_max = 1e20

    if lam_min < 0:
        w_min = (1 + 1e-5)/(2*lam_min)
    else:
        w_min = -1e20
    y_min = c_y_prime(w_min)
    y_max = c_y_prime(w_max)

    # initialize
    w_hat = np.zeros(j_bar)
    c_y_w_hat = np.zeros(j_bar)  # c(w_hat)
    c_y_second_w_hat = np.zeros(j_bar)  # c''(w_hat)

    idx = np.argsort(y)
    w_last = w_min
    for j in range(j_bar):
        if y[idx[j]] <= y_min:
            w_hat[idx[j]] = w_min
        elif y[idx[j]] >= y_max:
            w_hat[idx[j]] = w_max
        else:
            # Brent’s method for finding the root of the function.
            # Since y is sorted and c_y_prime is a monotone increasing function
            # it is guaranteed that the solution w is in the interval
            # [w_last, w_max]
            w_hat[idx[j]] = brentq(lambda w: c_y_prime(w) - y[idx[j]], w_last, w_max)
            w_last = w_hat[idx[j]]

        c_y_w_hat[idx[j]] = c_y(w_hat[idx[j]])
        c_y_second_w_hat[idx[j]] = c_y_second(w_hat[idx[j]])

    # cdf and pdf
    r = np.sign(w_hat)*np.sqrt(2.*(w_hat*y - c_y_w_hat))
    u = w_hat*np.sqrt(c_y_second_w_hat)
    cdf = norm.cdf(r) - norm.pdf(r)*(1./u - 1./r)
    pdf = np.exp(c_y_w_hat - w_hat*y)/np.sqrt(2*np.pi*c_y_second_w_hat)

    return np.squeeze(cdf), np.squeeze(pdf)
