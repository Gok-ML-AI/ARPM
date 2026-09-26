# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate, stats


def call_option_value(t_hor, x_s_thor, x_y_thor, tau_y, x_sig_thor, m_sig, tau_sig, k_strk,
                      t_end, sr=1, logsig=1, eta=0.013):
    """This function calculates the value of a call option at a given time horizon using various parameters 
    such as maturities, volatilities, strike price, and others. It computes moneyness, yield or shadow yield, 
    implied volatility, and then computes the call option value based on the Black-Scholes formula, 
    returning the option value as a scalar.

    Parameters
    ----------
        t_hor : date
        x_s_thor : scalar
        x_y_thor : scalar
        tau_y : shape(k_bar,)
        x_sig_thor : array, shape(k_bar*l_bar,) or shape(k_bar, l_bar)
        m_sig : shape(l_bar,)
        tau_sig : shape(k_bar,)
        k_strk : scalar
        t_end : date
        sr : boolean {0,1}
        logsig : boolean {0,1}
        eta : scalar

    Return
    ------
        v : scalar
    """

    x_sig_thor = x_sig_thor.reshape(-1)

    if (sr != 0) and (sr != 1):
        sr = 1

    if (logsig != 0) and (logsig != 1):
        logsig = 1

    # time to expiry of the call option at the horizon
    tau_star = np.busday_count(t_hor, t_end)/252

    # compute m moneyness
    m_star = np.log(np.exp(x_s_thor)/k_strk)/np.sqrt(tau_star)

    # compute yield/shadow yield for the time to expiry
    if x_y_thor.shape[0] == 1:
        x_y_star = x_y_thor
    else:
        interp = interpolate.interp1d(tau_y.flatten(), x_y, axis=1, fill_value='extrapolate')
        x_y_star = interp(tau_star)
    if sr == 1:
        x_y_star_tmp = np.zeros((x_y_star.shape))
        x_y_star_tmp[x_y_star >= eta] = x_y_star[x_y_star >= eta]
        x_y_star_tmp[x_y_star < eta] = eta*np.exp(x_y_star[x_y_star < eta]/eta - 1)
        x_y_star = x_y_star_tmp

    # (Log-)implied volatility at the horizon moneyness and time to expire
    points = list(zip(*[grid.flatten() for grid in np.meshgrid(*[tau_sig, m_sig])]))
    m_e = min(max(m_star, min(m_sig)), max(m_sig))  # extrapolation
    t_e = min(max(tau_star, tau_sig[0]), tau_sig[-1])  # extrapolation
    x_sig_star = interpolate.LinearNDInterpolator(points, x_sig_thor)(*np.r_[t_e, m_e])
    if logsig == 1:
        x_sig_star = np.exp(x_sig_star)

    # compute call option value
    d1 = (m_star + (x_y_star + (x_sig_star**2)/2)*np.sqrt(tau_star))/x_sig_star
    d2 = d1 - x_sig_star*np.sqrt(tau_star)
    v_call = np.exp(x_s_thor)*(stats.norm.cdf(d1) - np.exp(-(m_star*np.sqrt(tau_star) +\
                                                             x_y_star*tau_star))*stats.norm.cdf(d2))

    return v_call
