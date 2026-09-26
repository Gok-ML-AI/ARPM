import numpy as np
from scipy import optimize

def fit_stochastic_volatility_model(y, init_par):
    
    """Maximum likelihood estimation of the stochastic volatility model parameters.
    Reference: Shumway and Stoffer, Time Series Analysis and Its Applications: With R Examples, example 6.18.

    Parameters
    ----------
        par : array, shape(k_bar, ) parameters of the stochastic volatility model in order (phi0, phi1, sQ, alpha, sR0, mu1, sR1)
        y : array, shape (1, t_bar) time series of log of squared returns
    Returns
    ----------
        object.x: optimal parameters
        object.fun
        object.status
        object: response from scipy.optimize.minimize
    """
    
    def objective(par, y):
        like, _ = filter_stochastic_volatility(y, par)  # likelihood
        return like
    
    res = optimize.minimize(objective, init_par, args=(y,), options={'maxiter':6*500})
    
    return res.x, res.fun, res.status, res

def filter_stochastic_volatility(y, par):
    
    """This function calculates the negative log-likelihood and estimates 
    the log of squared hidden volatility based on a stochastic volatility model.

    Parameters
    ----------
        y : array, shape (1, t_bar) time series of log of squared returns
        par : array, shape(k_bar, ) parameters of the stochastic volatility model in order (phi0, phi1, sQ, alpha, sR0, mu1, sR1)
 
    Returns
    ----------
        likelihood : scalar, negative log likelihood
        h : array, shape (1, t_bar) log of squared hidden volatility
    """
    
    # parameters 
    phi0 = par[0]
    phi1 = par[1]
    sq = par[2]
    alpha = par[3]
    sr0 = par[4]
    mu1 = par[5]
    sr1 = par[6] 
    pf = (sq**2)/(1 - phi1)   
    if pf < 0:
        pf = 0  # not negative
    
    t_bar = len(y)  # length of time series
    h = np.zeros(t_bar)  # array for log-hidden volatility
    xf = 0  # log-hidden volatility 
    pi_0 = 0.5  # probability state 0 
    pi_1 = 0.5  # probability state 1
    likelihood = 0  # likelihood
    for i in range(t_bar):
        h[i] = phi1*xf + phi0  # fitted log-hidden volatility 
        pp = phi1*pf*phi1 + (sq**2)   
        # residuals volatility 
        sigma_0 = pp + (sr0**2)  # state-0  
        sigma_1 = pp + (sr1**2)  # state-1
        k_0 = pp/sigma_0
        k_1 = pp/sigma_1
        # residuals
        e_0 = y[i] - h[i] - alpha  # state-0  
        e_1 = y[i] - h[i] - mu1 - alpha  # state-1
        # densities 
        f_0 = (1/np.sqrt(sigma_0))*np.exp(-0.5*(e_0**2)/sigma_0)  # state-0
        f_1 = (1/np.sqrt(sigma_1))*np.exp(-0.5*(e_1**2)/sigma_1)  # state-1  
        # conditional probabilities
        f_pi_1 = pi_1*f_1/(pi_1*f_1 + pi_0*f_0)  # state-0
        f_pi_0 = pi_0*f_0/(pi_1*f_1 + pi_0*f_0)  # state-1
        # update log-hidden volatility 
        xf = h[i] + f_pi_1*k_1*e_1 + f_pi_0*k_0*e_0  
        pf = f_pi_1*(1 - k_1)*pp + f_pi_0*(1 - k_0)*pp
        likelihood = likelihood - 0.5*np.log(pi_1*f_1 + pi_0*f_0)  # likelihood 
        
    return likelihood, h
