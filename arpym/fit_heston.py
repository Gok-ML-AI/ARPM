import numpy as np
from scipy import optimize, stats, interpolate 
from collections import namedtuple

def fit_heston(tau, k, sigma_impl, r, s_0, z_0, lb=None, ub=None):
    """ This function estimates Heston model parameters to fit call option market prices.

    Parameters
    ----------
        tau : array, shape (t_bar ,) vector of times to maturity
        k : array, shape (t_bar, n_bar) strike values
        sigma_impl : array, shape (t_bar, n_bar) implied volatilities
        r : scalar, risk-free rate
        s_0 : scalar, current price of the underlying
        lb : array, shape (5 ,), lower bounds for the parameters (kappa, sigma_bar, eta, rho, sigma_0)
        up : array, shape (5 ,), upper bounds for the parameters (kappa, sigma_bar, eta, rho, sigma_0)
        z_0 : namedtuple, initial values of the parameters (kappa, sigma_bar, eta, rho, sigma_0)

    Returns
    ----------
        z : namedtuple, estimated parameters (kappa, sigma_bar, eta, rho, sigma_0)
        c_heston : array, shape (t_bar, n_bar), call option prices estimated by the Heston model
    
    """
    
    t_bar = k.shape[0]
    n_bar = k.shape[1]
    c_heston = np.zeros((t_bar, n_bar))
    kappa, sigma_bar, eta, rho, sigma_0 = z_0.kappa, z_0.sigma_bar, z_0.eta, z_0.rho, z_0.sigma_0
    
    # reparametrization to impose Feller-condition: 2*kappa@sigma_bar**2 > lam**2
    z0_tilde = np.r_[2*kappa*sigma_bar**2 - eta**2, sigma_bar, eta, rho, sigma_0]

    # constraints to impose a lower bound on parameters
    if lb is None: 
        lb = [0.1, 0, 0, -0.99, 0]
        
    # constraints to impose an upper bound on parameters
    if ub is None: 
        ub = [20, 5, 5, 0.99, 5]
    
    # objective function for Heston model
    def objective_heston(z_0, tau, k, sigma_impl, r, s_0):
        # parameters 
        sigma_bar, eta, rho, sigma_0 = z_0[1], z_0[2], z_0[3], z_0[4]
        kappa = (z_0[0] + eta**2)/(2*sigma_bar**2)

        error = np.zeros((t_bar, n_bar))
        c_heston = np.zeros((t_bar, n_bar))
        for j in range(t_bar):
            # call option price under Heston model 
            c_heston[j, :n_bar] = call_price_heston_fft(s_0, k[j, :], r, tau[j], 
                                                     np.r_[kappa, sigma_bar, eta, rho, sigma_0])
            # call option price under Black-Sholes-Merton model
            c_bsm = call_price_bsm(s_0, k[j,:], r, tau[j], sigma_impl[j, :])
            # error as difference between Black-Sholes-Merton and Heston prices
            error[j, :n_bar] = c_bsm - c_heston[j, :n_bar]
        return error.flatten()

    # optimization with least squares 
    res = optimize.least_squares(objective_heston, z0_tilde, args=(tau, k, sigma_impl, r, s_0), 
                                 bounds=(lb, ub), max_nfev=100, ftol=1e-6, verbose=0)
    # fitted parameters 
    z = namedtuple('par', 'kappa sigma_bar eta rho sigma_0')
    z.sigma_bar = res.x[1]
    z.eta = res.x[2]
    z.rho = res.x[3]
    z.sigma_0 = res.x[4]
    z.kappa = (res.x[0] + z.eta**2)/(2*z.sigma_bar**2)

    # call option price under Heston model for each maturity and strike
    for i in range(t_bar):
        for j in range(n_bar):
            c_heston[i, j] = call_price_heston_fft(s_0, k[i, j], r, tau[i],
                                                  np.r_[z.kappa, z.sigma_bar, z.eta, z.rho, z.sigma_0])
            
    return z, c_heston

def call_price_heston_fft(s_0, k, r, tau, z):
    
    """ This function computes the Heston price of European call options using FFT.
    
    Parameters
    ----------
        s_0 : scalar, current price of the underlying
        k : scalar, strike value
        r : scalar, Risk-free rate
        tau : scalar, time to maturity
        z : array, shape (5, ) Heston model parameters (kappa, sigma_bar, eta, rho, sigma_0)
    
    Returns
    ----------
        c_heston : scalar, call option prices estimated by the Heston model
    
    """
    
    # function to compute Simpson weights
    def simpson(n):
        simpson_weights = (1/3)*(3 + (-1)**(np.arange(1, n + 1, 1).T))
        simpson_weights[0] = 1/3
        return simpson_weights
    
    # characteristic function for Heston model
    def chf(v, y, d, z, t, n):
        t = t/n
        kappa, sigma_bar, eta, rho, sigma_0 = z[0], z[1]**2, z[2], z[3], z[4]**2
        A = 1j*v*((y - d)*t)
        zeta_ = -0.5*(v**2 + 1j*v)
        gamma_ = kappa - rho*eta*v*1j
        psi_bar = np.sqrt(gamma_**2 - 2*eta**2*zeta_)
        B = 2*zeta_*(1-np.exp(-psi_bar*t))*sigma_0/(2*psi_bar - (psi_bar - gamma_)*\
                                                    (1 - np.exp(-psi_bar*t)))
        C = -(kappa*sigma_bar/eta**2)*(2*np.log((2*psi_bar - (psi_bar - gamma_)*\
                                                 (1 - np.exp(-psi_bar*t)))/(2*psi_bar)) + (psi_bar - gamma_)*t)
        chf_points = np.exp(A + B + C)
        return chf_points

    # kernel function for Heston model
    def kernel(v, alpha, y, d, z, t, s):
        chf_points = chf(v - 1j*(1 + alpha), y, d, z, t, 1)
        kernel_points = np.exp(-y*t + s)*chf_points/(alpha**2 + alpha - v**2 + 1j*(2*alpha + 1)*v)
        return kernel_points

    n_fft = 1024*2
    eta = 0.250
    alpha = 1.5  # dampening factor
    lam = (2*np.pi)/(n_fft*eta)
    b = 0.5*n_fft*lam

    logstrike = -b + lam*np.arange(0, n_fft, 1).T
    min_k = int(-(np.floor((-np.log(0.65)/lam) + 1)) + n_fft/2 + 2) - 1
    max_k = int(np.floor((np.log(1.45)/lam)) + 1 + n_fft/2 + 1)
    strike_fft = s_0*np.exp(logstrike[min_k:max_k])
    v = eta*np.arange(0, n_fft, 1).T
    simpson_weights = simpson(n_fft)

    kernel_points = kernel(v, alpha, r, 0, z, tau, np.log(s_0))
    coeff = np.exp(1j*b*v)*kernel_points*simpson_weights
    price_tmp = np.fft.fft(coeff)
    price_tmp1 = np.real(price_tmp)*(np.exp(-alpha*logstrike)/np.pi)*eta
    price_tmp2 = price_tmp1[min_k:max_k]
    interp = interpolate.interp1d(strike_fft, price_tmp2, fill_value='extrapolate')
    c_heston = interp(k)
    
    return c_heston

def call_price_bsm(s, k, rf, t, v, div=0, cp=1):
    
    """
    This function computes option price under Black-Sholes-Merton model given: price, strike, risk-free rate, time and volatility. 
    For details, see here.
    """
    
    d1 = (np.log(s/k) + (rf - div + 0.5*pow(v, 2))*t)/(v*np.sqrt(t))
    d2 = d1 - v*np.sqrt(t)
    optprice = cp*s*np.exp(-div*t)*stats.norm.cdf(cp*d1) - cp*k*np.exp(-rf*t)*stats.norm.cdf(cp*d2)
    
    return optprice

def implvol_bsm(p, s, k, rf, t, div=0, cp=1):
    
    """
    This function computes impled volatility under Black-Sholes-Merton model, given: price, forward, strike, risk-free rate and time. 
    For details, see here.
    """
    
    f = lambda x: call_price_bsm(s, k, rf, t, x, div, cp) - p
    result = optimize.brentq(f, 1e-9, 1e+9)
    
    return result
