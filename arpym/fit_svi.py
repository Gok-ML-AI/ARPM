import numpy as np 
from scipy import interpolate, optimize, special, stats
from collections import namedtuple

def fit_sigma_svi(tau, delta, sigma_delta, y, theta_var_ATM_start, theta_phi_start, m_grid=100):
    
    """ This function calibrates the theta_1,...,theta_6 parameters of the SVI model such
        that the theoretical volatility surface curve best match the observed
        volatility surface.
        
     Parameters
    ----------
         tau : array, shape (n_bar, 1) times to maturity
         delta : array, shape (1, k_bar) delta-moneyness corresponding to columns of sigma_delta
         sigma_delta : array, shape (n_bar, k_bar) observed volatility surface
         y: array, shape (n_bar x 1) risk-free rates corresponding to times to maturity in tau
         theta_var_ATM : named tuple, shape (3,) starting parameters for fitting, 
                         named respectively, theta_1, theta_2, theta_3
         theta_phi : named tuple, shape (3,) starting parameters for fitting, 
                     named respectively, theta_4, theta_5, theta_6
     Returns
    ----------
         theta_var_ATM : array, shape (3,) fitted SVI parameters, respectively, theta_1, theta_2, theta_3
         theta_phi : array, shape (3,) fitted SVI parameters, respectively, theta_4, theta_5, theta_6
    """    
    
    n_bar = len(tau)
    k_bar = len(delta)
    if isinstance(y, float) or isinstance(y, int):
        m_ATM = -y*np.sqrt(tau)
    else:
        m_ATM = -y@np.sqrt(tau)

    # compute ATM forward m-moneyness
    sigma_m_ATM = np.zeros((n_bar, 1))
    for i in range(n_bar):
        m_data = stats.norm.ppf(delta)*sigma_delta[i,:] - (y + sigma_delta[i,:]**2/2)*np.sqrt(tau[i])
        # equispaced grid of m-moneyness
        m = np.min(m_data) + (np.max(m_data) - np.min(m_data))*np.arange(m_grid + 1)/m_grid 
        interp = interpolate.interp1d(m_data.flatten(), sigma_delta[i,:].flatten(), fill_value='extrapolate')
        # implied volatility as a function of m-moneyness
        sigma_m = interp(m.flatten())
        pchip = interpolate.interp1d(m, sigma_m)
        # ATM forward variance
        sigma_m_ATM[i,0] = pchip(m_ATM[i])

    # fit ATM forward variance
    var_mkt_ATM = sigma_m_ATM**2
    
    # initialization var_ATM
    var_ATM = np.zeros((len(tau), 1))
    
    # starting parameters 
    par_start = np.array([theta_var_ATM_start.theta1, theta_var_ATM_start.theta2, theta_var_ATM_start.theta3])
    
    # estimation 
    res = optimize.least_squares(objective_atm_svi, par_start, args=(tau, var_mkt_ATM), ftol=1e-10, 
                                 xtol=1e-10, max_nfev=2*600, verbose=0)
    # estimated parameters 
    theta_var_ATM = namedtuple('theta',['theta1', 'theta2', 'theta3'])
    theta_var_ATM.theta1 = res.x[0]
    theta_var_ATM.theta2 = res.x[1]
    theta_var_ATM.theta3 = res.x[2]
    var_ATM = res.x[0]*tau**2 + res.x[1]*tau + res.x[2]
    var_model_ATM = var_ATM/tau
    
    # optimization informations 
    ATMexitFlag = res.status
    ATMresNorm = res.optimality

    # from delta-moneyness to m-moneyness
    m = np.tile(stats.norm.ppf(delta),(n_bar, 1))*sigma_delta - (y + sigma_delta**2/2)*\
        np.tile(np.sqrt(tau[...,np.newaxis]),(1, k_bar))
     
    # starting parameters 
    par_start = [theta_phi_start.theta4, theta_phi_start.theta5, theta_phi_start.theta6]
    # estimation 
    res = optimize.least_squares(objective_phi_svi, par_start, args=(tau, m, y, theta_var_ATM, sigma_delta), 
                                 ftol=1e-9,xtol=1e-9, max_nfev=2*600)
    # estimated parameters 
    theta_phi = namedtuple('theta', ['theta4', 'theta5', 'theta6'])
    theta_phi.theta4 = res.x[0]
    theta_phi.theta5 = res.x[1]
    theta_phi.theta6 = res.x[2]
    
    # optimization informations 
    exitFlag = res.status
    resNorm = res.optimality
    
    # output 
    output_fit = namedtuple('output','ATMexitFlag ATMresNorm exitFlag_phi resNorm')
    output_fit.ATMexitFlag = ATMexitFlag
    output_fit.ATMresNorm = ATMresNorm
    output_fit.exitFlag_phi = exitFlag
    output_fit.resNorm = resNorm
    
    return theta_var_ATM, theta_phi

def map_svi_params(p_var_ATM, p_phi):
    
    """ This function converts the theta_1,...,theta_6 parametrization of the SVI model
        into its original parametrization a, b, c,rho, eta, gamma, found in equation (5.1) of the paper 
        "Arbitrage-free SVI volatility surfaces" by Jim Gatheral and Antoine Jacquiery, version April 6, 2012.
        
     Parameters
    ----------
         p_var_ATM : array, shape (3,) its three components contain, respectively, theta_1, theta_2, theta_3
         p_phi : array, shape (3,) its three components contain, respectively, theta_4, theta_5, theta_6
         
     Returns
    ----------
         par_var_ATM : array, shape (3,) its three components contain, respectively, a, b, c.
         par_phi : array, shape (3,) its three components contain, respectively, rho, eta, gamma.
    """    
    
    # initial parametrization 
    theta_1 = p_var_ATM[0]
    theta_2 = p_var_ATM[1]
    theta_3 = p_var_ATM[2]
    theta_4 = p_phi[0]
    theta_5 = p_phi[1]
    theta_6 = p_phi[2]
    
    # original parametrization 
    a = (np.exp(theta_1) + theta_2)/4
    b = (np.exp(theta_1) - 3*theta_2)/4
    c = theta_3 - a/5
    rho = special.erf(theta_4)
    eta = (special.erf(theta_5) + 1)*1.7/2
    gamma = (special.erf(theta_6) + 1)/2

    par_var_ATM = np.array([a, b, c])
    par_phi = np.array([rho, eta, gamma])

    return par_var_ATM, par_phi

def sigma_svi(tau, m, y, theta_var_ATM, theta_phi):
    
    """ This function computes the value of the volatility according to the SVI model 
        given its theta_1,...,theta_6 parametrization. It is based on the SVI volatility
        surface presented in equation (5.1) of the paper Arbitrage-free SVI volatility surfaces
        by Jim Gatheral and Antoine Jacquiery, version April 6, 2012.

    Parameters
    ----------
        tau : array, shape (n_bar, 1) times to maturity
        m : array, shape (n_bar, k_bar) m(i,j) correspond to the value of m-moneyness 
            for time to maturity tau[i] and an implicit value of strike k[j]
        y: array, shape (n_bar x 1) risk-free rates corresponding to times to maturity in tau
        theta_var_ATM : array, shape (3,) its three components contain, respectively, theta_1, theta_2, theta_3
        theta_phi : array, shape (3,) its three components contain, respectively, theta_4, theta_5, theta_6

    Returns
    ----------
        sigma_model : array, shape (n_bar, k_bar) volatility obtained from the SVI model, corresponding to vaues in m.
    """    
    
    # from theta parameterization to original parameterization
    arg1 = [theta_var_ATM.theta1, theta_var_ATM.theta2, theta_var_ATM.theta3]
    arg2 = [theta_phi.theta4, theta_phi.theta5, theta_phi.theta6]
    par_var_ATM, par_phi = map_svi_params(arg1, arg2)

    # parameters 
    a = par_var_ATM[0]
    b = par_var_ATM[1]
    c = par_var_ATM[2]
    rho = par_phi[0]
    eta = par_phi[1]
    gamma = par_phi[2]
    
    # times to maturities 
    tau = np.tile(tau[...,np.newaxis], (1, m.shape[1])) 
    # at-the-money forward variance
    lam = a*tau**2 + b*tau + c 
    phi = eta/((lam**(gamma))*((1 + lam)**(1 - gamma)))

    # convert m-moneyness to k-strike
    k = -np.sqrt(tau)*m - y*tau

    # svi formula for sigma
    sigma_model = np.sqrt((lam/2*(1 + rho*phi*k + np.sqrt((phi*k + rho)**2 + (1 - rho**2))))/tau)
    
    return sigma_model

def objective_atm_svi(par, tau, var_mkt_ATM):
    
    """ This is the objective function to calibrates the theta_1, theta_2, theta_3 parameters of the SVI model 
        such that the theoretical variance curve best match the observed at-the-money variance curve.
        Notice that for the ATM variance, only the first three parameters are needed.
        
     Parameters
     ----------
        par: array, shape(3,), first three parameters of the svi model  
        tau : array, times to maturities
        var_mkt_ATM: array, observed at-the-money variance curve

     Returns
     -------
        objective : scalar
    """
    
    par_tmp, _ = map_svi_params(par, np.zeros(3))
    a = par_tmp[0]
    b = par_tmp[1]
    c = par_tmp[2]
    var_ATM = a*tau**2 + b*tau + c
    objective = var_mkt_ATM.flatten() - var_ATM/tau
    
    return objective

def objective_phi_svi(p, tau, m, y, theta_var_ATM, sigma):
    
    """ This is the objective function to calibrates the theta_4, theta_5, theta_6 parameters of the SVI model 
        such that the theoretical volatility surface best match the observed volatility surface.
        Notice that the theta_1, theta_2, theta_3 parameters of the SVI model are a required input.
        
     Parameters
     ----------
        par: array, shape(3,), first three parameters of the svi model  
        tau : array, times to maturities
        var_mkt_ATM: array, observed at-the-money variance curve

     Returns
     -------
        objective : scalar
    """
    
    par = namedtuple('theta', ['theta4','theta5','theta6'])
    par.theta4 = p[0]
    par.theta5 = p[1]
    par.theta6 = p[2]
    sigma_model = sigma_svi(tau,m, y, theta_var_ATM, par)
    objective = (sigma_model - sigma).flatten()
    
    return objective
