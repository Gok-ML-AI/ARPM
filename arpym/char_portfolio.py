import numpy as np
from scipy.optimize import minimize


def char_portfolio(beta, sig2_pl, h_0=None, h_tilde=None, sigma2_max=None, v=None, v_0=None):
    """This function computes the characteristic portfolio for a given set of parameters, considering 
    constraints such as zero correlation, budget, and upper bound on variance. It solves for the characteristic portfolio either 
    with or without additional constraints, such as zero correlation or upper bounds on variance.

    Parameters
    ----------
        beta : array, shape(n_bar, )
        sig2_pl : array, shape(n_bar, n_bar)
        k_ : int
        h_0: array, optional, shape(n_bar, )
        h_tilde: array, optional, shape(n_bar, )
        sigma_2_max: scalar, optional
        v: array, shape(n_bar, )
        v_0: scalar

    Returns
    -------
        h_char : array, shape(n_bar, )

    """

    beta = beta.reshape(-1, 1)
    if (v is None and h_tilde is None and h_0 is None and
            v_0 is None and sigma2_max is None):
        h_char = np.linalg.solve(sig2_pl, beta)/(beta.T@np.linalg.solve(sig2_pl, beta))
    else:
        # zero correlation constraint
        if h_tilde is not None:
            aeq = h_tilde.reshape(1, -1)@sig2_pl
            beq = np.array([[0]])
        # budget constraint
        if v_0 is not None:
            aeq = np.r_[aeq, v.reshape(1, -1)]
            beq = np.r_[beq, np.atleast_2d(v_0)]
        eqcon = lambda x, A, b: -((A@x) - b).flatten()
        opts = {'maxiter': 10**6, 'disp': False}

        # upper bound constraint on variance
        cons = ({'type': 'ineq', 'fun': lambda h, sig2_pl,
                 sigma2_up: -(h.T@sig2_pl@h-sigma2_up),
                 'args': (sig2_pl, sigma2_max)},
                {'type': 'ineq', 'fun': lambda h,
                 sig2_pl: -(-h.T@sig2_pl@h + 10**-6), 'args': (sig2_pl, )},
                {'type': 'eq', 'fun': eqcon, 'args': (aeq[0], beq[0])},
                {'type': 'eq', 'fun': eqcon, 'args': (aeq[1], beq[1])})
        objfun = lambda h, beta: (-h@beta)
        res = minimize(objfun, h_0, method='SLSQP', args=(beta,),
                       jac=False, constraints=cons, options=opts)
        h_char = res.x
        
    return h_char.flatten()
