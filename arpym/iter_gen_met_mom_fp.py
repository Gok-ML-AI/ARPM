import numpy as np
from numpy.linalg import solve
from scipy.optimize import minimize


def iter_gen_met_mom_fp(eps, p, model, order=2):
    """This function computes the generalized method of moments with flexible probabilities 
       estimate for the parameter lam of an invariant which is Poisson distributed

    Parameters
    ----------
        eps : vector, shape (1, t_end) 
        p : vector, shape (1, t_end) 
        model : string
        order : scalar, optional

    Returns
    -------
        lam : scalar
    """

    if model=='Poisson':
        max_iter = 100
        lam = np.mean(eps)*np.ones(max_iter)
        conv = 0
        i = 1
        t_bar = p.shape[1]
        omega_2 = np.eye(2)  # initial weighting matrix
        a = (eps - lam[0])
        b = (eps**2) - lam[0]*(lam[0] + 1)
        v_lamda = np.r_[p@a.T, p@b.T]  # initial vector in weighting matrix
        quadform = v_lamda.T@omega_2@v_lamda@np.ones((1, max_iter))  # initial quadratic form in weighting matrix

        while i < max_iter and conv == 0:
            # update GMMFP estimate of the Poisson distribution parameter 
            lam[i] = gmm_poisson(eps, p, omega_2, lam[i - 1])  
            # update weighting matrix
            a = (eps - lam[i])
            b = (eps**2) - lam[i]*(lam[i] + 1)
            v_lamda = np.r_[p@a.T, p@b.T]
            rhs = np.r_[np.r_['-1', p@(a**2).T, p@(a*b).T], np.r_['-1', p@(a*b).T, p@(b**2).T]]
            omega_2 = solve(rhs, np.eye(rhs.shape[0]))
            # shrinkage towards identity of weighting matrix
            aa = np.sqrt(p@(a**2).T)
            bb = np.sqrt(p@(b**2).T)
            c = (omega_2/np.r_[np.r_['-1',aa**2, aa*bb], np.r_['-1',aa*bb, bb**2]])
            omega_2 = 0.5*np.diagflat(np.r_['-1',aa, bb])@c@np.diagflat(np.r_['-1',aa, bb]) + 0.5*np.eye(2) # new weighting matrix
            # convergence check
            quadform[0, i] = v_lamda.T@omega_2@v_lamda
            reldistance = abs((quadform[0, i] - quadform[0, i - 1])/quadform[0, i - 1])
            if reldistance < 10**-8:
                conv=1
            i = i + 1
        lam1 = lam[i - 1]
        
    return lam1


def gmm_poisson(eps, p, omega2, lam_0):
    """This function solves the minimization problem

    Parameters
    ----------
        eps : vector, shape (1, t_end) 
        p : vector, shape (1, t_end) 
        omega2 : matrix, shape (2, 2)
        lam_0 : scalar

    Returns
    -------
        lam : scalar
    """
    def gmm_p(lam_0, omega2, eps, p):
        a = (eps - lam_0)
        b = (eps**2) - lam_0*(lam_0 + 1)
        v = np.r_[p@a.T, p@b.T]
        f = v.T@omega2@v
        return f.squeeze()
    
    options = {'maxiter' : 5000}
    # solve the minimization problem
    lam = minimize(gmm_p, lam_0, args=(omega2, eps, p), options=options, tol=1e-8)
    return lam.x
