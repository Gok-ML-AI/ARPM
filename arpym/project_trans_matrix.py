# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import expm, logm
import cvxpy as cp

from arpym.min_rel_entropy_sp import min_rel_entropy_sp

def project_trans_matrix(p, delta_t, credit=False):
    """This function computes the projected transition matrix, 
    considering optional credit constraints

    Parameters
    ----------
        p : array, shape (n_bar, n_bar)
        delta_t : scalar
        credit : boolean (optional)

    Returns
    -------
        p_delta_t : array, shape (n_bar, n_bar)

    """

    # Step 1: Compute log-matrix
    l = logm(p)    # log-matrix of p

    # Step 2: Compute generator
    n_bar = len(p)  # dimension of transition matrix
    x = cp.Variable(n_bar*n_bar)  # optimization variable (vectorized generator g)
    
    p = np.eye(n_bar*n_bar)  # identity matrix of size n*n (quadratic term)
    q = -l.flatten()  # flattened linear term
    objective = cp.Minimize(0.5*cp.quad_form(x, p) + q.T@x)

    # box constraints: G*x <= h (off-diagonal elements of g must be non-negative)
    G = -np.diag(1*np.array(np.mod(np.arange(n_bar**2), n_bar + 1) != 0))
    h = np.zeros(n_bar**2)
    # equality constraints: a*x = b (rows of g must sum to zero)
    a = np.repeat(np.eye(n_bar), n_bar, axis=1)
    b = np.zeros(n_bar)
    # combine constraints
    constraints = [# box constraints
               G@x <= h,  
               # equality constraints
               a@x == b]
    # solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)
    # generator matrix
    g = x.value.reshape(n_bar, n_bar)

    # Step 3: Compute projected transition matrix
    g = np.array(g).reshape((n_bar, n_bar)) 
    p_delta_t = expm(delta_t*g)  # projected transition matrix

    if credit is True:
        p = p_delta_t
        # probability constraint
        a_eq = np.ones((1, n_bar))  # 1×n_bar dimensional vector of ones
        b_eq = np.array([1])
        # initialize monotonicity constraint
        a_ineq = {}
        a_ineq[0] = np.diagflat(np.ones((1, n_bar-1)), 1) - np.diagflat(np.ones((1, n_bar)), 0)  # (n_bar-1)×n_bar upper triangular matrix
        a_ineq[0] = a_ineq[0][:-1]
        b_ineq = np.zeros((n_bar-1))  # 1×(n_bar-1) dimensional vector of ones
        p_delta_t = np.zeros((n_bar, n_bar))
        for c in range(n_bar - 1):
            p_delta_t[c, :] = min_rel_entropy_sp(p[c, :], a_ineq[c], b_ineq, a_eq, b_eq, False)  # minimize relative entropy
            a_temp = a_ineq.get(c).copy()  # update monotonicity constraint
            a_temp[c, :] = -a_temp[c, :]
            a_ineq[c+1] = a_temp.copy()
        p_delta_t[-1, :] = np.zeros((1, p.shape[1]))  # default constraint
        p_delta_t[-1, -1] = 1
        
    return p_delta_t
