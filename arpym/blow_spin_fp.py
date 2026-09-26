import numpy as np

def blow_spin_fp(z, b_bar, s_bar, blowscale=[.1, 3], spinscale=1, method='inverse'):
    """ This function generates a pool of flexible probabilitiy profiles for the 
        dataset z based on a bivariate Gaussian kernel.
    
    Parameters
    ----------
        z: array, shape (2, t_bar)
        b_bar: scalar, number of blow frames (kernels with fixed center, "increasing" dispersion)
        s_bar: scalar, number of spin frames (kernels with fixed dispersion matrix, centers arranged around a circle)
        blowscale: array, shape (1, 2) min and max scale for the blow frames
        spinscale: scale for the spin frames
        method: string, can be direct, flexible probabilitiy profiles are proportional to the normal pdf, or 
                inverse, default, flexible probabilitiy profiles are proportional to (1 - normal PDF))

    Returns
    -------
         p: array, shape (b_bar, s_bar) flexible probability profiles
         ens: array, shape(1 x (b_bar + s_bar)) effective number of scenarios for each flexible probability profile
    """
    
    q_bar = b_bar + s_bar  # total number of FP profiles
    t_bar = z.shape[1]  # number of observations
    mu = np.mean(z, 1, keepdims=True)
    sigma2 = np.cov(z)

    ens = np.zeros((1, q_bar))  
    p = np.zeros((q_bar, t_bar))  

    # blow
    if b_bar == 1:
        b_scale = [blowscale[1]]
    else:
        b_scale = np.linspace(blowscale[0], blowscale[1], b_bar)

    for b in range(b_bar):
        for t in range(t_bar):
            z2 = b_scale[b]*(z[:, [t]] - mu).T@np.linalg.solve(sigma2,z[:, [t]] - mu)
            if method == 'direct':
                p[b, t] = np.exp(-z2)
            else:
                p[b, t] = 1 - np.exp(-z2)
        p[b, :] = p[b, :]/np.sum(p[b, :])
        # p[b, p == 0] = 10**(-250)  # avoid log(0) in ens computation
        ens[0, b] = np.exp(-p[b, :]@np.log(p[b, :].T))
        
    # spin
    if s_bar > 0:
        n_bar = sigma2.shape[0]
        lam, e = np.linalg.eig(sigma2)
        lam, idx = np.sort(lam), np.argsort(lam)
        e = e[:, idx]
        lam = np.tile(lam[n_bar - 1, np.newaxis],(n_bar, 1))
        angle = np.arange(0, 2*np.pi + 2*np.pi/s_bar, 2*np.pi/s_bar)
        angle[-1] = 0
        circle = e@np.diagflat(np.sqrt(lam))@np.array([[np.cos(angle[0])], [np.sin(angle[0])]])
        for i in range(1, s_bar):
            y = np.array([[np.cos(angle[i])], [np.sin(angle[i])]])
            circle = np.r_['-1', circle, e@np.diagflat(np.sqrt(lam))@y]
        m = mu@np.ones((1, s_bar)) + spinscale*circle
    else:
        m = []

    for q in range(b_bar, q_bar):
        for t in range(t_bar):
            z2 = (z[:, t] - m[:, q - b_bar]).T@np.linalg.solve(sigma2, z[:, t] - m[:, q - b_bar])
            p[q, t] = np.exp(-z2)
        p[q, :] = p[q, :]/sum(p[q, :])  # matrix containing the q_bar FP-profiles
        ens[0, q] = np.exp(-p[q, :]@np.log(p[q, :].T))  # effective number of scenarios

    return p, ens
