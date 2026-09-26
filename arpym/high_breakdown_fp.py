import numpy as np

def high_breakdown_fp(eps, p, last=0, c=0.5):
    
    """This function computes the High Breakdown Point with Flexible Probabilities estimators of
    location and dispersion, using an iterative algorithm.

    Parameters
    ----------
        eps : array, shape (i_bar, t_bar)
        p : array, shape (, t_bar)
        last : scalar, if last!=0 only the last computed mean and covariance are returned
        c : scalar, enclosed probability: the ellipsoid includes a probability of at least c 
            the breakdown point is approx. 1-c (when effective number of scenarios is large)
    
    Returns
    ----------
        mu_HBFP : array, shape (i_bar, k_bar) contains the mean vectors computed at each iteration
        sigma2_HBFP : array, shape (i_bar, i_bar, k_bar) contains the covariance matrices computed at each iteration
        p_tilde : array, shape (1, k_bar) vector containing the probability enclosed by the ellipsoid at each iteration
        v : array, shape (1, k_bar) vector containing the volumes of the ellipsoids computed at each iteraion
        t_out: array, shape (1, k_bar) vector containing the index of the outliers detected at each iteration
    """
    
    i_bar, t_bar = eps.shape
    p_tilde = np.zeros((1, 1))
    mu_hbfp = np.zeros((i_bar, 1))
    sigma2_hbfp = np.zeros((i_bar, i_bar, 1))
    v = np.zeros(1)
    t_out = np.zeros(1)

    k = 0
    t = np.arange(t_bar)
    while k == 0 or p_tilde[k] >= c:
        # compute probability
        p_tilde = np.r_[p_tilde, np.sum(p,keepdims=True)]
        
        # fit high-breakdown estimators of mean and covariance 
        j = 0
        det1 = 0
        k = k + 1
        t_bar = eps.shape[1]
        w = np.ones((1, t_bar))/t_bar
        mu = np.mean(eps, axis=1, keepdims=True)
        mu_mve = mu
        sigma2 = np.cov(eps)
        sigma2_mve = sigma2[..., np.newaxis]
        keep_loop = 1  # loop condition 
        while keep_loop:
            j = j + 1
            # z-scores
            z_score = np.sum((eps - mu)*np.linalg.solve(sigma2,(eps - mu)), axis=0)
            # update weights
            update = np.where(z_score > 1)
            w[0,update] = w[0, update]*z_score[update]
            # update mean vector
            mu = (eps@w.T)/np.sum(w)  
            mu_mve = np.r_['-1', mu_mve, mu]
            # update covariance matrix
            sigma2 = (eps - np.tile(mu, (1, t_bar)))@np.diagflat(w)@(eps - np.tile(mu, (1, t_bar))).T
            sigma2_mve = np.r_['-1', sigma2_mve, sigma2[..., np.newaxis]]
            # check convergence
            keep_loop = 1 - np.prod(z_score <= 1)     
            
        # high-breakdown estimators of mean and covariance 
        mu_hbfp = np.r_['-1',mu_hbfp, mu_mve[:, -1][..., np.newaxis]]
        sigma2_hbfp = np.r_['-1', sigma2_hbfp, sigma2_mve[:, :, -1][..., np.newaxis]]

        # compute volume
        v = np.r_[v, np.linalg.det(sigma2_hbfp[:, :, k])]

        # detect outliers
        # mean vector and covariance matrix
        mu = np.mean(eps, axis=1, keepdims=True)
        sigma2 = np.cov(eps)
        # normalize observations
        t_bar = eps.shape[1]
        z = np.sqrt(1/t_bar)*(eps - np.tile(mu, (1, t_bar))).T
        # compute FP-info matrix
        h = z@np.linalg.solve(sigma2, z.T)
        # determine singularities
        h = np.diag(h).T
        # exclude singularities
        h = np.where(h == 1, 0, h)
        t_index = np.where(h == 1, 0, range(t_bar))
        # outlier probability
        p = np.where(h == 1, 0, p)
        # outlier
        a = (1 - h)/(1 - p)
        # outlier position index 
        t_tilde = t_index[a[0] == np.min(a)]

        # remove outlier
        t_out = np.r_[t_out, t[t_tilde]]
        t = np.delete(t, t_tilde)
        eps = np.delete(eps, t_tilde, axis=1)
        p = np.delete(p, t_tilde, axis=1)
        print('\r{:.2f} %'.format(min(100*(1 - p_tilde[-1]).squeeze()/(1 - c), 100)), end='', flush=True)
            
    # when p_tilde < c return the output for index k - 1 so that the enclosed probability for the output is p_tilde>=c
    # mean vectors computed at each iteration
    mu_hbfp = mu_hbfp[:, :-1]
    # covariance matrix computed at each iteration
    sigma2_hbfp = sigma2_hbfp[:, :, :-1]
    # probability
    p_tilde = p_tilde[:-1]
    # volume
    v = v[:-1]
    # outliers computed at steps k and (k-1) are included in ellipsoid so they have to be removed 
    t_out = t_out[:-2]

    if last != 0:
        return mu_hbfp[:, -1], sigma2_hbfp[:, :, -1], p_tilde[-1], v[-1], t_out[-1]
    else:
        return mu_hbfp, sigma2_hbfp, p_tilde, v, t_out
