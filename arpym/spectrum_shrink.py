import numpy as np
from scipy import interpolate
from scipy import linalg
from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution as MP


def spectrum_shrink(sigma2_in, t_bar):
    """This function performs spectrum shrinkage on a covariance matrix, determining the optimal number
       of principal components to retain based on the Marcenko-Pastur distribution and minimizing the 
       mean squared error between the empirical and theoretical eigenvalue distributions

    Parameters
    ----------
         sigma_in : array, shape (i_bar, i_bar)
         t_bar : scalar

    Returns
    -------
         sigma_out : array, shape (i_bar, i_bar)
         lam_out : array, shape (i_bar,)
         k_bar : scalar
         err : scalar
         y_mp : array, shape (100,)
         x_mp : array, shape (100,)
         dist : array
    """

    i_bar = sigma2_in.shape[0]

    # PCA decomposition    
    n_bar = sigma2_in.shape[0]
    k_bar = n_bar
    lam, e = linalg.eigh(sigma2_in, subset_by_index=(n_bar - k_bar, n_bar-1))
    lam = lam[::-1]
    e = e[:, ::-1]

    # enforce a sign convention on the coefficients and largest
    # element in each eigenvector will have a positive sign
    ind = np.argmax(abs(e), axis=0)
    ind = np.diag(e[ind, :]) < 0
    e[:, ind] = -e[:, ind]

    # determine optimal k_bar
    ll = 1000

    dist = np.ones(i_bar - 1)*np.nan
    for k in range(i_bar - 1):
        lam_k = lam[k + 1:]
        lam_noise = np.mean(lam_k)
        q = t_bar/len(lam_k)
        # compute M-P on a very dense grid
        norm_mp = MP(ratio=1/q, sigma=np.sqrt(lam_noise))
        x_tmp = np.linspace(norm_mp.lambda_minus, norm_mp.lambda_plus, 1000)
        mp_tmp = norm_mp._pdf(x_tmp)
        if q > 1:
            x_tmp = np.r_[0, x_tmp[0], x_tmp]
            mp_tmp = np.r_[0, mp_tmp[0], mp_tmp]
        l_max = np.max(lam_k)
        if l_max > x_tmp[-1]:
            x_tmp = np.r_[x_tmp, x_tmp[-1], l_max]
            mp_tmp = np.r_[mp_tmp, 0, 0]
        # compute histogram of eigenvalues
        hgram, x_bin_edge = np.histogram(lam_k, bins='auto', density=True)
        bin_size = np.diff(x_bin_edge)[0]
        x_bin = x_bin_edge[:-1] + bin_size/2
        # interpolation
        interp = interpolate.interp1d(x_tmp, mp_tmp, fill_value='extrapolate')
        mp = interp(x_bin)
        dist[k] = np.mean((mp - hgram)**2)

    err_tmp, k_tmp = np.nanmin(dist), np.nanargmin(dist)
    k_bar = k_tmp
    err = err_tmp

    # isotropy
    lam_out = lam
    lam_noise = np.mean(lam[k_bar + 1:])

    # shrunk spectrum
    lam_out[k_bar + 1:] = lam_noise

    # output
    sigma2_out = e@np.diagflat(lam_out)@e.T

    # compute M-P on a very dense grid
    norm_mp = MP(ratio=(i_bar - k_bar - 1)/t_bar, sigma=np.sqrt(lam_noise))
    x_mp = np.linspace(norm_mp.lambda_minus, norm_mp.lambda_plus, 100)
    y_mp = norm_mp._pdf(x_mp)

    return sigma2_out, lam_out, k_bar, err, y_mp, x_mp, dist
