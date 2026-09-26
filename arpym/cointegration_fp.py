import numpy as np
from scipy import linalg
from statsmodels.stats.weightstats import DescrStatsW
from arpym.fit_lfm_mlfp import fit_lfm_mlfp


def cointegration_fp(x, p=None, *, b_threshold=0.99):
    
    """This function estimates cointegration vectors and their coefficients from a given dataset. 

    Parameters
    ----------
         x : array, shape(t_bar, d_bar)
         p : array, shape(t_bar, d_bar)
         b_threshold : scalar

    Returns
    -------
        c_hat : array, shape(d_bar, l_bar)
        beta_hat : array, shape(l_bar, )

    """

    t_bar = x.shape[0]
    if len(x.shape) == 1:
        x = x.reshape((t_bar, 1))
        d_bar = 1
    else:
        d_bar = x.shape[1]

    if p is None:
        p = np.ones(t_bar)/t_bar  # uniform probabilities

    # estimate HFP covariance matrix
    sigma2_hat = DescrStatsW(x, weights=p).cov 

    # find eigenvectors
    e_hat = linalg.eigh(sigma2_hat, subset_by_index=(0, sigma2_hat.shape[0] - 1))[1]
    e_hat = e_hat[:, ::-1]
    # enforce a sign convention on the coefficients
    # the largest element in each eigenvector will have a positive sign
    ind = np.argmax(abs(e_hat), axis=0)
    ind = np.diag(e_hat[ind, :]) < 0
    e_hat[:, ind] = -e_hat[:, ind]

    # detect cointegration vectors
    c_hat = []
    b_hat = []
    p = p[:-1]
    for d in np.arange(0, d_bar):
        # define series
        y_t = e_hat[:, d]@x.T

        # fit AR(1)
        yt = y_t[1:].reshape((-1, 1))
        ytm1 = y_t[:-1].reshape((-1, 1))
        _, b, _, _ = fit_lfm_mlfp(yt, ytm1, p/np.sum(p))
        if np.ndim(b) < 2:
            b = np.array(b).reshape(-1, 1)

        # check stationarity
        if abs(b[0, 0]) <= b_threshold:
            c_hat.append(list(e_hat[:, d]))
            b_hat.append(b[0, 0])

    # output
    c_hat = np.array(c_hat).T
    b_hat = np.array(b_hat)

    # sort according to the AR(1) parameters beta_hat
    c_hat = c_hat[:, np.argsort(b_hat)]
    b_hat = np.sort(b_hat)

    return c_hat, b_hat
