import numpy as np


def kalman_filter(x_t, alpha_hat, beta_hat, s2_hat, alpha_hat_h, beta_hat_h, s2_hat_h, h_t=None, p2=None):
    """This function implements the Kalman filter algorithm for state estimation in a time-series model with hidden factors

    Parameters
    ----------
        x_t : array, shape (t_bar, n_bar)
        alpha_hat : array, shape (n_bar,)
        beta_hat : array, shape (n_bar, k_bar)
        s2_hat : array, shape(t_bar, t_)
        alpha_hat_h : array, shape(k_bar,)
        beta_hat_h : array, shape(k_bar, k_bar)
        s2_hat_h : array, shape(k_bar, k_bar)
        h_t : array, shape (k_bar,), optional
        p2 : array, shape (k_bar, k_bar), optional

    Returns
    -------
        h_t : array, shape(t_bar, k_bar) if k_bar>1 or (t_bar, ) for k_bar=1

    """

    if len(x_t.shape) == 1:
        x_t = x_t.reshape(-1, 1).copy()
    t_bar, n_bar = x_t.shape
    k_bar = len(alpha_hat_h)

    # set initial value h_t[:,0] for hidden factors
    if h_t is None:
        h_t = x_t[[0], :]@beta_hat
    else:
        h_t = h_t.reshape((1, k_bar))

    # set initial covariance of h_t[:,0], expressing uncertainty of measurement
    if p2 is None:
        p2 = 10*np.eye(k_bar).reshape(k_bar, k_bar, 1)
    else:
        p2 = p2.reshape(k_bar, k_bar, 1)

    for t in range(1, t_bar):
        # estimate step
        h_t_tilde = alpha_hat_h + beta_hat_h@h_t[t - 1, :]
        p2_tilde = beta_hat_h@p2[:, :, t - 1]@beta_hat_h.T + s2_hat_h
        # correction step
        u = x_t[[t], :] - alpha_hat - beta_hat@h_t_tilde
        s2_hat_u = s2_hat + beta_hat@p2_tilde@beta_hat.T
        kappa = p2_tilde@beta_hat.T.dot(np.linalg.pinv(s2_hat_u))
        h_t = np.r_[h_t, h_t_tilde + u@kappa.T]
        p2 = np.r_['-1', p2, ((np.eye(k_bar) - kappa@beta_hat)@p2_tilde)[..., np.newaxis]]

    return np.squeeze(h_t)
