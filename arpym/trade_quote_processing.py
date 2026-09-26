import numpy as np


def trade_quote_processing(t, dates_quotes, q_ask, p_ask, q_bid, p_bid, t_n, dates_trades,
                           p_last, delta_q, delta_sgn, match, varargin=None):
    """This function organizes and processes trade and quote data, handling duplicates 
    and adjusting signs of transactions while accommodating various optional inputs 
    such as trade volumes and additional variables, returning the processed data.

    Parameters
    ----------
        t : array, shape (k1_bar,)
        dates_quotes : array, shape (k1_bar,)
        q_ask : array, shape (k1_bar,)
        p_ask : array, shape (k1_bar,)
        q_bid : array, shape (k1_bar,)
        p_bid : array, shape (k1_bar,)
        t_n : array, shape (k3_bar,)
        dates_trades : array, shape (k3_bar,)
        p_last : array, shape (k3_bar,)
        delta_q : array, shape (k3_bar,)
        delta_sgn : array, shape (k3_bar,)
        match : array, shape (k3_bar,)
        varargin{1} : array, shape (k1_bar,)
        varargin{2} : array, shape (k1_bar,)

    Returns
    -------
        t_p : array, shape (k1_bar,)
        dates_quotes_p : array, shape (k1_bar,)
        q_ask_p : array, shape (k1_bar,)
        p_ask_p : array, shape (k1_bar,)
        q_bid_p : array, shape (k1_bar,)
        p_bid_p : array, shape (k1_bar,)
        t_n_p : array, shape (k2_bar,)
        dates_trades_p : array, shape (k2_bar,)
        p_last_p : array, shape (k2_bar,)
        delta_q_p : array, shape (k2_bar,)
        delta_sgn_p : array, shape (k2_bar,)
        varargout{1} : array, shape (k1_bar,)
        varargout{2} : array, shape (k1_bar,)
    """

    # quotes: if there are repeated times, consider last stored observation and delete others
    t_unique = np.unique(t)
    k1_bar = len(t_unique)
    dates_quotes_p = {}
    t_p = np.zeros(k1_bar)
    p_bid_p = np.zeros(k1_bar)
    p_ask_p = np.zeros(k1_bar)
    q_bid_p = np.zeros(k1_bar)
    q_ask_p = np.zeros(k1_bar)
    varargout = {1: np.zeros(k1_bar), 2: np.zeros(k1_bar)}
    for k in range(k1_bar):
        index = np.where(t == t_unique[k])[0]
        dates_quotes_p[k] = dates_quotes[index[-1]]
        t_p[k] = t[index[-1]]
        p_bid_p[k] = p_bid[index[-1]]
        p_ask_p[k] = p_ask[index[-1]]
        q_bid_p[k] = q_bid[index[-1]]
        q_ask_p[k] = q_ask[index[-1]]
        if varargin is not None:
            varargout[2][k] = varargin[2][index[-1]]
            varargout[1][k] = varargin[1][index[-1]]

    # trades: set sign of transaction in delta_sgn where it is not defined: 
    # -if traded price is closer to best ask the sign is "BUY", i.e. "+1" 
    # -if traded price is closer to best bid the sign is "SELL", i.e. "-1"
    index = np.where(np.isnan(delta_sgn))[0]
    for i in range(len(index)):
        i_min = np.argmin(abs(t - t_n[index[i]]))
        if abs(p_last[index[i]] - p_ask[i_min]) < abs(p_last[index[i]] - p_bid[i_min]):
            delta_sgn[index[i]] = +1
        else:
            delta_sgn[index[i]] = -1

    # trades: concatenate "match" events wheres indices of elements NOT equal to NAN in vector match
    index = np.where(~np.isnan(match))[0]
    dates_trades_tmp = dates_trades[index]
    t_n_tmp = t_n[index]
    p_last_tmp = p_last[index]
    dv_tmp = np.zeros(len(index))
    dv_tmp[0] = np.sum(delta_q[:index[0] + 1])
    for k in range(1, len(index)):
        dv_tmp[k] = np.sum(delta_q[index[k - 1] + 1:index[k] + 1])
    dzeta_tmp = delta_sgn[index]

    # trades: if there are repeated times, consider cumulative volume, last price and last sign
    t_n_unique = np.unique(t_n_tmp)
    k2_bar = len(t_n_unique)
    dates_trades_p = {}
    t_n_p = np.zeros(k2_bar)
    p_last_p = np.zeros(k2_bar)
    delta_q_p = np.zeros(k2_bar)
    delta_sgn_p = np.zeros(k2_bar)
    for k in range(k2_bar):
        index = np.where(t_n_tmp == t_n_unique[k])[0]
        dates_trades_p[k] = dates_trades_tmp[index[-1]]
        t_n_p[k] = t_n_tmp[index[-1]]
        p_last_p[k] = p_last_tmp[index[-1]]
        delta_q_p[k] = np.sum(dv_tmp[index])
        delta_sgn_p[k] = dzeta_tmp[index[-1]]

    return t_p, dates_quotes_p, q_ask_p, p_ask_p, q_bid_p, p_bid_p, t_n_p, dates_trades_p, p_last_p, delta_q_p, delta_sgn_p, varargout
