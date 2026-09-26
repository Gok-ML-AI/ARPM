#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import argsort, atleast_1d, column_stack, floor, log, max, \
                  meshgrid, min, ones, sort, spacing, sqrt, sum, zeros
from bisect import bisect_right
from pandas import Series
from matplotlib.pyplot import axes, figure, show, xlabel, ylabel
from matplotlib.ticker import FormatStrFormatter
from seaborn import barplot, despine
from arpym.schweizer_wolff import schweizer_wolff

def invariance_test_copula(eps, lag_bar, k_bar=None, title='Copula test'):
    """
       This function assesses copula invariance by conducting a Schweizer-Wolff dependence test and plots the results,
       including a 3D histogram of bivariate data and a bar plot illustrating dependence over various lags.

    Parameters
    ----------
        eps : array, shape (t_bar,)
        lag_bar : scalar
        k_bar: int

    Returns
    -------
        sw: array, shape(lag_bar,)

    """

    t_bar = eps.shape[0]

    # Schweizer-Wolff dependence for lags
    sw = zeros(lag_bar)
    for l in range(lag_bar):
        sw[l] = schweizer_wolff(column_stack((eps[(l + 1):], eps[: -(l + 1)])))

    # grades scenarios
    x_lag = eps[:-lag_bar]
    y_lag = eps[lag_bar:]
    p = ones(column_stack((x_lag, y_lag)).shape[0])/column_stack((x_lag, y_lag)).shape[0]  # equal probabilities
    x_grid, ind_sort = sort(column_stack((x_lag, y_lag)), axis=0), argsort(column_stack((x_lag, y_lag)), axis=0)  # sorted scenarios
    # marginal CDFs
    cdf_x = zeros(column_stack((x_lag, y_lag)).shape)
    for n in range(column_stack((x_lag, y_lag)).shape[1]):
        x_bar = x_grid[:, n]
        x_bar = atleast_1d(x_bar)
        x = column_stack((x_lag, y_lag))[:, n]

        # sorted scenarios-probabilities
        sort_x = argsort(x)
        x_sort = Series(x).iloc[sort_x]
        p_sort = p[sort_x]

        # cumulative sums of sorted probabilities
        u_sort = zeros(x.shape[0] + 1)
        for j in range(1, x.shape[0] + 1):
            u_sort[j] = sum(p_sort[:j])

        # output cdf
        cindx = [0]*x_bar.shape[0]
        for k in range(x_bar.shape[0]):
            cindx[k] = bisect_right(x_sort, x_bar[k])
        cdf_x[:, n] = u_sort[cindx]
        
    # copula scenarios
    u = zeros(column_stack((x_lag, y_lag)).shape)
    for n in range(column_stack((x_lag, y_lag)).shape[1]):
        u[ind_sort[:, n], n] = cdf_x[:, n]
    u[u >= 1] = 1 - spacing(1)
    u[u <= 0] = spacing(1)  # clear spurious outputs
    
    # normalized histogram
    if k_bar is None:
        k_bar = floor(sqrt(7*log(t_bar)))
    k_bar = int(k_bar)
    
    p = ones(u.shape[0])/u.shape[0]  # uniform probabilities
    min_x_1 = min(u[:, 0])
    min_x_2 = min(u[:, 1])

    # bin width
    h_1 = (max(u[:, 0]) - min_x_1)/k_bar
    h_2 = (max(u[:, 1]) - min_x_2)/k_bar

    # bin centroids
    xi_1 = zeros(k_bar)
    xi_2 = zeros(k_bar)
    for k in range(k_bar):
        xi_1[k] = min_x_1 + (k + 1 - 0.5)*h_1
        xi_2[k] = min_x_2 + (k + 1 - 0.5)*h_2

    # normalized histogram heights
    f = zeros((k_bar, k_bar))
    for k_1 in range(k_bar):
            for k_2 in range(k_bar):
                # take edge cases into account
                if k_1 > 0 and k_2 > 0:
                    ind = ((u[:, 0] > xi_1[k_1] - h_1/2)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] > xi_2[k_2] - h_2/2)&(u[:, 1] <= xi_2[k_2] + h_2/2))
                elif k_1 > 0 and k_2 == 0:
                    ind = ((u[:, 0] > xi_1[k_1] - h_1/2)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] >= min_x_2)&(u[:, 1] <= xi_2[k_2] + h_2/2))
                elif k_1 == 0 and k_2 > 0:
                    ind = ((u[:, 0] >= min_x_1)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] > xi_2[k_2] - h_2/2) & (u[:, 1] <= xi_2[k_2] + h_2/2))
                else:
                    ind = ((u[:, 0] >= min_x_1)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] >= min_x_2)&(u[:, 1] <= xi_2[k_2] + h_2/2))

                f[k_1, k_2] = sum(p[ind])/(h_1*h_2)

    ############################################################## plots ##############################################################
    # 2D histogram
    ax = axes(projection='3d')
    xpos, ypos = meshgrid(xi_1 - (xi_1[1] - xi_1[0])/2, xi_2 - (xi_2[1] - xi_2[0])/2)  # adjust bin centers to left edges
    ax.bar3d(xpos.flatten('F'), ypos.flatten('F'), zeros(xpos.flatten('F').shape), xi_1[1] - xi_1[0], xi_2[1] - xi_2[0], f.flatten())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')); ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f')); ax.invert_xaxis()
    ax.set(xlabel='Grade obs.', ylabel='Grade lagged obs.', title=title)
    show()

    # dependence plot
    x_vals = range(1, lag_bar + 1)
    barplot(x=x_vals, y=sw, width=0.5, facecolor='#969696', edgecolor='#212529')
    barplot(x=list(x_vals)[lag_bar - 1], y=[sw[lag_bar - 1]], width=0.5, facecolor='#f56502', edgecolor='#212529').set(
            ylim=[0, 1], xlabel='Lag', label='Dependence');
    despine(); show()

    return sw
