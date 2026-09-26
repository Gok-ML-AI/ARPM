#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import arange, array, concatenate, cov, degrees, linspace, \
                  log, max, mean, percentile, round, sqrt
from numpy.linalg import eig
from scipy.stats import expon, norm as normal, pearsonr, poisson
from math import atan
from statsmodels.api import graphics
from seaborn import barplot, despine, histplot, JointGrid, jointplot, lineplot, scatterplot
from matplotlib.patches import Ellipse
from matplotlib.pyplot import gca, show

def invariance_test_ellipsoid(eps, l_bar, *, conf_lev=0.95, fit=0, r=2, title='Invariance test',
                              bl=None, bu=None, plot_test=True):
    """This function performs an invariance test on a given set of data by computing sample 
    autocorrelations, confidence intervals, and plotting an ellipsoid in a scatter plot, 
    with options for adjusting the plot and fitting distributions.

    Parameters
    ----------
        eps : array, shape(t_bar)
        l_bar : scalar
        conf_lev : scalar, optional
        fit : scalar, optional
        r : scalar, optional
        title : string, optional
        bl : scalar, optional
        bu : scalar, optional
        plot_test : boolean, optional

    Returns
    -------
        rho : array, shape(l_bar)
        conf_int : array, shape(2)
    """

    if len(eps.shape) == 2:
        eps = eps.reshape(-1)
    if bl is None:
        bl = percentile(eps, 0.25)
    if bu is None:
        bu = percentile(eps, 99.75)

    nb = int(round(10*log(eps.shape)))  # number of bins for histograms

    # compute sample autocorrelations
    rho = array([pearsonr(eps[:-k] - mean(eps[:-k], axis=0),
                eps[k:] - mean(eps[k:], axis=0))[0] for k in range(1, l_bar + 1)])

    # compute confidence interval
    alpha = 1 - conf_lev
    z_alpha_half = normal.ppf(1 - alpha/2)/sqrt(eps.shape[0])
    conf_int = array([-z_alpha_half, z_alpha_half])
    eps_original = eps.copy()
    
    # plot ellipse, if requested
    if plot_test:
        # ellipsoid test: location-dispersion parameters
        x = eps[:-l_bar]
        eps = eps[l_bar:]
        z = concatenate((x.reshape((-1, 1)), eps.reshape((-1, 1))), axis=1)

        # compute sample mean, sample covariance and generate figure
        mu_hat = mean(z, axis=0)
        sigma2_hat = cov(z.T)

        # scatter plot with ellipse
        ax = JointGrid(x=x, y=eps, xlim=(bl, bu), ylim=(bl, bu), marginal_ticks=True)
        ax.plot_joint(scatterplot, marker='.', s=10)
        lam, e = eig(sigma2_hat)
        center = mu_hat; width = 2*r*sqrt(lam[0]); height = 2*r*sqrt(lam[1]); angle = degrees(atan(e[1, 0]/e[1, 1]))
        ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, fill=False, color='#f56502', lw=2)
        ax.ax_joint.add_patch(ellipse)
        ax.set_axis_labels('obs', 'lagged obs.')
        ax.ax_joint.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax.figure.suptitle(title, y=1.02)
        
        # marginal distributions
        if fit == 1: # normal     
            histplot(x=x, ax=ax.ax_marg_x, bins=nb, stat='density', label='Marginal distr')
            xmin, xmax = ax.ax_marg_x.get_xlim()
            x_range = linspace(xmin, xmax, 100)
            p = normal.pdf(x_range, *normal.fit(x))
            lineplot(x=x_range, y=p, ax=ax.ax_marg_x, label='Normal fit')
        
            histplot(y=eps, ax=ax.ax_marg_y, bins=nb, stat='density')  
            ymin, ymax = ax.ax_marg_y.get_ylim() 
            y_range = linspace(ymin, ymax, 100)
            p_y = normal.pdf(y_range, *normal.fit(eps))
            lineplot(x=p_y, y=y_range, estimator=None, orient='y', ax=ax.ax_marg_y)
        
        elif fit == 2 and sum(x < 0) == 0: # exponential
            histplot(x=x, ax=ax.ax_marg_x, bins=nb, stat='density', label='Marginal distr')
            xmin, xmax = ax.ax_marg_x.get_xlim()
            x_range = linspace(xmin, xmax, 100)
            p = expon.pdf(x_range, *expon.fit(x))
            lineplot(x=x_range, y=p, ax=ax.ax_marg_x, label='Exponential fit')
        
            histplot(y=eps, ax=ax.ax_marg_y, bins=nb, stat='density')
            ymin, ymax = ax.ax_marg_y.get_ylim()
            y_range = linspace(ymin, ymax, 100)
            p = expon.pdf(y_range, *expon.fit(eps))
            lineplot(x=p, y=y_range, estimator=None, ax=ax.ax_marg_y)
         
        elif fit == 3 and sum(x < 0) == 0: # poisson
            histplot(x=x, ax=ax.ax_marg_x, bins=nb, stat='density', label='Marginal distr')
            mlest = mean(x) 
            k = arange(int(max(x)) + 1)  
            lineplot(x=k, y=poisson.pmf(k, mlest),  ax=ax.ax_marg_x, label='Poisson fit')
            
            histplot(y=eps, ax=ax.ax_marg_y, bins=nb, stat='density')
            mlest = mean(eps) 
            k = arange(int(max(eps)) + 1)  
            lineplot(x=poisson.pmf(k, mlest), y=k, estimator=None, orient='y', ax=ax.ax_marg_y)
        
        else:
            ax.plot_marginals(histplot, kde=True, label='KDE') 
        show()
        
        # autocorrelation plot
        graphics.tsa.plot_acf(eps_original, lags=l_bar, zero=False, alpha=1-conf_lev)
        gca().set(xlabel='Lag', ylabel='Autocorrelation', ylim=(-1, 1), xlim=(0.5, l_bar+0.5))
        despine(); show()
        
    return rho, conf_int

