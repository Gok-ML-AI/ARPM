#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import arange, delete, log, max, ones, percentile, round, sort, sqrt
from numpy.random import randint, permutation
from scipy.stats import ks_2samp
from seaborn import despine, histplot, scatterplot
from matplotlib.pyplot import plot, show, subplot2grid, suptitle, tight_layout

def invariance_test_ks(eps, *, conf_lev=0.95, title='Kolmogorov-Smirnov test', plot_test=True):
    """This function conducts a Kolmogorov-Smirnov test on a given dataset by computing 
    the sample autocorrelations, confidence intervals, and plotting the results in a histogram, 
    with options for adjusting the plot and fitting distributions.

    Parameters
    ----------
        eps : array, shape (t_bar, )
        conf_lev : scalar, optional
        title : string, optional
        plot_test : boolean, optional

    Returns
    -------
        z_ks : float
            KS statistic
        z : float
            Confidence interval
    """

    # generate two random mutually exclusive partitions of observations
    t_bar = eps.shape[0]
    half_t_bar = t_bar // 2
    # if t_bar is odd, remove one element at random to ensure dimension match on partitions
    if t_bar % 2 != 0:
        eps = delete(eps, randint(t_bar))
    # random permutation
    eps_perm = permutation(eps)
    # create two partitions
    eps_a = eps_perm[:half_t_bar]
    eps_b = eps_perm[half_t_bar:]
    # compute hfp cdfs of two partitions
    cdf_a = arange(len(eps_a))/len(eps_a)
    cdf_b = arange(len(eps_b))/len(eps_b)
    # compute KS statistic
    z_ks, _ = ks_2samp(eps_a, eps_b)
    # compute confidence interval
    alpha = 1 - conf_lev
    z = sqrt(-log(alpha/2)*(len(eps_a) + len(eps_b))/(2*len(eps_a)*len(eps_b)))

    # generate figure
    if plot_test:
        # build band for Kolmogorov-Smirnov test
        band_mid = 0.5*(cdf_a + cdf_b)
        band_up = band_mid + 0.5*z 
        band_low = band_mid - 0.5*z

        ax1 = subplot2grid((2, 2), (0, 0))
        ax2 = subplot2grid((2, 2), (0, 1))
        ax3 = subplot2grid((2, 2), (1, 0), colspan=2)
        
        # plot histogram of sample 1
        histplot(x=eps_a, bins=int(round(10*log(len(eps_a.flatten())))), stat='density', color='orange', edgecolor='k', ax=ax1).set(xlabel='Sample 1')
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))

        # plot histogram of sample 2
        histplot(x=eps_b, bins=int(round(10*log(len(eps_a.flatten())))), stat='density', color='steelblue', edgecolor='k', ax=ax2).set(xlabel='Sample 2')
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        
        # max and min value of first reference axis settings
        xlim_1 = percentile(eps, 1.5)
        xlim_2 = percentile(eps, 98.5)
        ax1.set_xlim(xlim_1, xlim_2)
        ax2.set_xlim(xlim_1, xlim_2)
      
        ylim = max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
        ax1.set_ylim([0, ylim])
        ax2.set_ylim([0, ylim])
        
        # plot CDFs
        sorted_eps_a = sort(eps_a)
        sorted_eps_b = sort(eps_b)
        
        scatterplot(x=sorted_eps_a, y=cdf_a, color='#E64D00', s=5, ax=ax3).set(xlabel='Data', ylabel='Cdf')
        scatterplot(x=sorted_eps_b, y=cdf_b, color='darkblue', s=5, ax=ax3)
        
        scatterplot(x=eps_a, y=0.002*ones(len(eps_a)), color='#E64D00', s=2, ax=ax3)
        scatterplot(x=eps_b, y=0.002*ones(len(eps_b)), color='darkblue', s=2, ax=ax3)
        
        plot(sorted_eps_a, band_up, '-', color='k', lw=0.5)
        plot(sorted_eps_b, band_low, '-', color='k', lw=0.5)
        
        ax3.set_xlim([xlim_1, xlim_2])
        ax3.set_ylim([-0.05, 1.05])
        ax3.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))

        suptitle(title)
        tight_layout(rect=[0, 0.03, 1, 0.95])
        despine(); show()

    return z_ks, z
