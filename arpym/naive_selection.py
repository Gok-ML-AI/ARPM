# -*- coding: utf-8 -*-
import numpy as np


def naive_selection(optim, n_bar, k_bar=None):
    """This function performs a naive selection of elements from a given set
    based on optimization results, where it optimizes over 1-element selections, 
    sorts them, then iteratively builds and optimizes over k-element selections.

    Parameters
    ----------
        optim : function
        n_bar : int
        k_bar : int

    Returns
    -------
        x_naive : list
        f_x_naive : array, shape(k_bar,)
        s_k_naive_list : list
    """

    if k_bar:
        k_bar = min(k_bar, n_bar)
    else:
        k_bar = n_bar

    # step 1: optimize over 1-element selections
    f_x1 = np.zeros(n_bar)
    for n in range(1, n_bar + 1):
        all = optim(np.array([n]))
        f_x1[n - 1] = all[1]

    # step 2: naive sorting
    n_sort = np.argsort(f_x1)

    # step 3: initialize selection
    s_k_naive = []
    s_k_naive_list = []

    x_naive = []
    f_x_naive = np.zeros(k_bar)
    for k in range(k_bar):
        # step 4: build naive selection (set)
        s_k_naive = n_sort[:k + 1] + 1
        s_k_naive_list.append(s_k_naive)

        # step 5: optimize over k-element selections
        all = optim(s_k_naive)
        x_naive.append(all[0]),
        f_x_naive[k] = all[1]

    return x_naive, f_x_naive, s_k_naive_list
