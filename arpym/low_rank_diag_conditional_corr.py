#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
from numpy.linalg import matrix_rank, solve, eig, svd


def low_rank_diag_conditional_corr(c2, d, k_bar):
    """This function computes the low rank diagonal conditional correlation

    Parameters
    ----------
        c2 : array, shape (n_bar, n_bar) 
        d : array, shape (m_bar, n_bar)
        k_bar : scalar

    Returns
    -------
        c2_lrd : array, shape (n_bar, n_bar)
        beta : array, shape (n_bar, k_bar)
        n_iter : scalar
        constraint : scalar, boolean indicator, 1 in case the constraint is satisfied, i.e. d@beta = 0
    """

    conditional = 1
    if np.sum(abs(d.flatten())) == 0:
        conditional = 0

    n_bar = c2.shape[0]

    if k_bar > n_bar - matrix_rank(d):
        raise Warning('k_bar has to be <= rho.shape[0]-rank[d]')

    eps1 = 1e-9
    eta = 0.01
    gamma = 0.1
    constraint = 0
    c2_lrd = c2
    dist = np.zeros
    n_iter = 0

    diag_lam2, e = eig(c2)
    lam2 = diag_lam2
    lam2_ord, order = np.sort(lam2)[::-1], np.argsort(lam2)[::-1]
    lam = np.real(np.sqrt(lam2_ord[:k_bar]))
    e_ord = e[:, order]
    beta = np.real(e_ord[:n_bar,:k_bar]@np.diagflat(np.maximum(lam, eps1)))
    c = c2

    for j in range(1000):
        # conditional PC
        a = c - np.eye(n_bar) + np.diagflat(np.diag(beta@beta.T))
        if conditional == 1:
            lam2, e = conditional_pc(a, d)
            lam2 = lam2[:k_bar]
            e = e[:,:k_bar]
            lam = np.sqrt(lam2)
        else:
            # if there aren't constraints: standard PC using covariance matrix
            diag_lam2, e1 = eig(a)
            lam2 = diag_lam2
            lam2_ord, order = np.sort(lam2)[::-1], np.argsort(lam2)[::-1]
            e_ord = e1[:, order]
            e = e_ord[:,:k_bar]
            lam = np.sqrt(lam2_ord[:k_bar])

        # loadings
        beta_new = e@np.diagflat(np.maximum(lam, eps1))
        # rows length
        l_n = np.sqrt(np.sum(beta_new**2, 1))
        # rows scaling
        beta_new[l_n > 1,:] = beta_new[l_n > 1, :]/np.tile(l_n[l_n > 1, np.newaxis]*(1 + gamma), (1, k_bar))
        # reconstruction
        c = beta_new@beta_new.T + np.eye(n_bar, n_bar) - np.diag(np.diag(beta_new@beta_new.T))
        # convergence check
        distance = 1/n_bar*np.sum(np.sqrt(np.sum((beta_new - beta)**2, 1)))
        if distance <= eta:
            c2_lrd = c
            dist = distance
            n_iter = j
            beta = beta_new.copy()
            if d.shape == (1, 1):
                tol = np.max(abs(d*beta))
            else:
                tol = np.max(abs(d.dot(beta)))
            if tol < 1e-9:
                constraint = 1
                break
        else:
            beta = beta_new.copy()
            beta = np.real(beta)
            c2_lrd = np.real(c2_lrd)
            c2_lrd = (c2_lrd + c2_lrd.T)/2
    return c2_lrd, beta, dist, n_iter, constraint


def conditional_pc(sigma2, d):
    """This function computes the conditional principal directions/variances of a symmetric positive definite matrix

    Parameters
    ----------
        sigma2 : array, shape (n_bar, n_bar) 
        d : array, shape (k_bar, n_bar)

    Returns
    -------
        lam2_d : vector, shape (n_bar, 1)
        e_d : array, shape (n_bar, n_bar)
    """

    n_bar = sigma2.shape[0]
    m_bar = n_bar - matrix_rank(d)
    lam2_d = np.empty((n_bar, 1))
    e_d = np.empty((n_bar, n_bar))

    # constraints initialization
    a_n = d
    for n in range(n_bar):
        # orthogonal projection matrix
        p = np.eye(n_bar) - a_n.T.dot(solve(a_n@a_n.T, a_n))
        # conditional dispersion matrix
        s2 = p@sigma2@p
        # conditional principal directions/variances
        w, v = eig(s2)
        ind = np.argsort(-w)
        eigvec, eigval = v[:, ind], w[ind]
        _,_,eigvec = svd(s2)
        eigvec = eigvec.T
        p, d = eigvec.shape
        eigvec = eigvec*np.sign(eigvec[np.array(np.argmax(abs(eigvec), axis=0)), np.arange(0, d)])[np.newaxis,...]
        e_d[:, n] = eigvec[:, 0]
        lam2_d[n] = eigval[0]

    return lam2_d, e_d
