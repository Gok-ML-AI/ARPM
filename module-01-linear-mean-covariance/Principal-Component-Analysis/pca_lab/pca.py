"""
Reusable PCA toolbox for ARPM Unit 3 (Principal Component Analysis)

Each variable comment gives the ARPM symbol and the short name ARPM's own notebooks use,
so I can cross-reference the theory pages and lab code.
"""
from numpy import argsort, argmax, abs, diag, cumsum
from numpy.linalg import eigh

"""
Principal directions of a covariance matrix, most important first.

Input - covariance matrix (n_variables, n_variables) array
σ² (ARPM: s2) - e.g. the covariance of n_stocks returns.

Returns - 
variances along directions : (n_variables,) array
λ₁ ≥ λ₂ ≥.... (ARPM: lam) - eigenvalues: how much variance lies along each principal direction.
principal directions : (n_variables, n_variables) array
e₁, e₂, ... (ARPM: e) - eigenvectors, 1 per column.
Column K is the direction paired with variances_along_directions[k]
"""
def sorted_eig(covariance_matrix):
    # eigh is numpy's eigen-solves for symmetric matrices (covariances always are).
    # It returns eigenvalues in ASCENDING  order (smallest first).
    variances_along_directions, principal_directions = eigh(covariance_matrix)

    # Positions that put the eigenvalues in DESCENDING order.
    # Negating flips the order, so argsort's smallest-first becomes largest first
    largest_first_order = argsort(-variances_along_directions)

    # Reorder the eigenvalues using those positions.
    variances_along_directions = variances_along_directions[largest_first_order]

    # Reorder the eigenvector COLUMNS identically, so each direction stays
    # paired with its own eigenvalue.
    principal_directions = principal_directions[:, largest_first_order]

    # Sign convention
    # A direction and its -ve describe the same line, and eigh may return either one.
    # to make results repeatable, flip any column whose largest-magnitude entry is -ve.
    # Columns are flipped WHOLE.  mixed signs inside a column are kept (they carry meaning)

    # For each column, the row holding its largest entry by size
    # (abs is only used to measure size; nothing is changed here).
    row_of_largest_entry = argmax(abs(principal_directions), axis=0)

    # True for each column whose largest entry is -ve.
    largest_entry_per_column = diag(principal_directions[row_of_largest_entry, :]) < 0
    columns_to_flip = largest_entry_per_column < 0

    # Negate only those columns.
    principal_directions[:, columns_to_flip] = -principal_directions[:, columns_to_flip]
    return variances_along_directions, principal_directions

"""
Share of total variance explained by the first 1,2,... components.

Input - variances_along_directions : λ (ARPM: lam) sorted largest first.

Returns - cumulative_share_explained : (n_variables, ) array
r²(k) = (λ₁ + ... + λₖ) / (λ₁ + ... + λₙ)   (ARPM: r2)
The last entry is always 1: all components rebuild everything
"""
def r_squared(variances_along_directions):
    cumulative_share_explained = cumsum(variances_along_directions) / variances_along_directions.sum()
    return cumulative_share_explained

