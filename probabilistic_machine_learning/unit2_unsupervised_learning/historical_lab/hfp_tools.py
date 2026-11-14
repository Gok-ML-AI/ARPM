"""
hfp_tools.py
Shared helpers for ARPM Probabilistic ML - Unit 2 (historical / HFP estimation).

Every function maps to one ARPM equation, so the code reads like the book:
    exp_decay_probs   -> (17.6)   time exponential-decay flexible probabilities
    ens               -> (17.21)  effective number of scenarios
    hfp_cdf           -> (17.29)  HFP cdf  (equal weights = historical cdf, 17.24)
    hfp_mean / hfp_sd -> (26.a.1) HFP mean and standard deviation
    hfp_quantile      -> (17.44)  smallest q whose cumulative probability >= c
    hfp_es            -> plug-in (17.35): expected shortfall of the HFP distribution
    ks_2samp_hfp      -> (17.26)  sup distance between two HFP cdfs (two-sample KS test)
"""
import numpy as np


def exp_decay_probs(t_bar, tau_hl, t_star=None):
    """p_t proportional to exp(-ln2/tau_hl * |t_star - t|), t = 1..t_bar  (17.6).

    The weight halves every tau_hl steps away from the target time t_star
    (default: the last observation, t_star = t_bar)."""
    t_star = t_bar if t_star is None else t_star
    t = np.arange(1, t_bar + 1)
    p = np.exp(-np.log(2) / tau_hl * np.abs(t_star - t))
    return p / p.sum()


def ens(p):
    """Effective number of scenarios = exp(entropy) = exp(-sum p ln p)  (17.21).

    Uniform weights over t_bar scenarios give ens = t_bar (the maximum);
    all weight on one scenario gives ens = 1."""
    p = np.asarray(p)
    p = p[p > 0]
    return float(np.exp(-p @ np.log(p)))


def hfp_cdf(x_grid, scenarios, p=None):
    """HFP cdf  F(x) = sum_t p_t * 1{eps_t <= x}  (17.29).
    With p=None the weights are 1/t_bar, i.e. the historical cdf (17.24)."""
    scenarios = np.asarray(scenarios)
    t = len(scenarios)
    p = np.full(t, 1.0 / t) if p is None else np.asarray(p)
    order = np.argsort(scenarios)
    cum_p = np.concatenate(([0.0], np.cumsum(p[order])))  # staircase heights
    idx = np.searchsorted(scenarios[order], x_grid, side="right")
    return cum_p[idx]


def hfp_mean(scenarios, p):
    """HFP mean  sum_t p_t eps_t."""
    return float(np.asarray(p) @ np.asarray(scenarios))


def hfp_sd(scenarios, p):
    """HFP standard deviation  sqrt(E_p[eps^2] - E_p[eps]^2)."""
    m = hfp_mean(scenarios, p)
    return float(np.sqrt(np.asarray(p) @ np.asarray(scenarios) ** 2 - m**2))


def hfp_es(scenarios, p, c):
    """HFP expected shortfall of the LEFT tail: probability-weighted average of the
    scenarios at or below the HFP c-quantile ("average of the worst c% outcomes")."""
    scenarios, p = np.asarray(scenarios), np.asarray(p)
    q = hfp_quantile(scenarios, p, c)
    tail = scenarios <= q
    return float(p[tail] @ scenarios[tail] / p[tail].sum())


def ks_2samp_hfp(x, y, px=None, py=None):
    """Two-sample Kolmogorov-Smirnov test with flexible probabilities.

    Statistic: D = sup_z |F_x(z) - F_y(z)|, the Glivenko-Cantelli distance between
    two HFP cdfs (17.29). With equal weights it equals scipy.stats.ks_2samp.
    p-value: asymptotic KS distribution with the sample sizes replaced by the
    effective numbers of scenarios ens(px), ens(py) - fewer effective scenarios,
    less evidence, bigger p-value.
    Returns (D, p_value)."""
    from scipy.stats import kstwo

    x, y = np.asarray(x), np.asarray(y)
    px = np.full(len(x), 1 / len(x)) if px is None else np.asarray(px)
    py = np.full(len(y), 1 / len(y)) if py is None else np.asarray(py)
    z = np.concatenate([x, y])                      # sup is reached at a data point
    d = float(np.max(np.abs(hfp_cdf(z, x, px) - hfp_cdf(z, y, py))))
    nx, ny = ens(px), ens(py)
    n_eff = max(int(round(nx * ny / (nx + ny))), 1)
    return d, float(kstwo.sf(d, n_eff))


def hfp_quantile(scenarios, p, c):
    """HFP c-quantile: inf{q : sum_{t: eps_t <= q} p_t >= c}  (17.44)."""
    scenarios = np.asarray(scenarios)
    order = np.argsort(scenarios)
    cum_p = np.cumsum(np.asarray(p)[order])
    # small tolerance so floating-point sums like 0.1+0.2 still hit c exactly
    return float(scenarios[order][np.argmax(cum_p >= c - 1e-12)])
