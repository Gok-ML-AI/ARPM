"""
s_glivenko_cantelli.py
ARPM Probabilistic ML - Unit 2, Section 17.2.1 (Historical estimation)

Verifies the Glivenko-Cantelli theorem:
    sup_x |F_hist(x) - F(x)| -> 0   as t_bar -> infinity   (eq. 17.26)
for an i.i.d. lognormal invariant eps_t ~ LogN(mu, sigma2).

PyCharm port of s_glivenko_cantelli.ipynb, with two fixes:
  1. scipy's lognorm takes the shape parameter s = sigma (std of ln eps),
     not sigma2. The original passes sigma2 straight in as s.
  2. The original passes mu as `loc` AND uses scale=exp(mu), which counts mu
     twice. It only works because mu = 0. Here loc stays at 0.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, kstest

from hfp_tools import hfp_cdf

# ---------------------------------------------------------------------------
# 1. Generate lognormal scenarios
# ---------------------------------------------------------------------------
mu = 0.0          # location of ln(eps)
sigma2 = 0.25     # variance of ln(eps)  -> ARPM's LogN(0, 0.25)
t_bar = 2500      # number of observations
rng = np.random.default_rng(42)

dist = lognorm(s=np.sqrt(sigma2), scale=np.exp(mu))  # true distribution F_eps
eps = dist.rvs(size=t_bar, random_state=rng)


# ---------------------------------------------------------------------------
# 2. Historical cdf (eq. 17.24) and true cdf
# ---------------------------------------------------------------------------
x_grid = np.linspace(eps.min(), eps.max(), 1000)
cdf_hist = hfp_cdf(x_grid, eps)
cdf_true = dist.cdf(x_grid)

# ---------------------------------------------------------------------------
# 3. True pdf vs normalised histogram (the historical pdf, "smoothed" into bins)
# ---------------------------------------------------------------------------
pdf_true = dist.pdf(x_grid)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(x_grid, cdf_true, label="True")
ax[0].step(x_grid, cdf_hist, where="post", label="Historical")
ax[0].set_title(f"cdf, t_bar = {t_bar}")
ax[0].legend()

ax[1].hist(eps, bins=80, density=True, alpha=0.6, label="Historical (histogram)")
ax[1].plot(x_grid, pdf_true, label="True pdf")
ax[1].set_title("pdf")
ax[1].legend()
fig.tight_layout()

# ---------------------------------------------------------------------------
# 4. Extra: watch the theorem happen (sup distance shrinks as t_bar grows)
# ---------------------------------------------------------------------------
# kstest's statistic is exactly sup_x |F_hist(x) - F(x)|  (eq. 17.26)
t_bars = np.array([50, 200, 1_000, 5_000, 20_000, 100_000])
sup_dist = np.array([
    kstest(dist.rvs(size=t, random_state=rng), dist.cdf).statistic
    for t in t_bars
])

print(f"{'t_bar':>8} | {'sup|F_hist - F|':>16} | {'1/sqrt(t_bar)':>13}")
for t, d in zip(t_bars, sup_dist):
    print(f"{t:>8} | {d:>16.4f} | {1/np.sqrt(t):>13.4f}")

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.loglog(t_bars, sup_dist, "o-", label="sup |F_hist - F|")
ax2.loglog(t_bars, 1 / np.sqrt(t_bars), "--", label="1/sqrt(t_bar)")
ax2.set_xlabel("t_bar")
ax2.set_title("Glivenko-Cantelli: convergence")
ax2.legend()
fig2.tight_layout()

plt.show()
