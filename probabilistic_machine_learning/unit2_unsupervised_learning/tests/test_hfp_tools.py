"""Tests for historical_lab/hfp_tools.py  (run from repo root:  pytest tests)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "historical_lab"))
from hfp_tools import exp_decay_probs, ens, hfp_cdf, hfp_mean, hfp_sd, hfp_quantile  # noqa: E402


def test_exp_decay_probs_sum_to_one_and_halve_every_half_life():
    p = exp_decay_probs(t_bar=100, tau_hl=10)
    assert p.sum() == pytest.approx(1.0)
    assert p[-1] / p[-11] == pytest.approx(2.0)      # 10 steps back = half the weight


def test_ens_limits():
    assert ens(np.full(50, 1 / 50)) == pytest.approx(50)    # uniform -> t_bar
    assert ens(np.r_[1.0, np.zeros(9)]) == pytest.approx(1)  # point mass -> 1


def test_historical_cdf_small_example():
    # the 4-return example from the notes: F(1%) = 3/4
    eps = np.array([-0.02, 0.01, 0.03, 0.01])
    assert hfp_cdf(np.array([0.01]), eps)[0] == pytest.approx(0.75)
    assert hfp_cdf(np.array([-0.05, 0.05]), eps).tolist() == [0.0, 1.0]


def test_hfp_mean_sd_quantile_small_example():
    eps, p = np.array([2.0, 4.0, 6.0]), np.array([0.2, 0.3, 0.5])
    assert hfp_mean(eps, p) == pytest.approx(4.6)
    assert hfp_sd(eps, p) == pytest.approx(np.sqrt(0.2*4 + 0.3*16 + 0.5*36 - 4.6**2))
    assert hfp_quantile(eps, p, 0.2) == 2.0     # cumulative 0.2 reached at 2
    assert hfp_quantile(eps, p, 0.5) == 4.0     # 0.2 + 0.3 = 0.5 reached at 4
