"""
Shared test fixtures.

Puts the unit root on sys.path (so `import clustering_lab` works when pytest
is launched from PyCharm or the repo root) and provides a small synthetic
copy of db_stocks_sp.csv, so no test ever depends on the real database.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

UNIT_ROOT = Path(__file__).resolve().parents[1]
if str(UNIT_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIT_ROOT))


def make_fake_db_stocks_sp(
    path: Path, n_sectors: int = 4, per_sector: int = 5, t_bar: int = 300, seed: int = 0
) -> Path:
    """
    Write a csv with ARPM's layout: 2-row header (sector, ticker), dates as index.
    Returns follow a sector factor model, so stocks in the same sector correlate.
    """
    rng = np.random.default_rng(seed)
    n_bar = n_sectors * per_sector
    sector_of = np.repeat(np.arange(n_sectors), per_sector)
    market = rng.normal(0, 0.01, t_bar)
    sector_f = rng.normal(0, 0.01, (t_bar, n_sectors))
    eps = (0.5 * market[:, None] + sector_f[:, sector_of]
           + rng.normal(0, 0.005, (t_bar, n_bar)))
    v = 100 * np.exp(np.cumsum(eps, axis=0))
    cols = pd.MultiIndex.from_arrays(
        [[f"Sector{s}" for s in sector_of], [f"S{s}_{i}" for s, i in
                                             zip(sector_of, np.tile(np.arange(per_sector), n_sectors))]]
    )
    dates = pd.bdate_range("2020-01-01", periods=t_bar)
    pd.DataFrame(v, index=dates, columns=cols).to_csv(path)
    return path


@pytest.fixture
def fake_db_csv(tmp_path):
    return make_fake_db_stocks_sp(tmp_path / "db_stocks_sp.csv")
