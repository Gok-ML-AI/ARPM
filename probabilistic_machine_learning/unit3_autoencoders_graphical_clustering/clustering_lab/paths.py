"""
Central place for every filesystem path used in Unit 3.

Why this exists
---------------
The ARPM notebooks hard-code ``~/Databases/global-databases/...``. That breaks the
moment the notebook runs from PyCharm, a different machine, or a test. Every
notebook, pipeline and test imports its paths from here instead, so moving the
data only ever means changing one line (or one environment variable).
"""
from __future__ import annotations

import os
from pathlib import Path

# .../unit3_autoencoders_graphical_clustering
UNIT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = UNIT_ROOT / "data"
OUTPUTS_DIR: Path = UNIT_ROOT / "outputs"

# ARPM's original location, kept as a fallback so nothing breaks if you
# already have the ARPM database folder on this machine.
_ARPM_DEFAULT = (
    Path.home() / "Databases" / "global-databases" / "equities"
    / "db_stocks_SP500" / "db_stocks_sp.csv"
)


def stocks_csv_path() -> Path:
    """
    Locate ``db_stocks_sp.csv`` (S&P 500 stock values with a 2-level header:
    sector / ticker).

    Search order:
      1. environment variable ``ARPM_DB_STOCKS_SP`` (full path to the csv)
      2. ``<unit root>/data/db_stocks_sp.csv``
      3. ARPM's default ``~/Databases/.../db_stocks_sp.csv``
    """
    candidates = []
    env = os.environ.get("ARPM_DB_STOCKS_SP")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [DATA_DIR / "db_stocks_sp.csv", UNIT_ROOT.parents[1] / "data" / "db_stocks_sp.csv", _ARPM_DEFAULT]

    for path in candidates:
        if path.is_file():
            return path

    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "db_stocks_sp.csv not found. Copy it into the unit's data/ folder "
        f"or set ARPM_DB_STOCKS_SP. Looked in:\n  {tried}"
    )
