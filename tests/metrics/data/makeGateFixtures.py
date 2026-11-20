"""Remake the gate fixtures: stored numbers of calibration and engineering visits, without verdicts.

- ``iqQaMetrics-validation.parquet``, from ``u/wtg/qa-thresholds/010``: arcs, quartz and twilights of
  Runs 25, 27, 29 and 30 reduced afresh on one pipeline, with ``imageQualityQa``. Run25's are a stable
  calibration set; Run27's hold the SM1 defocus and offsets against a calibration built later; Run30's
  hold the visits with obstructed fibers. ``dmDriftMetrics-validation.parquet`` (``makeDriftFixture.py``)
  holds the same visits' line residuals.
- ``iqQaMetrics-iis-run30.parquet``, from ``u/wtg/qa-iis/003b``: Run30's nightly frames lit through
  the engineering fibers (an IIS Ne arc, an IIS Kr arc and an IIS halogen quartz, 15 nights).

The inputs are the collections' ``iqQaMetrics`` exported to parquet, one file per collection, named
``iqQaMetrics-<collection with / as _>.parquet``. A fixture whose export is not in ``DIR`` is kept, less
any column not listed here.

    uv run --no-project --with pandas --with pyarrow python makeGateFixtures.py [DIR]

``DIR`` holds the exports; default ``~/Projects/Subaru/PFS/reports/PIPE2D-1938``.
"""

import sys
from pathlib import Path

import pandas as pd

#: The columns the gate reads.
COLUMNS = [
    "visit",
    "arm",
    "spectrograph",
    "obsType",
    "illumination",
    "seqName",
    "traceOnly",
    "medFwhm",
    "pctFlagged",
    "medDxCenter",
    "dxCenterRms",
    "nLines",
    "linesPerFiber",
    "nLitFibers",
    "nFibers",
]

source = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Projects/Subaru/PFS/reports/PIPE2D-1938").expanduser()
here = Path(__file__).parent

for stem, name in (
    ("u_wtg_qa-thresholds_010", "iqQaMetrics-validation.parquet"),
    ("u_wtg_qa-iis_003b", "iqQaMetrics-iis-run30.parquet"),
):
    path = source / f"iqQaMetrics-{stem}.parquet"
    metrics = pd.read_parquet(path if path.exists() else here / name)
    metrics = metrics[[column for column in COLUMNS if column in metrics.columns]]
    metrics.sort_values(["visit", "arm", "spectrograph"], ignore_index=True).to_parquet(
        here / name, index=False
    )
    print(f"{name}: {len(metrics)} rows from {path if path.exists() else 'the fixture itself'}")
