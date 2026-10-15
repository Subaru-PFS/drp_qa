"""Remake the gate regression fixtures from the threshold runs.

The inputs are what ``docs/qa-thresholds.ipynb`` caches for each collection:
the validation visits' ``iqQaMetrics``, and the thresholds derived from them.
All are engineering and calibration visits.

- ``iqQaMetrics-validation.parquet`` and ``iqQaThresholds-run25.yaml``, from
  ``u/wtg/qa-thresholds/004``: ``003``'s reductions with ``imageQualityQa``
  rerun once the calexp trace width worked (PIPE2D-1918). The adopted
  thresholds are made from these.
- ``iqQaMetrics-config.parquet``, from ``u/wtg/qa-thresholds/003``, whose
  verdicts the task made from its config alone, before it read a thresholds
  file: the regression check of ``gate.configThresholds``.
- ``linesPerFiber``, ``nLitFibers`` and ``nFibers`` in the validation fixture,
  from ``u/wtg/qa-thresholds/005``: 004's reductions with ``imageQualityQa``
  rerun once it counted the fibers lit (PIPE2D-1935). 004 was measured before,
  and its other columns are unchanged in 005.

    uv run --no-project --with pandas --with pyarrow python makeGateFixtures.py [DIR]

``DIR`` holds the cached files; default ``~/Projects/Subaru/PFS/data``.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

STEM = "u_wtg_qa-thresholds_004"
LIT_FIBER_STEM = "u_wtg_qa-thresholds_005"
CONFIG_STEM = "u_wtg_qa-thresholds_003"

#: The columns the gate reads, and the verdict the task stored.
COLUMNS = [
    "visit",
    "arm",
    "spectrograph",
    "obsType",
    "seqName",
    "traceOnly",
    "medFwhm",
    "pctFlagged",
    "medDxCenter",
    "dxCenterRms",
    "nLines",
    "qaStatus",
]

source = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Projects/Subaru/PFS/data").expanduser()
here = Path(__file__).parent
#: The columns 004's metrics lack, from 005 (PIPE2D-1935).
LIT_FIBER_COLUMNS = ["linesPerFiber", "nLitFibers", "nFibers"]
KEYS = ["visit", "arm", "spectrograph"]

for stem, name in ((STEM, "iqQaMetrics-validation.parquet"), (CONFIG_STEM, "iqQaMetrics-config.parquet")):
    metrics = pd.read_parquet(source / f"iqQaMetrics-{stem}.parquet")[COLUMNS]
    if stem == STEM:
        litFibers = pd.read_parquet(source / f"iqQaMetrics-{LIT_FIBER_STEM}.parquet")[
            KEYS + LIT_FIBER_COLUMNS
        ]
        metrics = metrics.merge(litFibers, on=KEYS, how="left", validate="one_to_one")
    metrics.sort_values(["visit", "arm", "spectrograph"], ignore_index=True).to_parquet(
        here / name, index=False
    )
shutil.copyfile(source / f"iqQaThresholds-{STEM}.yaml", here / "iqQaThresholds-run25.yaml")
