"""Remake the gate regression fixtures from the PIPE2D-1914 threshold run.

The inputs are what ``docs/qa-thresholds.ipynb`` caches for the collection
``u/wtg/qa-thresholds/003``: the validation visits' ``iqQaMetrics``, and the
thresholds derived from them. Both are engineering and calibration visits.

    uv run --no-project --with pandas --with pyarrow python makeGateFixtures.py [DIR]

``DIR`` holds the cached files; default ``~/Projects/Subaru/PFS/data``.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

STEM = "u_wtg_qa-thresholds_003"

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
metrics = pd.read_parquet(source / f"iqQaMetrics-{STEM}.parquet")[COLUMNS]
metrics.sort_values(["visit", "arm", "spectrograph"], ignore_index=True).to_parquet(
    here / "iqQaMetrics-validation.parquet", index=False
)
shutil.copyfile(source / f"iqQaThresholds-{STEM}.yaml", here / "iqQaThresholds-run25.yaml")
