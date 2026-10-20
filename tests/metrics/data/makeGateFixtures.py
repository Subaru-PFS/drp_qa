"""Remake the gate regression fixtures from the threshold runs.

The inputs are what ``docs/qa-thresholds.ipynb`` caches for each collection:
the validation visits' ``iqQaMetrics``, and the thresholds derived from them.
All are engineering and calibration visits.

- ``iqQaMetrics-validation.parquet`` and ``iqQaThresholds-run25.yaml``, from
  ``u/wtg/qa-thresholds/010`` (PIPE2D-1938): every validation visit reduced
  afresh with datamodel w.2026.40, obs_pfs w.2026.40 and drp_stella
  w.2026.41-6-ge56938d2, the pipeline of the Run25-30 comparison, with
  ``imageQualityQa`` and ``dmLineResiduals``; the validation set with Run29's
  detectorMap set as the b2 and n4 quartz reference. The adopted thresholds are
  made from these by ``makeRun25Thresholds.py``.
- ``iqQaMetrics-config.parquet``, from ``u/wtg/qa-thresholds/003``, whose
  verdicts the task made from its config alone, before it read a thresholds
  file: the regression check of ``gate.configThresholds``. Remade only when its
  cache is in ``DIR``.

    uv run --no-project --with pandas --with pyarrow python makeGateFixtures.py [DIR]

``DIR`` holds the cached files; default ``~/Projects/Subaru/PFS/reports/PIPE2D-1938``.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

STEM = "u_wtg_qa-thresholds_010"
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

source = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Projects/Subaru/PFS/reports/PIPE2D-1938").expanduser()
here = Path(__file__).parent
#: The lit-fiber columns (PIPE2D-1935), beside COLUMNS.
LIT_FIBER_COLUMNS = ["linesPerFiber", "nLitFibers", "nFibers"]

for stem, name, columns in (
    (STEM, "iqQaMetrics-validation.parquet", [*COLUMNS, *LIT_FIBER_COLUMNS]),
    (CONFIG_STEM, "iqQaMetrics-config.parquet", COLUMNS),
):
    path = source / f"iqQaMetrics-{stem}.parquet"
    if not path.exists():
        print(f"{name}: no {path}, kept as is")
        continue
    metrics = pd.read_parquet(path)[columns]
    metrics.sort_values(["visit", "arm", "spectrograph"], ignore_index=True).to_parquet(
        here / name, index=False
    )
    print(f"{name}: {len(metrics)} rows from {stem}")
shutil.copyfile(source / f"iqQaThresholds-{STEM}.yaml", here / "iqQaThresholds-run25.yaml")
print("iqQaThresholds-run25.yaml copied from", STEM)
