"""Make ``dmDriftMetrics-validation.parquet``, the drift test fixture (PIPE2D-1921).

The judged rows of ``dmDriftMetrics`` (``statusType`` ``BOTH`` or ``UNFITTED``)
for the validation visits, from ``u/wtg/qa-thresholds/008`` (005's reductions
with ``dmDriftMonitor`` run), one row per detector and species, with the
calibration's input visits from ``dmDriftSummary`` and the sequence from
``iqQaMetrics-validation.parquet``. All are engineering and calibration visits.

    uv run --no-project --with pandas --with pyarrow python makeDriftFixture.py [DIR]

``DIR`` holds the files ``reports/PIPE2D-1921/exportDrift.py`` wrote; default
``~/Projects/Subaru/PFS/reports/PIPE2D-1921``.
"""

import sys
from pathlib import Path

import pandas as pd

STEM = "u_wtg_qa-thresholds_008"
KEYS = ["visit", "arm", "spectrograph"]
METRICS = ["nLines", "linesPerFiber", "medianDeltaX", "robustRmsDeltaX", "medianDeltaY", "robustRmsDeltaY"]

source = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Projects/Subaru/PFS/reports/PIPE2D-1921").expanduser()
here = Path(__file__).parent
metrics = pd.read_parquet(source / f"dmDriftMetrics-{STEM}.parquet")
summary = pd.read_parquet(source / f"dmDriftSummary-{STEM}.parquet")
kinds = pd.read_parquet(here / "iqQaMetrics-validation.parquet")[[*KEYS, "obsType", "seqName"]]

judged = metrics[metrics["statusType"].isin(["BOTH", "UNFITTED"])]
wide = judged.pivot(
    index=[*KEYS, "cameraSet", "description", "statusType"], columns="metric", values="value"
).reset_index()
wide = wide.reindex(columns=[*KEYS, "cameraSet", "description", "statusType", *METRICS])
wide = wide.merge(summary[[*KEYS, "calibFirstVisit", "calibLastVisit"]], on=KEYS).merge(
    kinds, on=KEYS, how="left"
)
wide = wide.astype({"arm": "string", "cameraSet": "string", "description": "string", "statusType": "string"})
wide = wide.sort_values([*KEYS, "description"], ignore_index=True)
wide.columns.name = None
path = here / "dmDriftMetrics-validation.parquet"
wide.to_parquet(path, index=False)
print(f"{len(wide)} rows -> {path} ({path.stat().st_size / 1e3:.0f} kB)")
