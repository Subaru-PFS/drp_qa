r"""Make ``dmDriftMetrics-validation.parquet``, the line-residual test fixture (PIPE2D-1921, PIPE2D-1956).

The judged rows of ``dmLineResidualMetrics`` for the validation visits, one row per detector,
reference and species: against the calibration (``statusType`` ``BOTH`` or ``UNFITTED``), the
drift, and against the visit's detectorMap (``RESERVED``), the scatter; with the calibration's
input visits from ``dmLineResidualSummary`` and the sequence from ``iqQaMetrics-validation.parquet``.
All are engineering and calibration visits.

The tables come from the thresholds notebook's cache: section 5 of ``docs/qa-thresholds.ipynb``
writes ``dmLineResidualMetrics-<run>.parquet`` and ``dmLineResidualSummary-<run>.parquet`` under
``~/.cache/drp_qa``, the run being the collection with ``/`` made ``_``. ``bin.src/exportMetrics.py``
names its files the same way, so its output directory serves too.

    PYTHONPATH=python uv run --no-project --with pandas --with pyarrow --with numpy --with pyyaml --with scipy \\
        python tests/metrics/data/makeDriftFixture.py [RUN [DIR]]

``RUN`` is the collection (default ``u/wtg/qa-thresholds/009``), ``DIR`` the directory (default
``~/.cache/drp_qa``). The fixture before this one, from ``u/wtg/qa-thresholds/008`` and
``dmDriftMonitor``, held rows against the calibration only, with no ``reference`` column; the test
adds it when missing.
"""

import sys
from pathlib import Path

import pandas as pd

from pfs.drp.qa.metrics.lineResiduals import BOTH, CALIB, RESERVED, UNFITTED, VISIT, wideResiduals

KEYS = ["visit", "arm", "spectrograph"]

run = sys.argv[1] if len(sys.argv) > 1 else "u/wtg/qa-thresholds/009"
source = Path(sys.argv[2] if len(sys.argv) > 2 else "~/.cache/drp_qa").expanduser()
stem = run.replace("/", "_")
here = Path(__file__).parent
metrics = pd.read_parquet(source / f"dmLineResidualMetrics-{stem}.parquet")
summary = pd.read_parquet(source / f"dmLineResidualSummary-{stem}.parquet")
kinds = pd.read_parquet(here / "iqQaMetrics-validation.parquet")[[*KEYS, "obsType", "seqName"]]

wide = pd.concat(
    [
        wideResiduals(metrics, CALIB, (BOTH, UNFITTED), summary, kinds),
        wideResiduals(metrics, VISIT, (RESERVED,), summary, kinds),
    ],
    ignore_index=True,
)
wide = wide.drop(columns=[column for column in ("calibId", "calibRun", "calibCreated") if column in wide])
wide = wide.astype(
    {
        "arm": "string",
        "cameraSet": "string",
        "reference": "string",
        "description": "string",
        "statusType": "string",
    }
)
wide = wide.sort_values([*KEYS, "reference", "description"], ignore_index=True)
path = here / "dmDriftMetrics-validation.parquet"
wide.to_parquet(path, index=False)
print(f"{len(wide)} rows -> {path} ({path.stat().st_size / 1e3:.0f} kB)")
