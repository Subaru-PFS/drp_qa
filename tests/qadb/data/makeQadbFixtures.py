"""Remake the qadb loader fixtures: four detector images of two Run25 calibration visits.

Engineering visits (Run25's arc and quartz sets), so the fixtures are public.
From the exports of PIPE2D-1956 (``u/wtg/qa-thresholds/009``) and the gate
fixture ``tests/metrics/data/iqQaMetrics-validation.parquet``:

- ``dmLineResidualSummary.parquet`` and ``dmLineResidualMetrics.parquet``: as
  ``dmLineResiduals`` stores them, with ``obsType`` added as
  `pfs.drp.qa.metrics.lineResiduals.addObsType` does from the iqQaMetrics.
- ``iqQaMetrics.parquet``: the gate fixture's columns, a subset of the task's.

    uv run --no-project --with pandas --with pyarrow python makeQadbFixtures.py [DIR]

``DIR`` holds the PIPE2D-1956 exports; default ``~/Projects/Subaru/PFS/reports/PIPE2D-1956/PIPE2D-1956``.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
STEM = "u_wtg_qa-thresholds_009"
#: visit, arm, spectrograph: an arc and a quartz, two cameras each.
IMAGES = [(133025, "b", 1), (133025, "r", 1), (133040, "b", 1), (133040, "r", 1)]


def main(directory: Path) -> None:
    keys = ["visit", "arm", "spectrograph"]
    wanted = pd.DataFrame(IMAGES, columns=keys)
    iq = pd.read_parquet(HERE.parents[1] / "metrics" / "data" / "iqQaMetrics-validation.parquet")
    iq = iq.merge(wanted, on=keys).sort_values(keys, ignore_index=True)
    assert len(iq) == len(IMAGES), len(iq)
    iq.to_parquet(HERE / "iqQaMetrics.parquet")
    for name in ("dmLineResidualSummary", "dmLineResidualMetrics"):
        table = pd.read_parquet(directory / f"{name}-{STEM}.parquet").merge(wanted, on=keys)
        if "obsType" not in table.columns:
            table = table.merge(iq[[*keys, "obsType"]], on=keys, how="left")
        table = table.sort_values(keys, ignore_index=True)
        assert table["visit"].nunique() == 2, name
        table.to_parquet(HERE / f"{name}.parquet")
        print(f"{name}: {len(table)} rows")
    print(f"iqQaMetrics: {len(iq)} rows")


if __name__ == "__main__":
    main(
        Path(
            sys.argv[1] if len(sys.argv) > 1 else "~/Projects/Subaru/PFS/reports/PIPE2D-1956/PIPE2D-1956"
        ).expanduser()
    )
