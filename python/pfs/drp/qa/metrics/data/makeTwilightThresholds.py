#!/usr/bin/env python
"""Make ``twilightThresholds.yaml``, the thresholds ``bin.src/twilightQa.py`` judges twilight sequences with.

The values are `pfs.drp.qa.metrics.twilight.twilightThresholds`'s provisional ones (PIPE2D-1973): a
detector's level against its sequence's (``twilightLevelOffset``, on its absolute value), the field's RMS
beyond photon noise (``twilightResidRms``) and its deepest coherent patch (``twilightPatch``), with the n
arm deliberately unjudged. They are written as a thresholds file so that the judge records what it judged
with (`pfs.drp.qa.metrics.gate.thresholdsRecord`) and the qadb loader cites it in ``qa_gate``
(``twilight_thresholds_id``), as it does for the other tasks' files (INSTRM-3089).

    python makeTwilightThresholds.py
"""

import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from pfs.drp.qa.metrics.calibration import THRESHOLDS_VERSION
from pfs.drp.qa.metrics.twilight import TWILIGHT_THRESHOLDS_FILE, twilightThresholds

#: Where the provisional values came from.
COLLECTION = "prototype over 16 twilight sequences, 2025-03 to 2026-09 (PIPE2D-1973)"


def entries(table) -> list[dict]:
    """Return the `twilightThresholds` rows as thresholds-file entries."""
    rows = []
    for _, row in table.iterrows():
        population = {} if pd.isna(row["arm"]) else {"arm": str(row["arm"])}
        rows.append(
            {
                "metric": row["metric"],
                "population": population,
                "higherIsWorse": bool(row["higherIsWorse"]),
                "absolute": bool(row["absolute"]),
                "method": "fixed",
                "warn": None if pd.isna(row["warn"]) else float(row["warn"]),
                "fail": None if pd.isna(row["fail"]) else float(row["fail"]),
                "provenance": row["provenance"],
            }
        )
    return rows


def main() -> None:
    path = Path(__file__).parent / TWILIGHT_THRESHOLDS_FILE
    try:
        version = subprocess.run(
            ["git", "describe", "--dirty"], capture_output=True, text=True, check=True, cwd=path.parent
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        version = "unknown"
    document = {
        "version": THRESHOLDS_VERSION,
        "derivedOn": date.today().isoformat(),
        "collection": COLLECTION,
        "referenceRuns": [],
        "drpQaVersion": version,
        "thresholds": entries(twilightThresholds()),
    }
    text = f"# Made by {Path(__file__).name} from pfs.drp.qa.metrics.twilight.twilightThresholds; see its docstring.\n"
    text += yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=110)
    path.write_text(text)
    print(f"Wrote {len(document['thresholds'])} entries to {path}")


if __name__ == "__main__":
    main()
