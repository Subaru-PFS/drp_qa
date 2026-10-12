"""Make ``iqQaThresholds-run25.yaml``, the thresholds ``imageQualityQa`` uses by default.

The source is the file ``docs/qa-thresholds.ipynb`` derived from the Run25
validation visits (collection ``u/wtg/qa-thresholds/003``), kept in the tests as
``tests/metrics/data/iqQaThresholds-run25.yaml`` beside the metrics it was
derived from. Adopting it (PIPE2D-1917) changed three things:

- **pctFlagged is not judged.** Its populations hold 5 visits each, so FAIL
  sits at the sample's maximum, and Run27's known-good n-arm flag rates are 2-5
  points higher, likely because Run27 had no calibrations of its own. The task
  config's fallback (n 15/20 %) is no better: Run25's known-good n-arm Neon
  sets flag 34 % and Run30's 26-35 % (comparison mode, PIPE2D-1929). One entry
  with neither WARN nor FAIL stops the search, so the config isn't used either.
- **Twilight nLines is not judged.** On a twilight frame the count follows the
  sky brightness, not the instrument.
- **Quartz nLines is set by hand.** Its derivation was degenerate (WARN = FAIL
  at the sample's minimum): WARN is 0.5 % and FAIL 1 % below the smallest
  known-good Run25 value of the arm, rounded down to 1000.

Run from the repository root:

    PYTHONPATH=python python python/pfs/drp/qa/metrics/data/makeRun25Thresholds.py
"""

import math
from pathlib import Path

import pandas as pd
import yaml

from pfs.drp.qa.metrics.calibration import GOOD, labelRows
from pfs.drp.qa.metrics.validationVisits import loadValidationVisits

ROOT = Path(__file__).parents[6]
SOURCE = ROOT / "tests" / "metrics" / "data" / "iqQaThresholds-run25.yaml"
METRICS = ROOT / "tests" / "metrics" / "data" / "iqQaMetrics-validation.parquet"
OUTPUT = Path(__file__).parent / "iqQaThresholds-run25.yaml"

#: Fractions below the smallest known-good value for quartz nLines WARN and FAIL.
TRACE_MARGINS = (0.005, 0.01)
ADOPTED = "2026-10-06"
#: When pctFlagged stopped being judged.
FLAGS_UNJUDGED = "2026-10-08"


def adopt(document: dict, metrics: pd.DataFrame) -> dict:
    """Return the derived thresholds ``document`` as adopted.

    Parameters
    ----------
    document : `dict`
        The derived thresholds file, as YAML.
    metrics : `pandas.DataFrame`
        The ``iqQaMetrics`` they were derived from.

    Returns
    -------
    `dict`
        The adopted thresholds file, as YAML.
    """
    labelled = labelRows(metrics, loadValidationVisits(), "nLines")
    good = labelled[(labelled["validation"] == GOOD) & (labelled["seqName"] == "Trace")]
    smallest = good.groupby("arm")["nLines"].min()

    entries = []
    for entry in document["thresholds"]:
        entry = dict(entry)
        seqName = entry["population"].get("seqName")
        if entry["metric"] == "pctFlagged":
            continue
        if entry["metric"] == "nLines" and seqName == "Twilight sky":
            entry |= {"warn": None, "fail": None, "degenerate": False}
            entry["provenance"] = (
                f"Not judged (adopted {ADOPTED}, PIPE2D-1917): on a twilight frame nLines follows the sky "
                f"brightness, not the instrument. Derived values were: {entry['provenance']}"
            )
        elif entry["metric"] == "nLines" and seqName == "Trace":
            minimum = int(smallest[entry["population"]["arm"]])
            warn, fail = (1000 * math.floor(minimum * (1 - margin) / 1000) for margin in TRACE_MARGINS)
            entry |= {"warn": float(warn), "fail": float(fail), "degenerate": False}
            entry["provenance"] = (
                f"Set by hand {ADOPTED} (PIPE2D-1917): the derivation from visits {entry['visitRange']} "
                f"(n={entry['nGood']}) was degenerate, WARN = FAIL at the sample minimum. WARN 0.5 % and "
                f"FAIL 1 % below the smallest known-good value, {minimum}."
            )
        entries.append(entry)
    entries.append(
        {
            "metric": "pctFlagged",
            "population": {},
            "higherIsWorse": True,
            "absolute": False,
            "warn": None,
            "fail": None,
            "provenance": (
                f"Not judged ({FLAGS_UNJUDGED}, PIPE2D-1929) until there is a valid reference: the Run25 "
                "derivation had 5 visits per population, and the config's n-arm 15/20 % is below the known-good "
                "n-arm Neon sets of Run25 (34 %) and Run30 (26-35 %)."
            ),
        }
    )
    return {key: value for key, value in document.items() if key != "thresholds"} | {"thresholds": entries}


def main() -> None:
    document = adopt(yaml.safe_load(SOURCE.read_text()), pd.read_parquet(METRICS))
    header = f"# Made by {Path(__file__).name} from {SOURCE.relative_to(ROOT)}; see its docstring.\n"
    OUTPUT.write_text(header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=110))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
