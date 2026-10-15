"""Make ``iqQaThresholds-run25.yaml``, the thresholds ``imageQualityQa`` uses by default.

The source is the file ``docs/qa-thresholds.ipynb`` derived from the Run25
validation visits (collection ``u/wtg/qa-thresholds/004``: the reductions of
``003`` with ``imageQualityQa`` rerun once the calexp trace width worked,
PIPE2D-1918), kept in the tests as ``tests/metrics/data/iqQaThresholds-run25.yaml``
beside the metrics it was derived from. Arc metrics are the same in both runs.
Adopting it changed five things:

- **pctFlagged is not judged.** Its populations hold 5 visits each, so FAIL
  sits at the sample's maximum, and Run27's known-good n-arm flag rates are 2-5
  points higher, likely because Run27 had no calibrations of its own. The task
  config's fallback (n 15/20 %) is no better: Run25's known-good n-arm Neon
  sets flag 34 % and Run30's 26-35 % (comparison mode, PIPE2D-1929). One entry
  with neither WARN nor FAIL stops the search, so the config isn't used either.
- **linesPerFiber replaces nLines** (PIPE2D-1935). ``nLines`` counts rows of
  ``lines``: on an arc mostly trace centroids, whose number follows the
  illumination, and only the fibers a design lights, so an arc lighting one
  fiber group failed it. ``linesPerFiber`` counts unflagged lamp lines (trace
  centroids on a quartz) per lit fiber, measured in ``u/wtg/qa-thresholds/005``
  (004's reductions, ``imageQualityQa`` rerun) and merged into the validation
  fixture. It is judged as a tolerance below the
  Run25 known-good median of each arm and sequence (``seqName``): arcs 25 %/50 %
  (known-good arcs reach 19 % below, on n HgCd; the obstructed ones sit 60-98 %
  below), quartz 2.5 %/5 % (known-good 1.1 %, obstructed 9-55 %). An arm whose
  lamp gives under ``MIN_LINES_PER_FIBER`` lines per fiber (b Argon) is not
  judged. Twilight is not measured.
- **medFwhm is judged per detector and lamp** (PIPE2D-1942): WARN 3 % and FAIL
  5 % above the Run25 known-good median of its arm, spectrograph, obsType and
  lamp, adopted as derived. The spectrographs of one arm differ from each
  other by more than any one of them varies (r quartz: r2 3.27 px, r4 3.39 px,
  each within 0.03 px), so the per-arm percentiles put each arm's widest
  spectrograph at FAIL. The per-arm entries stay, judging
  only an image whose detector and lamp have no Run25 median.
- **Per-arm quartz medFwhm is judged on b, r and n as derived** (PIPE2D-1918),
  from the calexp width. The m arm's derivation has 16 known-good values,
  below the 20-sample floor, so it judges nothing: the config's 3.2 px WARN
  would flag every known-good one (3.31-3.42 px).

Run from the repository root:

    PYTHONPATH=python python python/pfs/drp/qa/metrics/data/makeRun25Thresholds.py
"""

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from pfs.drp.qa.metrics.calibration import GOOD, labelRows
from pfs.drp.qa.metrics.thresholds import deriveTolerance
from pfs.drp.qa.metrics.validationVisits import loadValidationVisits

ROOT = Path(__file__).parents[6]
SOURCE = ROOT / "tests" / "metrics" / "data" / "iqQaThresholds-run25.yaml"
METRICS = ROOT / "tests" / "metrics" / "data" / "iqQaMetrics-validation.parquet"
OUTPUT = Path(__file__).parent / "iqQaThresholds-run25.yaml"

#: WARN and FAIL as fractions below the known-good median of linesPerFiber, by obsType.
LINES_PER_FIBER_MARGINS = {"arc": (0.25, 0.5), "trace": (0.025, 0.05)}
#: Below this known-good median (lines per lit fiber), a lamp's count on an arm is not judged.
MIN_LINES_PER_FIBER = 5.0
#: When linesPerFiber was adopted.
LINES_PER_FIBER_ADOPTED = "2026-10-14"
#: When pctFlagged stopped being judged.
FLAGS_UNJUDGED = "2026-10-08"
#: When quartz medFwhm was adopted.
TRACE_FWHM_ADOPTED = "2026-10-12"
#: When medFwhm per detector and lamp was adopted.
TOLERANCE_ADOPTED = "2026-10-13"


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
    entries = []
    for entry in document["thresholds"]:
        entry = dict(entry)
        population = entry["population"]
        if entry["metric"] in ("pctFlagged", "nLines"):
            continue
        if entry["method"] == "tolerance":
            entry["provenance"] = f"Adopted {TOLERANCE_ADOPTED} (PIPE2D-1942). {entry['provenance']}"
        elif entry["metric"] == "medFwhm" and population.get("obsType") == "trace":
            if entry["reliable"]:
                entry["provenance"] = f"Adopted {TRACE_FWHM_ADOPTED} (PIPE2D-1918). {entry['provenance']}"
            else:
                entry |= {"warn": None, "fail": None}
                entry["provenance"] = (
                    f"Not judged ({TRACE_FWHM_ADOPTED}, PIPE2D-1918): the derivation from visits "
                    f"{entry['visitRange']} has n={entry['nGood']}, below the 20-sample floor, and the config's "
                    "medFwhm WARN would flag every known-good value. Derived values were: "
                    f"{entry['provenance']}"
                )
        entries.append(entry)
    entries.extend(linesPerFiberEntries(metrics))
    entries.append(
        {
            "metric": "pctFlagged",
            "population": {},
            "higherIsWorse": True,
            "absolute": False,
            "method": "percentile",
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


def linesPerFiberEntries(metrics: pd.DataFrame) -> list[dict]:
    """Return the linesPerFiber entries: a tolerance below each arm and sequence's known-good median.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        The validation visits' ``iqQaMetrics``, with ``linesPerFiber``.

    Returns
    -------
    `list` [`dict`]
        One entry per arm and ``seqName`` of the known-good arcs and quartz.
    """
    labelled = labelRows(metrics, loadValidationVisits(), "linesPerFiber")
    good = labelled[
        (labelled["validation"] == GOOD)
        & labelled["obsType"].isin(list(LINES_PER_FIBER_MARGINS))
        & labelled["linesPerFiber"].notna()
    ]
    entries = []
    for (arm, seqName), group in good.groupby(["arm", "seqName"], sort=True):
        warnFraction, failFraction = LINES_PER_FIBER_MARGINS[group["obsType"].iloc[0]]
        visitRange = f"{group['visit'].min()}-{group['visit'].max()}"
        suggestion = deriveTolerance(
            group["linesPerFiber"],
            metric=f"linesPerFiber[{arm}/{seqName}]",
            warnFraction=warnFraction,
            failFraction=failFraction,
            higherIsWorse=False,
            visitRange=visitRange,
            derivedOn=date.fromisoformat(LINES_PER_FIBER_ADOPTED),
        )
        entry = {
            "metric": "linesPerFiber",
            "population": {"arm": arm, "seqName": seqName},
            "higherIsWorse": False,
            "absolute": False,
            "method": "tolerance",
            "warn": float(suggestion.warn),
            "fail": float(suggestion.fail),
            "nGood": int(suggestion.numSamples),
            "nGoodVisits": int(group["visit"].nunique()),
            "visitRange": visitRange,
            "reliable": bool(suggestion.reliable),
            "failBounded": bool(suggestion.failBounded),
            "degenerate": False,
            "provenance": f"Adopted {LINES_PER_FIBER_ADOPTED} (PIPE2D-1935). {suggestion.provenance}",
        }
        if suggestion.median < MIN_LINES_PER_FIBER:
            entry |= {"warn": None, "fail": None}
            entry["provenance"] = (
                f"Not judged ({LINES_PER_FIBER_ADOPTED}, PIPE2D-1935): the lamp gives a median of "
                f"{suggestion.median:.2g} lines per lit fiber on this arm, too few to tell a loss of light "
                f"from chance. Derived values were: {suggestion.provenance}"
            )
        entries.append(entry)
    return entries


def main() -> None:
    document = adopt(yaml.safe_load(SOURCE.read_text()), pd.read_parquet(METRICS))
    header = f"# Made by {Path(__file__).name} from {SOURCE.relative_to(ROOT)}; see its docstring.\n"
    OUTPUT.write_text(header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=110))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
