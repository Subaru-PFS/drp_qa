"""Make ``iqQaThresholds-iis-run30.yaml``: the thresholds for IIS frames (PIPE2D-1954).

An IIS frame lights the 16 engineering fibers of each spectrograph with the IIS
lamps, and ``imageQualityQa`` measures it on those fibers, with
``illumination: iis`` (PIPE2D-1954). Its widths (about 1.8 px) are not the
science fibers' (2.5-3.3 px), so it has thresholds of its own, which judge IIS
frames alone: every entry here names ``illumination: iis``, and an entry without
it is for full-slit frames (`pfs.drp.qa.metrics.gate.DEFAULT_POPULATION`).

The reference is Run30's nightly "ImageQuality" sequences, the first IIS series
taken every night: an IIS Ne arc, an IIS Kr arc and an IIS halogen quartz from
2026-09-01 to 09-15, reduced afresh on one pipeline (``u/wtg/qa-iis/003b``) and
listed as known good in the validation set, less 148829 n4 (its lamp header read
off, so it was reduced against the sky line list; known bad) and the night of
149593-149595 (SM2 and SM4 moved 0.26-0.39 px). The metrics are
``tests/metrics/data/iqQaMetrics-iis-run30.parquet``; `derive` writes the derived
file beside them and `adopt` turns it into the shipped one:

- **medFwhm is judged per detector and lamp**, WARN 3 % and FAIL 5 % above the
  known-good median (as the PFI arcs, PIPE2D-1942): the 14 nights sit within
  1.5 % of their medians on b and r and on the n-arm Kr arc and quartz.
  Not judged: the n-arm Ne arc, whose 15 faint IIS lines per fiber give a
  median width that scatters 2-4 % night to night and crossed WARN on 1-3 of 14
  known-good nights per camera (the Kr arc judges the n arm's focus nightly);
  and the m arm, taken on one night (4 values). The per-arm percentiles judge
  only an image whose detector and lamp have no entry.
- **linesPerFiber** is a tolerance below each arm and sequence's known-good median:
  arcs 25 %/50 %, quartz 2.5 %/5 % (as PIPE2D-1935). The m arm is not judged.
- **medDxCenter** keeps the task config's physical limits, WARN 1 px and FAIL 2 px:
  a flexure limit is not a population property. The derived percentiles (0.1 px)
  are far too tight. The known-bad 148829 n4 sits at 1.98 px.
- **pctFlagged is not judged**: 54-96 % of the IIS lamps' lines are flagged on
  every night, a property of the line lists and the 60 um fibers, not of the
  exposure.

Run from the repository root:

    PYTHONPATH=python python python/pfs/drp/qa/metrics/data/makeIisThresholds.py
"""

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from pfs.drp.qa.metrics.calibration import GOOD, calibrate, labelRows, readThresholds, writeThresholds
from pfs.drp.qa.metrics.thresholds import deriveTolerance
from pfs.drp.qa.metrics.validationVisits import loadValidationVisits

ROOT = Path(__file__).parents[6]
METRICS = ROOT / "tests" / "metrics" / "data" / "iqQaMetrics-iis-run30.parquet"
SOURCE = ROOT / "tests" / "metrics" / "data" / "iqQaThresholds-iis-run30.yaml"
OUTPUT = Path(__file__).parent / "iqQaThresholds-iis-run30.yaml"

#: The collection the metrics came from: Run30's nightly IIS frames reduced afresh.
COLLECTION = "u/wtg/qa-iis/003b"
#: The pipeline that reduced them. datamodel and drp_stella are as the pfsArm headers record
#: them (reports/PIPE2D-1954/pipelines-003b.parquet). obs_pfs was master at 56b12d7, tagged
#: w.2026.43, set up from a checkout whose version module had not been rebuilt, so its headers
#: say "w.2026.16-1-ga922722"; a reduction from that checkout reads as another pipeline to
#: the gate's provisional check until the module is rebuilt.
PIPELINE = {"datamodel": "w.2026.40", "obs_pfs": "w.2026.43", "drp_stella": "w.2026.41-6-ge56938d2"}
#: The drp_qa that measured them: tickets/PIPE2D-1954 at the commit before this file's.
DRP_QA_VERSION = "w.2026.35-217-g6c5c143"
#: When these were derived and adopted.
DERIVED_ON = date(2026, 10, 26)
#: The population column every entry names: the IIS frames' own thresholds.
ILLUMINATION = "iis"
#: WARN and FAIL as fractions below the known-good median of linesPerFiber, by obsType.
LINES_PER_FIBER_MARGINS = {"arc": (0.25, 0.5), "trace": (0.025, 0.05)}
#: medDxCenter limits, the task config's: physical, not derived.
DX_CENTER = (1.0, 2.0)
#: Fewer known-good values than this and a population is not judged (one night of the m arm).
MIN_SAMPLES = 10


def derive(metrics: pd.DataFrame, path: Path = SOURCE) -> Path:
    """Derive the medFwhm thresholds from the known-good IIS frames and write them to ``path``."""
    table = calibrate(metrics, loadValidationVisits(), metricNames=["medFwhm"], derivedOn=DERIVED_ON)
    table["illumination"] = ILLUMINATION
    table["groupBy"] = table["groupBy"].map(lambda by: f"{by}/illumination" if by else "illumination")
    return writeThresholds(
        table,
        path,
        collection=COLLECTION,
        referenceRuns=[30],
        drpQaVersion=DRP_QA_VERSION,
        derivedOn=DERIVED_ON,
        pipelineVersions=PIPELINE,
    )


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
    adopted = f"Adopted {DERIVED_ON.isoformat()} (PIPE2D-1954)."
    entries = []
    for entry in document["thresholds"]:
        entry = dict(entry)
        population = entry["population"]
        assert population.get("illumination") == ILLUMINATION, population
        if entry["metric"] != "medFwhm":
            continue
        if entry["nGood"] < MIN_SAMPLES:
            entry |= {"warn": None, "fail": None}
            entry["provenance"] = (
                f"Not judged ({DERIVED_ON.isoformat()}, PIPE2D-1954): {entry['nGood']} known-good values, from "
                f"one night of the m arm. Derived values were: {entry['provenance']}"
            )
        elif (
            entry["method"] == "tolerance"
            and population.get("arm") == "n"
            and population.get("species") == "Ne"
        ):
            entry |= {"warn": None, "fail": None}
            entry["provenance"] = (
                f"Not judged ({DERIVED_ON.isoformat()}, PIPE2D-1954): the IIS Ne lamp gives about 15 faint lines "
                "per engineering fiber on the n arm, and the median width scatters 2-4 % night to night, "
                "crossing a 3 % WARN on 1-3 of the 14 known-good nights per camera; the Kr arc judges the n "
                f"arm's focus nightly. Derived values were: {entry['provenance']}"
            )
        else:
            entry["provenance"] = f"{adopted} {entry['provenance']}"
        entries.append(entry)
    entries.extend(linesPerFiberEntries(metrics))
    entries.append(
        {
            "metric": "medDxCenter",
            "population": {"illumination": ILLUMINATION},
            "higherIsWorse": True,
            "absolute": True,
            "method": "config",
            "warn": DX_CENTER[0],
            "fail": DX_CENTER[1],
            "provenance": (
                f"{adopted} The task config's limits, 1 px WARN and 2 px FAIL: a flexure limit is physical, "
                "not a population property, and an IIS frame is judged only by entries naming it. The 14 "
                "known-good nights sit within 0.56 px (the b arm read with m, PIPE2D-1927); the known-bad "
                "148829 n4, reduced against the sky line list, at 1.98 px."
            ),
        }
    )
    entries.append(
        {
            "metric": "pctFlagged",
            "population": {"illumination": ILLUMINATION},
            "higherIsWorse": True,
            "absolute": False,
            "method": "percentile",
            "warn": None,
            "fail": None,
            "provenance": (
                f"Not judged ({DERIVED_ON.isoformat()}, PIPE2D-1954): 54-96 % of the IIS lamps' lines are flagged "
                "on every known-good night (b Ne 96 %, n Ne 88 %, r Ne 54 %), a property of the line lists and "
                "the fibers rather than of the exposure."
            ),
        }
    )
    return {key: value for key, value in document.items() if key != "thresholds"} | {"thresholds": entries}


def linesPerFiberEntries(metrics: pd.DataFrame) -> list[dict]:
    """Return the linesPerFiber entries: a tolerance below each arm and sequence's known-good median."""
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
            derivedOn=DERIVED_ON,
        )
        entry = {
            "metric": "linesPerFiber",
            "population": {"arm": arm, "seqName": seqName, "illumination": ILLUMINATION},
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
            "provenance": f"Adopted {DERIVED_ON.isoformat()} (PIPE2D-1954). {suggestion.provenance}",
        }
        if suggestion.numSamples < MIN_SAMPLES:
            entry |= {"warn": None, "fail": None}
            entry["provenance"] = (
                f"Not judged ({DERIVED_ON.isoformat()}, PIPE2D-1954): {suggestion.numSamples} known-good values, "
                f"from one night of the m arm. Derived values were: {suggestion.provenance}"
            )
        entries.append(entry)
    return entries


def main() -> None:
    metrics = pd.read_parquet(METRICS)
    source = derive(metrics)
    print(f"Wrote {source}")
    document = adopt(yaml.safe_load(source.read_text()), metrics)
    header = f"# Made by {Path(__file__).name} from {SOURCE.relative_to(ROOT)}; see its docstring.\n"
    OUTPUT.write_text(header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=110))
    print(f"Wrote {OUTPUT}: {len(document['thresholds'])} entries")
    readThresholds(OUTPUT)  # it reads back


if __name__ == "__main__":
    main()
