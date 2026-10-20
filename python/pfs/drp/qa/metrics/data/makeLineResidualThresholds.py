"""Make ``dmLineResidualThresholds-run25.yaml``, the scatter thresholds ``dmLineResiduals`` uses by default.

The source is the file ``docs/qa-thresholds.ipynb`` derived from the validation
visits (collection ``u/wtg/qa-thresholds/010``, PIPE2D-1938: every validation
visit reduced afresh with the pipeline the file records), kept in the tests as
``tests/metrics/data/dmLineResidualThresholds-run25.yaml`` beside the metrics it
was derived from: the robust RMS of the lines the visit's detectorMap fit held
back (``RESERVED``), against that adjusted map, as percentiles per arm, line
species, camera set and observation type, and as a tolerance per detector,
species and observation type (`pfs.drp.qa.metrics.registry`, PIPE2D-1956). The
calibration's own input visits stay in: they sit at zero against the
calibration by construction, which is the drift's concern, while the scatter is
measured against the visit's own map on lines held back from its fit, where a
calibration input is a visit like any other. Adopting it:

- **Trace centroids on quartz frames are judged per detector** (``obsType``
  ``trace``), by the tolerance: WARN 20 % and FAIL 50 % above the detector's
  known-good median, where the derivation has the three values a median needs.
  A detector's quartz trace scatter repeats to about 1 % within a run and
  differs 20 % between detectors (Run25 b1-b4: 0.040, 0.044, 0.042, 0.046 px),
  so a pooled percentile is set by the worst detector and cuts through the best
  one's cluster; the regular trace sets of Runs 26-30 sit within +12 % of Run25's
  per-detector median (+24 % at the end of Run26 on b1), the Run27 SM1 defocus at
  1.7-4x.
- **Lamp-line species are judged per detector in x, by the same tolerance**, the
  mercury and cadmium lines (`HGCD_SPECIES`) aside. Not in y: the HgCd species'
  y scatter is intrinsically variable (medians 0.6-1.1 px on n and r) and a
  tolerance on it failed 5-20 % of Runs 25-28's healthy arcs; the other species'
  y waits for a check of its own. A fixed percentile over the run's arcs does not
  transfer between runs (thresholds from Run29's arcs flagged 20-70 % of Runs
  25-28's healthy arcs), the per-detector tolerance does.
- **Trace centroids on arcs and twilights are not judged.** The continuum is
  faint there, and the centroids scatter two to three times more than on a
  quartz (Runs 25-30 medians 0.10-0.13 px against 0.02-0.04): one population
  with the quartz traces, they failed 14-20 % of Runs 25-27's healthy arcs.
- **Sky-line species are not judged**: OH, O2, OI and NaI on the twilights give
  scatter up to several pixels on the b arm, which is the sky lines' fit, not the
  instrument's state; judging science frames is later work (PIPE2D-1939 for
  the n arm, PIPE2D-1923 for the rollup).
- **Nothing below the sample floor is judged**: 20 values for a percentile, 3
  for a tolerance.

A population judged by the tolerance keeps its tolerance entries only (an
equally specific percentile entry beside them would be an error in the gate);
one that is not keeps its percentile entries, unjudged, with the reason. The
derivation keeps every row. The task's config fields still judge the median
offsets from ``detectorMap_calib`` (1 and 2 px); this file holds only the
scatter entries, so the two never overlap. ``bin.src/judgeScatterThresholds.py``
is the acceptance check: the adopted file must not fail a run's healthy frames
in bulk.

Run from the repository root:

    PYTHONPATH=python python python/pfs/drp/qa/metrics/data/makeLineResidualThresholds.py
"""

from pathlib import Path

import yaml

from pfs.drp.qa.metrics.lineResiduals import QUARTZ, SKY_SPECIES, TRACE

ROOT = Path(__file__).parents[6]
SOURCE = ROOT / "tests" / "metrics" / "data" / "dmLineResidualThresholds-run25.yaml"
OUTPUT = Path(__file__).parent / "dmLineResidualThresholds-run25.yaml"

#: The mercury and cadmium lines of the HgCd lamp: scatter in y too variable for a tolerance.
HGCD_SPECIES = ("HgI", "CdI")
ADOPTED = "2026-10-20"


def judgedByTolerance(population: dict) -> bool:
    """Return whether ``population`` (an entry's) is judged per detector by the tolerance."""
    description, obsType = population.get("description"), population.get("obsType")
    if description == TRACE:
        return obsType == QUARTZ
    return description not in SKY_SPECIES and description not in HGCD_SPECIES


def adopt(document: dict) -> dict:
    """Return the derived scatter thresholds ``document`` as adopted.

    Parameters
    ----------
    document : `dict`
        The derived thresholds file, as YAML.

    Returns
    -------
    `dict`
        The adopted thresholds file, as YAML: the tolerance entries of the
        populations judged per detector and the percentile entries of the
        others, those not judged with ``warn`` and ``fail`` null and the
        derived values in the provenance.
    """
    entries = []
    for entry in document["thresholds"]:
        entry = dict(entry)
        population = entry["population"]
        description, obsType = population.get("description"), population.get("obsType")
        tolerance = entry.get("method") == "tolerance"
        if tolerance != judgedByTolerance(population):
            continue
        reason = None
        if description in SKY_SPECIES:
            reason = (
                "a sky-line species on twilight frames, whose scatter is the sky lines' fit rather than the "
                "instrument's state"
            )
        elif description == TRACE and obsType != QUARTZ:
            reason = (
                f"trace centroids on {obsType} frames, where the continuum is faint and they scatter two to "
                "three times more than on a quartz; judged on quartz frames only"
            )
        elif description in HGCD_SPECIES:
            reason = (
                "a mercury or cadmium line, whose scatter in y is intrinsically variable (medians 0.6-1.1 px on "
                "the n and r arms): a per-detector tolerance failed 5-20 % of Runs 25-28's healthy arcs"
            )
        elif entry["metric"] == "robustRmsDeltaY":
            reason = (
                "the y scatter of a lamp species, not yet checked against later runs on its own; the x scatter "
                "is judged"
            )
        elif not entry.get("reliable"):
            floor = 3 if tolerance else 20
            reason = (
                f"the derivation has n={entry.get('nGood')} from {entry.get('nGoodVisits')} visits, below the "
                f"{floor}-sample floor"
            )
        if reason is not None:
            entry |= {"warn": None, "fail": None}
            entry["provenance"] = (
                f"Not judged ({ADOPTED}, PIPE2D-1956): {reason}. Derived values were: {entry['provenance']}"
            )
        else:
            entry["provenance"] = f"Adopted {ADOPTED} (PIPE2D-1956). {entry['provenance']}"
        entries.append(entry)
    return {key: value for key, value in document.items() if key != "thresholds"} | {"thresholds": entries}


def main() -> None:
    document = adopt(yaml.safe_load(SOURCE.read_text()))
    header = f"# Made by {Path(__file__).name} from {SOURCE.relative_to(ROOT)}; see its docstring.\n"
    OUTPUT.write_text(header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=110))
    judged = sum(entry["warn"] is not None or entry["fail"] is not None for entry in document["thresholds"])
    print(f"Wrote {OUTPUT}: {judged} of {len(document['thresholds'])} entries judged")


if __name__ == "__main__":
    main()
