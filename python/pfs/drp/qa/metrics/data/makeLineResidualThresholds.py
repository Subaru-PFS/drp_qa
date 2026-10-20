"""Make ``dmLineResidualThresholds-run25.yaml``, the scatter thresholds ``dmLineResiduals`` uses by default.

The source is the file ``docs/qa-thresholds.ipynb`` derived from the validation
visits (collection ``u/wtg/qa-thresholds/010``, PIPE2D-1938: every validation
visit reduced afresh with the pipeline the file records), kept in the tests as
``tests/metrics/data/dmLineResidualThresholds-run25.yaml`` beside the metrics it
was derived from: the robust RMS of the lines the visit's detectorMap fit held
back (``RESERVED``), against that adjusted map, per arm, line species, camera
set and observation type (PIPE2D-1956), each calibration's own input visits left
out. Adopting it:

- **Trace centroids on quartz frames are judged** (``obsType`` ``trace``), WARN
  at p95 and FAIL at p99 as derived, where the derivation is reliable (20 or
  more known-good values); with fewer than `MIN_VISITS` visits behind them, at
  WARN only, since a p99 FAIL would then sit at the sample maximum.
- **Trace centroids on arcs and twilights are not judged.** The continuum is
  faint there, and the centroids scatter two to three times more than on a
  quartz (Runs 25-30 medians 0.10-0.13 px against 0.02-0.04): one population
  with the quartz traces, they failed 14-20 % of Runs 25-27's healthy arcs.
- **Lamp-line species are not judged.** Their scatter differs between observing
  runs by more than it varies within one: thresholds from one run's arcs (Run29's
  2026-07-14 and 07-23 sets, 24-28 values each) flag 20-70 % of Runs 25-28's
  healthy arcs and about none of Runs 29-30's. A fixed reference does not
  transfer; a within-run baseline is later work on PIPE2D-1956. Stored, with the
  derived values in the provenance.
- **Sky-line species are not judged**: OH, O2, OI and NaI on the twilights give
  scatter up to several pixels on the b arm, which is the sky lines' fit, not the
  instrument's state; judging science frames is later work (PIPE2D-1939 for
  the n arm, PIPE2D-1923 for the rollup).
- **Nothing below the 20-sample floor is judged.**

The task's config fields still judge the median offsets from ``detectorMap_calib``
(1 and 2 px); this file holds only the scatter entries, so the two never overlap.
``bin.src/judgeScatterThresholds.py`` is the acceptance check: the adopted file
must not fail a run's healthy frames in bulk.

Run from the repository root:

    PYTHONPATH=python python python/pfs/drp/qa/metrics/data/makeLineResidualThresholds.py
"""

from pathlib import Path

import yaml

from pfs.drp.qa.metrics.lineResiduals import QUARTZ, SKY_SPECIES, TRACE

ROOT = Path(__file__).parents[6]
SOURCE = ROOT / "tests" / "metrics" / "data" / "dmLineResidualThresholds-run25.yaml"
OUTPUT = Path(__file__).parent / "dmLineResidualThresholds-run25.yaml"

#: Fewest known-good visits for a FAIL threshold: on 20 values from 5, a p99 sits at the sample maximum.
MIN_VISITS = 10
ADOPTED = "2026-10-20"


def adopt(document: dict) -> dict:
    """Return the derived scatter thresholds ``document`` as adopted.

    Parameters
    ----------
    document : `dict`
        The derived thresholds file, as YAML.

    Returns
    -------
    `dict`
        The adopted thresholds file, as YAML: the same entries, those not judged
        with ``warn`` and ``fail`` null and the derived values in the provenance.
    """
    entries = []
    for entry in document["thresholds"]:
        entry = dict(entry)
        population = entry["population"]
        description, obsType = population.get("description"), population.get("obsType")
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
        elif description != TRACE:
            reason = (
                "a lamp species, whose scatter differs between observing runs by more than it varies within "
                "one, so no fixed reference transfers (thresholds from Run29's arcs flag 20-70 % of Runs 25-28's "
                "healthy arcs)"
            )
        elif not entry.get("reliable"):
            reason = f"the derivation has n={entry.get('nGood')} from {entry.get('nGoodVisits')} visits, below the 20-sample floor"
        if reason is not None:
            entry |= {"warn": None, "fail": None}
            entry["provenance"] = (
                f"Not judged ({ADOPTED}, PIPE2D-1956): {reason}. Derived values were: {entry['provenance']}"
            )
        elif (entry.get("nGoodVisits") or 0) < MIN_VISITS:
            entry["fail"] = None
            entry["provenance"] = (
                f"Adopted at WARN only ({ADOPTED}, PIPE2D-1956): n={entry.get('nGood')} from "
                f"{entry.get('nGoodVisits')} visits, so a p99 FAIL would sit at the sample maximum and flag one "
                f"known-good image in twenty. Derived values were: {entry['provenance']}"
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
