"""Make ``dmLineResidualThresholds-run25.yaml``, the scatter thresholds ``dmLineResiduals`` uses by default.

The source is the file ``docs/qa-thresholds.ipynb`` derived from the validation
visits (collection ``u/wtg/qa-thresholds/010``, PIPE2D-1938: every validation
visit reduced afresh with the pipeline the file records), kept in the tests as
``tests/metrics/data/dmLineResidualThresholds-run25.yaml`` beside the metrics it
was derived from: the robust RMS of the lines the visit's detectorMap fit held
back (``RESERVED``), against that adjusted map, per arm, line species and camera
set (PIPE2D-1956), each calibration's own input visits left out. Adopting it:

- **Judged where the derivation is reliable**, 20 or more known-good values:
  every arm's trace centroids in x (40-116 values from 10-35 visits), WARN at
  p95 and FAIL at p99 as derived; and the b arm's Krypton, Neon and Xenon lines
  read with the m camera set, in x and y, at WARN only. Those three have their
  20 values from 5 visits (`MIN_VISITS`), so a p99 FAIL would sit at the sample
  maximum and flag one known-good image in twenty; a crossing shows as WARN
  until a second reference run bounds the tail.
- **Not judged below the floor**: the other lamp species have 8-12 values from
  2-3 Run25 visits, too few for a tail percentile; Run29's arcs are not a
  reference. Stored, with the derived values in the provenance.
- **Sky-line species are not judged**: OH, O2, OI and NaI on the twilights give
  scatter up to several pixels on the b arm, which is the sky lines' fit, not the
  instrument's state; judging science frames is later work (PIPE2D-1939 for
  the n arm, PIPE2D-1923 for the rollup).

The task's config fields still judge the median offsets from ``detectorMap_calib``
(1 and 2 px); this file holds only the scatter entries, so the two never overlap.

Run from the repository root:

    PYTHONPATH=python python python/pfs/drp/qa/metrics/data/makeLineResidualThresholds.py
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[6]
SOURCE = ROOT / "tests" / "metrics" / "data" / "dmLineResidualThresholds-run25.yaml"
OUTPUT = Path(__file__).parent / "dmLineResidualThresholds-run25.yaml"

#: Line species of the night sky, measured on twilights: not judged here.
SKY_SPECIES = ("OH", "O2", "OI", "NaI")
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
        if population.get("description") in SKY_SPECIES:
            entry |= {"warn": None, "fail": None}
            entry["provenance"] = (
                f"Not judged ({ADOPTED}, PIPE2D-1956): a sky-line species on twilight frames, whose scatter "
                f"is the sky lines' fit rather than the instrument's state. Derived values were: {entry['provenance']}"
            )
        elif not entry.get("reliable"):
            entry |= {"warn": None, "fail": None}
            entry["provenance"] = (
                f"Not judged ({ADOPTED}, PIPE2D-1956): the derivation has n={entry.get('nGood')} from "
                f"{entry.get('nGoodVisits')} visits, below the 20-sample floor; Run29's arcs are not a reference. "
                f"Derived values were: {entry['provenance']}"
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
