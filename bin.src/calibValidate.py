#!/usr/bin/env python3
r"""Validate candidate detectorMap calibrations against each other on the same visits (PIPE2D-1941).

``run`` reduces the chosen visits once per candidate, with the candidate's collection first in the
inputs so its ``detectorMap_calib`` is the one used, through ``pipelines/qaThresholds.yaml``
(reduceExposure, imageQualityQa, dmLineResiduals) with drpActor's arc/trace configuration, into
``<prefix>/<label>``. ``report`` reads each candidate's ``dmLineResidualMetrics`` from the Butler
(or ``--exports`` directories, one per candidate) and writes the Markdown comparison of
`pfs.drp.qa.comparison.calibValidate` with a figure per detector.

    python bin.src/calibValidate.py run --period run31 --data-dir ~/qa-comparison \\
        --candidate asis= --candidate pick=u/wtg/pipe2d-1941/detectorMaps-pick \\
        --detectors b1 r2 --prefix u/wtg/pipe2d-1941/validate
    python bin.src/calibValidate.py report ... -o validate-run31.md

The visits are the period's set-cadence arcs and traces plus its daily arcs and traces
(the nightly Arc: Neon and Trace), or ``--visits FIRST LAST``.
"""

import argparse
import datetime
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pfs.drp.qa.comparison.calibInputs import setVisits
from pfs.drp.qa.comparison.calibValidate import buildValidationReport, compareCalibrations
from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.findings import lampsOf
from pfs.drp.qa.comparison.reduction import drpActorConfig
from pfs.drp.qa.comparison.runs import loadPeriods


def chooseVisits(args) -> pd.DataFrame:
    periods = loadPeriods(args.runs)
    if args.period not in periods:
        raise SystemExit(f"unknown period {args.period}; known: {', '.join(periods)}")
    period = periods[args.period]
    listingPath = Path(args.data_dir) / "opdb" / f"comparison-{period.name}-listing.parquet"
    if not listingPath.exists():
        raise SystemExit(f"{listingPath} not found: run qaComparison.py fetch --period {period.name} first")
    classified = classifyVisits(pd.read_parquet(listingPath), periods.values())
    if args.visits:
        first, last = args.visits
        chosen = classified[(classified["pfs_visit_id"] >= first) & (classified["pfs_visit_id"] <= last)]
    else:
        chosen = classified[classified["cadence"].isin(["set", "daily"])]
    chosen = chosen[chosen["sequence_type"].isin(["scienceArc", "scienceTrace"])]
    # Not the engineering-fiber lamps: they test nothing the science fibers' calibration is built from.
    chosen = chosen[~chosen["cmd_str"].map(lambda c: "(IIS)" in " ".join(lampsOf(c)))]
    # Not a visit the Butler lacks raw or pfsConfig for: pipetask refuses the whole pass over it.
    coverages = sorted(
        (Path(args.data_dir) / period.name / "qa").glob(f"*/coverage-{period.name}.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    if coverages:
        coverage = pd.read_parquet(coverages[-1])
        # Only the requested detectors: n4 lacking a raw image is no reason to drop b1's visit.
        ofDetectors = (coverage["arm"].astype(str) + coverage["spectrograph"].astype(str)).isin(
            args.detectors
        )
        coverage = coverage[ofDetectors]
        unusable = set(coverage.loc[coverage["status"].isin(["no pfsConfig", "no raw"]), "visit"].astype(int))
        dropped = sorted(set(chosen["pfs_visit_id"].astype(int)) & unusable)
        if dropped:
            print(f"left out (no raw or pfsConfig in the Butler): {dropped}", flush=True)
        chosen = chosen[~chosen["pfs_visit_id"].isin(unusable)]
    if chosen.empty:
        raise SystemExit("no arcs or traces among the chosen visits")
    return chosen


def candidates(args) -> dict[str, str]:
    out = {}
    for item in args.candidate:
        label, _, collection = item.partition("=")
        out[label] = collection
    return out


def runCommand(args, label: str, collection: str, visits: pd.DataFrame, detectors: list[str]) -> list[str]:
    inputs = ([collection] if collection else []) + list(args.inputs)
    where = " OR ".join(f"(arm = '{d[0]}' AND spectrograph = {d[1:]})" for d in detectors)
    visitList = ", ".join(str(int(v)) for v in sorted(visits["pfs_visit_id"]))
    config = drpActorConfig("scienceArc")
    command = [
        "pipetask",
        "--long-log",
        "--log-level",
        "pfs=INFO",
        "run",
        "-j",
        str(args.jobs),
        "-b",
        args.butler,
        "-p",
        args.pipeline,
        "-i",
        ",".join(inputs),
        "-o",
        f"{args.prefix}/{label}",
        "-d",
        f"instrument = 'PFS' AND ({where}) AND visit IN ({visitList})",
    ]
    for key, value in config.items():
        command += ["-c", f"{key}={value}"]
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--period", required=True)
    parser.add_argument("--data-dir", required=True, help="comparison mode's data directory")
    parser.add_argument("--runs", default=None)
    parser.add_argument("--visits", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=COLLECTION",
        help="a candidate: its collection first in the inputs; an empty collection means the inputs as they are",
    )
    parser.add_argument("--detectors", nargs="+", required=True, help="e.g. b1 r2")
    parser.add_argument("--butler", default="/work/datastore")
    parser.add_argument("--pipeline", default=None, help="default: drp_qa's pipelines/qaThresholds.yaml")
    parser.add_argument(
        "--inputs", nargs="+", default=["PFS/defaults"], help="the inputs after the candidate"
    )
    parser.add_argument("--prefix", required=True, help="output collections go to <prefix>/<label>")
    parser.add_argument(
        "--exports",
        nargs="*",
        default=None,
        help="report: exportMetrics.py directories, one per candidate in order",
    )
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-o", "--output", default=None, help="report: the Markdown file")
    args = parser.parse_args(argv)

    visits = chooseVisits(args)
    cands = candidates(args)
    shown = {
        label: f"`{coll}`" if coll else "the inputs as they are: " + ", ".join(f"`{i}`" for i in args.inputs)
        for label, coll in cands.items()
    }
    print(
        f"{len(visits)} visits {int(visits['pfs_visit_id'].min())}–{int(visits['pfs_visit_id'].max())}, "
        f"detectors {', '.join(args.detectors)}, candidates {', '.join(cands)}",
        flush=True,
    )
    if args.command == "run":
        if args.pipeline is None:
            args.pipeline = str(Path(__file__).resolve().parents[1] / "pipelines" / "qaThresholds.yaml")
        failed = []
        for label, collection in cands.items():
            command = runCommand(args, label, collection, visits, args.detectors)
            print(f"\n# {label}\n{shlex.join(command)}", flush=True)
            if args.dry_run:
                continue
            log = (
                Path(args.data_dir)
                / f"validate-{args.period}-{label}-{datetime.datetime.now():%Y%m%dT%H%M%S}.log"
            )
            print(f"running {label}; log: {log}", flush=True)
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            print(f"{label}: exit {result.returncode}", flush=True)
            if result.returncode:
                failed.append(label)
        if failed:
            print(f"failed: {', '.join(failed)}", file=sys.stderr, flush=True)
            return 1
        return 0

    if not args.output:
        raise SystemExit("report needs -o")
    metrics = {}
    if args.exports:
        for label, directory in zip(cands, args.exports, strict=True):
            files = sorted(Path(directory).glob("dmLineResidualMetrics-*.parquet"))
            if not files:
                raise SystemExit(f"no dmLineResidualMetrics-*.parquet in {directory}")
            metrics[label] = pd.read_parquet(files[-1])
    else:
        from lsst.daf.butler import Butler

        from pfs.drp.qa.metrics.readers import readMetrics

        butler = Butler(args.butler, writeable=False)
        for label in cands:
            metrics[label] = readMetrics(
                butler,
                [int(v) for v in visits["pfs_visit_id"]],
                datasetType="dmLineResidualMetrics",
                collections=[f"{args.prefix}/{label}"],
                progress=True,
            )
    detectors = set(args.detectors)
    chosenVisits = {int(v) for v in visits["pfs_visit_id"]}
    for label in metrics:
        m = metrics[label]
        m = m[(m["arm"].astype(str) + m["spectrograph"].astype(str)).isin(detectors)]
        metrics[label] = m[m["visit"].astype(int).isin(chosenVisits)]  # an export may hold more
    theSet = setVisits(visits[visits["cadence"] == "set"])
    table = compareCalibrations(metrics, theSet["visit"])
    out = Path(args.output)
    figures = {}
    import matplotlib

    matplotlib.use("Agg")
    from pfs.drp.qa.plotting.calibInputs import plotCalibrationComparison

    for det in args.detectors:
        sub = table[table["detector"] == det]
        if sub.empty:
            continue
        path = out.with_name(f"{out.stem}-{det}.png")
        plotCalibrationComparison(sub, det).savefig(path, dpi=130, bbox_inches="tight")
        figures[det] = path.name
    note = (
        f"Visits: {int(visits['pfs_visit_id'].min())}–{int(visits['pfs_visit_id'].max())}, "
        f"{len(theSet)} in the set and {len(visits) - len(theSet)} nightly arcs and traces after it."
    )
    out.write_text(
        buildValidationReport(
            table, f"{args.period} {', '.join(args.detectors)}: candidate calibrations", shown, figures, note
        )
    )
    table.to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
