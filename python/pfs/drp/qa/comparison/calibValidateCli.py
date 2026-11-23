r"""``drpQa.py calib validate``: validate candidate detectorMap calibrations against each other on the same visits.

``run`` reduces the chosen visits once per candidate, with the candidate's collection first in the
inputs so its ``detectorMap_calib`` is the one used, in two ``pipetask`` commands — ``reduceExposure``
with drpActor's arc/trace configuration, then ``dmLineResiduals`` on the reductions — into
``<prefix>/<label>``. ``report`` reads each candidate's ``dmLineResidualMetrics`` from the Butler and
writes the Markdown comparison of `pfs.drp.qa.comparison.calibValidate` with a figure per detector,
and a slit-offset section per candidate -- the fibers whose slit offset is an outlier in two or more
arms -- with the change from the previous run's calibration when ``--previous`` names it.

    drpQa.py calib validate run --period run31 --opdb-host pfsa-db \\
        --candidate asis= --candidate pick=u/wtg/pipe2d-1941/detectorMaps-pick \\
        --detectors b1 r2 --prefix u/wtg/pipe2d-1941/validate
    drpQa.py calib validate report ... -o validate-run31.md

The visits (the period's set-cadence arcs and traces plus its daily arcs and traces, or
``--visits FIRST LAST``) come from the opdb (``--opdb-host``).
"""

import argparse
import datetime
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pfs.drp.qa.comparison.calibInputs import lampsOf, setVisits
from pfs.drp.qa.comparison.calibValidate import (
    buildSlitOffsetSection,
    buildValidationReport,
    compareCalibrations,
    slitOffsetHealth,
)
from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.reduction import (
    MEASURE_PIPELINE,
    REDUCE_LABELS,
    REDUCE_PIPELINE,
    drpActorConfig,
)
from pfs.drp.qa.comparison.runs import loadPeriods


def chooseVisits(args) -> pd.DataFrame:
    periods = loadPeriods(args.runs)
    if args.period not in periods:
        raise SystemExit(f"unknown period {args.period}; known: {', '.join(periods)}")
    period = periods[args.period]
    from pfs.drp.qa.comparison import queries
    from pfs.utils.database.opdb import OpDB

    opdb = OpDB(host=args.opdb_host, port=args.opdb_port, dbname=args.opdb_dbname, user="public_user")
    listing = queries.readVisitListing(opdb, period.start, min(period.end, datetime.datetime.now()))
    classified = classifyVisits(listing, periods.values())
    if args.visits:
        first, last = args.visits
        chosen = classified[(classified["pfs_visit_id"] >= first) & (classified["pfs_visit_id"] <= last)]
    else:
        chosen = classified[classified["cadence"].isin(["set", "daily"])]
    chosen = chosen[chosen["sequence_type"].isin(["scienceArc", "scienceTrace"])]
    # Not the engineering-fiber lamps: they test nothing the science fibers' calibration is built from.
    chosen = chosen[~chosen["cmd_str"].map(lambda c: "(IIS)" in " ".join(lampsOf(c)))]
    if chosen.empty:
        raise SystemExit("no arcs or traces among the chosen visits")
    return chosen


def candidates(args) -> dict[str, str]:
    out = {}
    for item in args.candidate:
        label, _, collection = item.partition("=")
        out[label] = collection
    return out


def candidateCommands(
    args, label: str, collection: str, visits: pd.DataFrame, detectors: list[str]
) -> list[tuple[str, list[str]]]:
    """Return the ``reduce`` then ``measure`` ``pipetask`` commands for one candidate.

    The candidate's collection goes first in the inputs so its ``detectorMap_calib`` is the one
    used. The reduce phase runs ``reduceExposure`` with drpActor's arc configuration; the measure
    phase runs ``dmLineResiduals`` alone (its report reads only ``dmLineResidualMetrics``) on the
    reductions, which it finds in the output chain.
    """
    inputs = ([collection] if collection else []) + list(args.inputs)
    where = " OR ".join(f"(arm = '{d[0]}' AND spectrograph = {d[1:]})" for d in detectors)
    visitList = ", ".join(str(int(v)) for v in sorted(visits["pfs_visit_id"]))
    output = f"{args.prefix}/{label}"
    data = f"instrument = 'PFS' AND ({where}) AND visit IN ({visitList})"

    def base(subset: str, rebase: bool) -> list[str]:
        command = ["pipetask", "--long-log", "--log-level", "pfs=INFO", "run", "-j", str(args.jobs)]
        command += ["-b", args.butler, "-p", subset, "-i", ",".join(inputs), "-o", output]
        if rebase:
            command += ["--rebase"]
        return command

    reduce = base(f"{REDUCE_PIPELINE}#{','.join(REDUCE_LABELS)}", rebase=False)
    for key, value in drpActorConfig("scienceArc").items():
        reduce += ["-c", f"{key}={value}"]
    reduce += ["-d", data]
    # The reduce phase makes the output chain; the measure phase rebases onto it to read the reductions.
    measure = [*base(f"{MEASURE_PIPELINE}#dmLineResiduals", rebase=True), "-d", data]
    return [("reduce", reduce), ("measure", measure)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--period", required=True)
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
    parser.add_argument("--opdb-host", default="pfsa-db", help="opdb host, for the visit listing")
    parser.add_argument("--opdb-port", type=int, default=5432, help="opdb port")
    parser.add_argument("--opdb-dbname", default="opdb", help="opdb database name")
    parser.add_argument(
        "--inputs", nargs="+", default=["PFS/defaults"], help="the inputs after the candidate"
    )
    parser.add_argument("--prefix", required=True, help="output collections go to <prefix>/<label>")
    parser.add_argument(
        "--work-dir", default="~/.cache/drp_qa/validate", help="run: where the pipetask logs go"
    )
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-o", "--output", default=None, help="report: the Markdown file")
    parser.add_argument(
        "--previous",
        default=None,
        help="report: a collection with the previous run's detectorMap_calib, for the slit-offset change",
    )
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
        failed = []
        for label, collection in cands.items():
            for phase, command in candidateCommands(args, label, collection, visits, args.detectors):
                print(f"\n# {label} ({phase})\n{shlex.join(command)}", flush=True)
                if args.dry_run:
                    continue
                workDir = Path(args.work_dir).expanduser()
                workDir.mkdir(parents=True, exist_ok=True)
                log = (
                    workDir
                    / f"validate-{args.period}-{label}-{phase}-{datetime.datetime.now():%Y%m%dT%H%M%S}.log"
                )
                print(f"running {label} ({phase}); log: {log}", flush=True)
                with log.open("w") as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                print(f"{label} ({phase}): exit {result.returncode}", flush=True)
                if result.returncode:
                    failed.append(f"{label} ({phase})")
                    break  # a candidate whose reduction failed is not measured
        if failed:
            print(f"failed: {', '.join(failed)}", file=sys.stderr, flush=True)
            return 1
        return 0

    if not args.output:
        raise SystemExit("report needs -o")
    from lsst.daf.butler import Butler

    from pfs.drp.qa.metrics.readers import readMetrics

    butler = Butler(args.butler, writeable=False)
    metrics = {}
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
        metrics[label] = m[m["visit"].astype(int).isin(chosenVisits)]
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
    report = buildValidationReport(
        table, f"{args.period} {', '.join(args.detectors)}: candidate calibrations", shown, figures, note
    )

    # Slit-offset health of each candidate's detectorMap_calib, and the change from the previous run.
    from pfs.drp.qa.comparison.butlerQueries import readSlitOffsets
    from pfs.drp.qa.metrics.slitOffsets import crossArmConfirmed

    previousConfirmed = None
    if args.previous:
        previousConfirmed = crossArmConfirmed(slitOffsetHealth(readSlitOffsets(butler, args.previous)))
    for label in cands:
        flagged = slitOffsetHealth(readSlitOffsets(butler, f"{args.prefix}/{label}"))
        if flagged.empty:
            continue
        flagged.to_parquet(out.with_name(f"{out.stem}-{label}.slitoffsets.parquet"))
        heading = f" ({label})" if len(cands) > 1 else ""
        section = buildSlitOffsetSection(
            crossArmConfirmed(flagged), previousConfirmed, previousLabel=f"`{args.previous}`"
        )
        report += "\n" + section.replace("## Slit offsets", f"## Slit offsets{heading}", 1)

    out.write_text(report)
    table.to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {out}", flush=True)
    return 0
