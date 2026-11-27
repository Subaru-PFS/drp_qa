r"""``drpQa.py calib profiles``: measure candidate fiberProfiles calibrations before one is certified.

``run`` measures each candidate with ``fiberProfilesQa``, the candidate's collection first in the inputs
so its ``fiberProfiles`` is the one resolved, into ``<prefix>/<label>``. A candidate with an empty
collection is the inputs as they are: the calibration in force. ``report`` reads each candidate's
``fiberProfilesQaFibers`` and ``fiberProfilesQaMetrics`` from the Butler and writes the Markdown report
of `pfs.drp.qa.comparison.calibProfiles`, with a figure per candidate: what to do before certifying, the
fibers whose wing light stands above their neighbours' (measured ones and copies apart), and each
detector's numbers. With ``--previous`` the candidate is set against the calibration in force, which
then gets no section of its own.

    drpQa.py calib profiles run --visit 156021 155123 --prefix u/wtg/profiles/run31 \\
        --candidate asis= --candidate new=u/wtg/run31/fiberProfilesGen
    drpQa.py calib profiles report --visit 156021 155123 --prefix u/wtg/profiles/run31 \\
        --candidate asis= --candidate new=u/wtg/run31/fiberProfilesGen --previous asis -o profiles-run31.md

A candidate is ``LABEL=COLLECTION``. The label is a name of your choosing: it names the candidate in the
report and its output collection, ``<prefix>/<label>``. That collection is defined with the candidate's
collection first, so a label is tied to the collection it was first run with: measure a rebuilt candidate
under a new label, or a new ``--prefix``.

``--visit`` names visits of the run the candidates are for, of any frame type: each gives the task its
fiber configuration, the date a calibration in force is resolved at, and the detectors it exposed. The
medium-resolution profiles are measured on a visit taken with the red arm in medium resolution, so name
one of each kind to measure all four arms. Nothing of a visit is reduced.
"""

import argparse
import datetime
import shlex
import subprocess
import sys
import time
from pathlib import Path

from pfs.drp.qa.comparison.calibProfiles import buildProfileReport, calibrationSources

__all__ = ["candidateCommand", "candidates", "main"]

#: A candidate's RUNs are told from the inputs after it by name: they are the collection or sit under it, as
#: the RUNs of a pipetask output chain do.

#: The pipeline that measures a candidate, and its one task.
PIPELINE = "$DRP_QA_DIR/pipelines/fiberProfilesQa.yaml"
LABEL = "fiberProfilesQa"


def candidates(args) -> dict[str, str]:
    """Return each candidate's collection by label; an empty collection is the inputs as they are.

    Raises
    ------
    SystemExit
        An item has no ``=`` or no label, or a label is given twice.
    """
    out = {}
    for item in args.candidate:
        label, equals, collection = item.partition("=")
        if not equals or not label:
            # Without the "=" the whole item would be a label with no collection: the inputs as they are.
            raise SystemExit(
                f"--candidate {item}: give LABEL=COLLECTION, or LABEL= for the inputs as they are (asis=)"
            )
        if label in out:
            raise SystemExit(f"--candidate {item}: the label {label} is given twice")
        out[label] = collection
    return out


def candidateCommand(args, label: str, collection: str) -> list[str]:
    """Return the ``pipetask`` command that measures one candidate.

    The candidate's collection goes first in the inputs so its ``fiberProfiles`` is the one resolved,
    certified or not. ``anyFrameType`` lets the visit be of any type: it only anchors the quanta.
    """
    inputs = ([collection] if collection else []) + list(args.inputs)
    data = f"instrument = 'PFS' AND visit IN ({', '.join(str(int(visit)) for visit in args.visit)})"
    if args.detectors:
        where = " OR ".join(f"(arm = '{d[0]}' AND spectrograph = {int(d[1:])})" for d in args.detectors)
        data += f" AND ({where})"
    command = ["pipetask", "--long-log", "--log-level", "pfs=INFO", "run", "-j", str(args.jobs)]
    command += ["--register-dataset-types", "-b", args.butler, "-p", PIPELINE]
    command += ["-i", ",".join(inputs), "-o", f"{args.prefix}/{label}"]
    command += ["-c", f"{LABEL}:anyFrameType=True", "-d", data]
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument(
        "--visit",
        type=int,
        nargs="+",
        required=True,
        help=(
            "visits of the run the candidates are for, of any frame type; each gives the detectors it exposed, so"
            " name one with the red arm in low resolution and one in medium to measure all four arms"
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=COLLECTION",
        help=(
            "a candidate: its collection first in the inputs; an empty collection (asis=) means the inputs as"
            " they are. The label is yours to choose and names the output collection, <prefix>/<label>: give a"
            " rebuilt candidate a new label. Give the collection the build's RUNs sit under, as pipetask names"
            " them"
        ),
    )
    parser.add_argument("--prefix", required=True, help="output collections go to <prefix>/<label>")
    parser.add_argument("--detectors", nargs="+", default=None, help="e.g. b1 r2; default every detector")
    parser.add_argument("--butler", default="/work/datastore")
    parser.add_argument(
        "--inputs", nargs="+", default=["PFS/defaults"], help="the inputs after the candidate"
    )
    parser.add_argument(
        "--work-dir", default="~/.cache/drp_qa/profiles", help="run: where the pipetask logs go"
    )
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-o", "--output", default=None, help="report: the Markdown file")
    parser.add_argument(
        "--previous",
        default=None,
        help="report: the label of the calibration in force, to set the others against",
    )
    args = parser.parse_args(argv)

    cands = candidates(args)
    shown = {
        label: f"`{coll}`" if coll else "the inputs as they are: " + ", ".join(f"`{i}`" for i in args.inputs)
        for label, coll in cands.items()
    }
    if args.previous is not None and args.previous not in cands:
        raise SystemExit(f"--previous {args.previous} is not one of the candidates: {', '.join(cands)}")
    visits = ", ".join(str(visit) for visit in args.visit)
    print(f"visits {visits}, candidates {', '.join(cands)}", flush=True)

    if args.command == "run":
        failed = []
        started = time.time()
        for number, (label, collection) in enumerate(cands.items(), start=1):
            command = candidateCommand(args, label, collection)
            print(f"\n# {label} ({number} of {len(cands)})\n{shlex.join(command)}", flush=True)
            if args.dry_run:
                continue
            workDir = Path(args.work_dir).expanduser()
            workDir.mkdir(parents=True, exist_ok=True)
            log = workDir / f"profiles-{args.visit[0]}-{label}-{datetime.datetime.now():%Y%m%dT%H%M%S}.log"
            print(f"running {label}; log: {log}", flush=True)
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            print(
                f"{label}: exit {result.returncode}; {number} of {len(cands)} done, "
                f"{time.time() - started:.0f} s elapsed",
                flush=True,
            )
            if result.returncode:
                failed.append(label)
        if failed:
            print(f"failed: {', '.join(failed)}", file=sys.stderr, flush=True)
            return 1
        return 0

    if not args.output:
        raise SystemExit("report needs -o")
    from lsst.daf.butler import Butler

    from pfs.drp.qa.comparison.butlerQueries import collectionExists, readProfileQa

    butler = Butler(args.butler, writeable=False)
    named = {visit: place for place, visit in enumerate(args.visit)}
    tables = {}
    for label in cands:
        if not collectionExists(butler, f"{args.prefix}/{label}"):
            raise SystemExit(f"{args.prefix}/{label} does not exist: run first")
        fibers, metrics = readProfileQa(butler, f"{args.prefix}/{label}", args.visit)
        if not fibers.empty:
            # A detector exposed in several of the visits was measured once for each: keep the first named.
            fibers = fibers.sort_values("visit", key=lambda column: column.map(named), kind="stable")
            metrics = metrics.sort_values("visit", key=lambda column: column.map(named), kind="stable")
            fibers = fibers.drop_duplicates(["arm", "spectrograph", "fiberId"])
            metrics = metrics.drop_duplicates(["arm", "spectrograph", "metric"])
        if fibers.empty:
            raise SystemExit(
                f"{args.prefix}/{label} holds no fiberProfilesQaFibers for visits {visits}: run first"
            )
        if args.detectors:
            keep = set(args.detectors)
            fibers = fibers[(fibers["arm"].astype(str) + fibers["spectrograph"].astype(str)).isin(keep)]
            metrics = metrics[(metrics["arm"].astype(str) + metrics["spectrograph"].astype(str)).isin(keep)]
        tables[label] = (fibers, metrics)
        print(
            f"{label}: {fibers.groupby(['arm', 'spectrograph']).ngroups} detectors, {len(fibers)} profiles",
            flush=True,
        )
        _, elsewhere = calibrationSources(fibers, cands[label])
        if elsewhere:
            print(
                f"WARNING: {label}: {' '.join(elsewhere)} were not read from the candidate's collection; the "
                "inputs after it were measured instead",
                flush=True,
            )
    out = Path(args.output)
    import matplotlib

    matplotlib.use("Agg")
    from pfs.drp.qa.plotting.fiberProfiles import plotWingExcess

    before = tables[args.previous][0] if args.previous is not None else None
    figures = {}
    for label, (fibers, _) in tables.items():
        if label == args.previous:
            continue
        path = out.with_name(f"{out.stem}-{label}.png")
        figure = plotWingExcess(
            fibers,
            before,
            title=f"{label}: wing light above the neighbouring fibers",
            previousLabel=args.previous or "in force",
        )
        figure.savefig(path, dpi=130, bbox_inches="tight")
        figures[label] = path.name
    report = buildProfileReport(
        tables,
        shown,
        "fiberProfiles before certification",
        previous=args.previous,
        note=f"Resolved for visits {visits}.",
        collections=cands,
        figures=figures,
    )
    out.write_text(report)
    for label, (fibers, _) in tables.items():
        fibers.to_parquet(out.with_name(f"{out.stem}-{label}.fibers.parquet"))
    print(f"wrote {out}", flush=True)
    return 0
