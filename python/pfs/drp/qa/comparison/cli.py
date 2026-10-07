"""Command line of comparison mode: ``bin.src/qaComparison.py``.

Four steps, each safe to repeat:

``fetch``
    Read a period's visits and what describes them from the opdb, read-only, into parquet.
``plan``
    Classify the visits, ask the Butler what it holds, and print the coverage and the
    ``pipetask`` commands that would complete it. Writes nothing to the Butler.
``run``
    Plan, then run those commands.
``report``
    Read the verdicts and write the period's report.

Files go under ``--data-dir``: the opdb reads in ``opdb/``, and everything for one drp_qa
version in ``<run>/qa/<version>/``. A pre-run period's files sit with its run's.
"""

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pfs.drp.qa.comparison.classify import SKY_TYPES, classifyVisits, designKinds
from pfs.drp.qa.comparison.plan import (
    STATUS_ORDER,
    coverage,
    outputCollection,
    passes,
    pipetaskCommand,
    summarize,
)
from pfs.drp.qa.comparison.runs import Period, loadPeriods

__all__ = ["drpQaVersion", "fetch", "loadFetched", "main"]

_FETCHED = ("listing", "notes", "telStatus", "designs")


def fetch(opdb, period: Period, dataDir: Path, now: datetime.datetime | None = None) -> dict[str, Path]:
    """Read a period's visits, notes, telescope status and design categories from the opdb.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The opdb, read-only.
    period : `Period`
        The period.
    dataDir : `pathlib.Path`
        Where to write, under ``opdb/``.
    now : `datetime.datetime`, optional
        The current time (HST); a period still under way is read up to it.

    Returns
    -------
    `dict` [`str`, `pathlib.Path`]
        The files written, by name.
    """
    from pfs.drp.qa.comparison import queries

    now = now or datetime.datetime.now()
    end = min(period.end, now)
    listing = queries.readVisitListing(opdb, period.start, end)
    sky = listing[listing["sequence_type"].isin(SKY_TYPES)]
    frames = {
        "listing": listing,
        "notes": queries.readNotes(opdb, listing["pfs_visit_id"], listing["iic_sequence_id"].dropna()),
        "telStatus": queries.readTelStatus(opdb, sky["pfs_visit_id"]),
        "designs": queries.readDesignCategories(opdb, sky["pfs_design_id"].dropna()),
    }
    directory = dataDir / "opdb"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"comparison-{period.name}-{name}.parquet" for name in frames}
    for name, frame in frames.items():
        frame.to_parquet(paths[name])
    stamp = {"period": period.name, "fetchedAt": now.isoformat(timespec="seconds"), "readUntil": str(end)}
    (directory / f"comparison-{period.name}-fetched.json").write_text(json.dumps(stamp, indent=1) + "\n")
    return paths


def loadFetched(period: Period, dataDir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Read back what `fetch` wrote.

    Parameters
    ----------
    period : `Period`
        The period.
    dataDir : `pathlib.Path`
        As given to `fetch`.

    Returns
    -------
    frames : `dict` [`str`, `pandas.DataFrame`]
        ``listing``, ``notes``, ``telStatus`` and ``designs``.
    stamp : `dict`
        When they were read.

    Raises
    ------
    FileNotFoundError
        If the period hasn't been fetched.
    """
    directory = dataDir / "opdb"
    stampPath = directory / f"comparison-{period.name}-fetched.json"
    if not stampPath.exists():
        raise FileNotFoundError(f"{period.name} hasn't been fetched into {directory}: run 'fetch' first")
    frames = {
        name: pd.read_parquet(directory / f"comparison-{period.name}-{name}.parquet") for name in _FETCHED
    }
    return frames, json.loads(stampPath.read_text())


def drpQaVersion(directory: Path | str | None = None) -> str:
    """Return ``git describe`` of the drp_qa checkout.

    Parameters
    ----------
    directory : `pathlib.Path` or `str`, optional
        The checkout. Defaults to ``$DRP_QA_DIR``, else the one this module
        was imported from.

    Returns
    -------
    `str`
        E.g. ``w.2026.41-3-gabc1234``, with ``-dirty`` for uncommitted changes.
    """
    directory = directory or os.environ.get("DRP_QA_DIR") or Path(__file__).resolve().parents[5]
    result = subprocess.run(
        ["git", "-C", str(directory), "describe", "--tags", "--always", "--dirty"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line."""
    args = _parser().parse_args(argv)
    periods = loadPeriods(args.runs)
    if args.period not in periods:
        raise SystemExit(f"Unknown period {args.period!r}; known: {', '.join(periods)}")
    period = periods[args.period]
    dataDir = Path(args.data_dir).expanduser()

    if args.command == "fetch":
        from pfs.utils.database.opdb import OpDB

        opdb = OpDB(host=args.host, user="public_user")
        for name, path in fetch(opdb, period, dataDir).items():
            print(f"{name}: {path}")
        return 0

    visits, frames, stamp = _classified(period, periods, dataDir)
    version = args.version or drpQaVersion()
    output = outputCollection(args.prefix, period.name, version)
    workDir = dataDir / period.run / "qa" / version
    workDir.mkdir(parents=True, exist_ok=True)
    print(f"{period.name}: {len(visits)} visits read from the opdb up to {stamp['readUntil']}")
    print(f"output collection: {output}")

    if args.command in ("plan", "run"):
        return _plan(args, period, visits, output, workDir)
    return _report(args, period, periods, visits, frames, stamp, output, version, workDir, dataDir)


def _classified(period: Period, periods: dict[str, Period], dataDir: Path):
    """Return a fetched period's classified visits, the fetched frames and their stamp."""
    frames, stamp = loadFetched(period, dataDir)
    visits = classifyVisits(
        frames["listing"], periods.values(), designKinds(frames["designs"]), frames["telStatus"]
    )
    return visits[visits["period"] == period.name].reset_index(drop=True), frames, stamp


def _report(args, period, periods, visits, frames, stamp, output, version, workDir, dataDir) -> int:
    """Run the ``report`` command."""
    from lsst.daf.butler import Butler

    from pfs.drp.qa.comparison.butlerQueries import detectorHoldings
    from pfs.drp.qa.comparison.findings import findings, judgeImages, taskThresholds
    from pfs.drp.qa.comparison.report import ReportInputs, buildReport, populations
    from pfs.drp.qa.metrics.readers import readMetrics

    butler = Butler(args.butler, writeable=False)
    judgedVisits = visits.loc[visits["judged"], "pfs_visit_id"]
    holdings = detectorHoldings(butler, judgedVisits, raw=args.raw, reductions=args.reductions, output=output)
    detectors = coverage(visits, holdings)
    done = detectors.loc[detectors["status"] == "judged", "visit"].unique()
    if not len(done):
        raise SystemExit(f"Nothing judged yet in {output}: run 'run' first")
    metrics = readMetrics(butler, done, collections=[output])
    metrics.to_parquet(workDir / f"iqQaMetrics-{period.name}.parquet")
    config = butler.get("imageQualityQa_config", collections=[output])
    judged = judgeImages(metrics, visits, taskThresholds(config))
    found = findings(metrics, judged, visits, frames["notes"])
    found.to_csv(workDir / f"findings-{period.name}.csv", index=False)

    reference = None
    if args.reference and args.reference != period.name:
        referencePeriod = periods[args.reference]
        referenceVersion = args.reference_version or version
        cache = (
            dataDir
            / referencePeriod.run
            / "qa"
            / referenceVersion
            / f"iqQaMetrics-{referencePeriod.name}.parquet"
        )
        if cache.exists():
            referenceVisits, _, _ = _classified(referencePeriod, periods, dataDir)
            reference = populations(pd.read_parquet(cache), referenceVisits)
        else:
            print(f"No {args.reference} metrics at {cache}: report {args.reference} first", file=sys.stderr)

    page = buildReport(
        ReportInputs(
            period=period.name,
            version=version,
            collection=output,
            readUntil=stamp["readUntil"],
            visits=visits,
            summary=summarize(visits, detectors),
            metrics=metrics,
            judged=judged,
            findings=found,
            reference=reference,
            referenceName=args.reference or "",
        )
    )
    path = workDir / f"report-{period.name}.html"
    path.write_text(page)
    print(f"report: {path}")
    return 0


def _plan(args, period: Period, visits: pd.DataFrame, output: str, workDir: Path) -> int:
    """Run the ``plan`` and ``run`` commands."""
    from lsst.daf.butler import Butler

    from pfs.drp.qa.comparison.butlerQueries import collectionExists, detectorHoldings

    butler = Butler(args.butler, writeable=False)
    judged = visits.loc[visits["judged"], "pfs_visit_id"]
    holdings = detectorHoldings(butler, judged, raw=args.raw, reductions=args.reductions, output=output)
    detectors = coverage(visits, holdings)
    detectors.to_parquet(workDir / f"coverage-{period.name}.parquet")
    summary = summarize(visits, detectors)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False))
    print("detector images:", ", ".join(f"{(detectors['status'] == s).sum()} {s}" for s in STATUS_ORDER))

    skip = [*args.reductions, *([output] if collectionExists(butler, output) else [])]
    commands = []
    for item in passes(visits, detectors):
        configFile = workDir / f"cosmicray-{period.name}-{item.name}.py"
        configFile.write_text(item.cosmicrayConfig())
        command = pipetaskCommand(
            item,
            butler=args.butler,
            pipeline=args.pipeline,
            inputs=args.inputs,
            output=output,
            skipExistingIn=skip,
            cosmicrayConfigFile=str(configFile),
            jobs=args.jobs,
        )
        commands.append((item.name, command))
        print(f"\n# {item.name}: {len(item.visits)} visits")
        print(f"cmd = {command!r}")
    if not commands:
        print("Nothing to do: every detector image is judged or has no raw data.")
    if args.command == "plan":
        return 0

    for name, command in commands:
        log = workDir / f"pipetask-{period.name}-{name}-{datetime.datetime.now():%Y%m%dT%H%M%S}.log"
        print(f"running {name}; log: {log}")
        with log.open("w") as stream:
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f"{name} failed ({result.returncode}); see {log}", file=sys.stderr)
            return result.returncode
    return 0


def _parser() -> argparse.ArgumentParser:
    drpQaDir = os.environ.get("DRP_QA_DIR", ".")
    parser = argparse.ArgumentParser(prog="qaComparison.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["fetch", "plan", "run", "report"])
    parser.add_argument("--period", required=True, help="e.g. run30 or run30-pre")
    parser.add_argument(
        "--data-dir", default="~/.cache/drp_qa/comparison", help="where opdb reads and reports are kept"
    )
    parser.add_argument("--runs", default=None, help="the run table (default: the packaged one)")
    parser.add_argument("--host", default="pfsa-db", help="opdb host, for fetch")
    parser.add_argument("--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument(
        "--prefix",
        default=f"u/{os.environ.get('USER', 'unknown')}/comparison",
        help="output collection prefix",
    )
    parser.add_argument("--version", default=None, help="drp_qa version (default: git describe)")
    parser.add_argument("--raw", default="PFS/defaults", help="collection holding the raw data")
    parser.add_argument(
        "--reductions",
        nargs="+",
        default=["drpActor/reductions"],
        help="collections whose reductions are reused",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["drpActor/reductions", "PFS/defaults"],
        help="pipetask input collections",
    )
    parser.add_argument(
        "--pipeline", default=f"{drpQaDir}/pipelines/qaThresholds.yaml", help="the reduction and QA pipeline"
    )
    parser.add_argument("--reference", default="run25", help="the period to compare with, for report")
    parser.add_argument(
        "--reference-version", default=None, help="the drp_qa version of the reference (default: --version)"
    )
    parser.add_argument("-j", "--jobs", type=int, default=8, help="pipetask processes")
    return parser
