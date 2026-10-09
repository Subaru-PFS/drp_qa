"""Command line of comparison mode: ``bin.src/qaComparison.py``.

Four steps, each safe to repeat:

``fetch``
    Read a period's visits and what describes them from the opdb, read-only, into parquet.
``plan``
    Classify the visits, ask the Butler what it holds, and print the coverage and the
    ``pipetask`` commands that would complete it. Writes nothing to the Butler.
``run``
    Plan, then run those commands; ``--dry-run`` only builds their graphs, and ``--pass`` picks
    passes (``calibration``, ``sky``, ``unvalidated-calibration``).

By default the reductions in ``drpActor/reductions`` are reused, each made with the pipeline of its
day. ``--fresh`` reduces everything with the current pipeline instead, so that runs are compared on
a level field. Its collection and files are named after that pipeline (`pipelineVersion`), and a
later drp_qa version reuses the fresh reductions of the same pipeline, judging them again.
``report``
    Read the verdicts and write the period's report.

Files go under ``--data-dir``: the opdb reads in ``opdb/``, and everything for one drp_qa
version in ``<run>/qa/<version>/``. A pre-run period's files sit with its run's.
"""

import argparse
import datetime
import json
import os
import re
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
    failedQuanta,
    outputCollection,
    passes,
    pipetaskCommand,
    summarize,
)
from pfs.drp.qa.comparison.runs import Period, loadPeriods

__all__ = [
    "DRPACTOR_REDUCTIONS",
    "drpQaVersion",
    "fetch",
    "fetchSummary",
    "loadFetched",
    "main",
    "pipelineVersion",
    "pipetaskFailures",
    "resolveCollections",
]

#: Where drpActor writes its reductions.
DRPACTOR_REDUCTIONS = "drpActor/reductions"

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


def fetchSummary(
    frames: dict[str, pd.DataFrame], stamp: dict, period: Period, periods: dict[str, Period]
) -> str:
    """Return what a fetch read, in a few lines.

    Parameters
    ----------
    frames : `dict` [`str`, `pandas.DataFrame`]
        From `loadFetched`.
    stamp : `dict`
        From `loadFetched`.
    period : `Period`
        The period fetched.
    periods : `dict` [`str`, `Period`]
        Every period, to classify against.

    Returns
    -------
    `str`
        Visits, sequences and nights; what the gate will judge, by sequence
        type and category; what it won't, and why; the sky visits' labels; the
        designs and the notes.
    """
    visits = classifyVisits(
        frames["listing"], periods.values(), designKinds(frames["designs"]), frames["telStatus"]
    )
    visits = visits[visits["period"] == period.name]
    lines = [
        f"{period.name}: nights {period.firstNight} to {period.lastNight}, read until {stamp['readUntil']}",
        f"  {len(visits):,} visits in {visits['iic_sequence_id'].nunique():,} sequences,"
        f" on {visits['night'].nunique()} nights",
    ]
    for label, subset in (
        ("gated", visits[visits["validated"]]),
        ("unvalidated (judged, no validated thresholds)", visits[visits["judged"] & ~visits["validated"]]),
    ):
        lines.append(f"  {len(subset):,} {label}")
        kinds = subset.groupby(["category", "sequence_type", "cadence"]).size()
        for (category, sequenceType, cadence), count in kinds.items():
            lines.append(f"    {count:6,}  {category:<12} {sequenceType} {cadence}".rstrip())
    others = visits.loc[~visits["judged"], "reason"].value_counts()
    if len(others):
        lines.append(f"  {int(others.sum()):,} not judged")
        lines += [f"    {count:6,}  {reason}" for reason, count in others.items()]
    sky = visits[visits["sequence_type"].isin(SKY_TYPES)]
    if len(sky):
        withStatus = int(sky["pfs_visit_id"].isin(frames["telStatus"]["pfs_visit_id"]).sum())
        designs = sky["pfs_design_id"].dropna().unique()
        kinds = designKinds(frames["designs"])
        science = int(sum(kinds.get(design, "engineering") == "science" for design in designs))
        lines.append(
            f"  sky: {len(sky):,} visits ({withStatus:,} with telescope status), "
            f"{int((sky['category'] == 'science').sum()):,} science; "
            f"{int(sky['focusSweep'].sum())} in focus sweeps, {int(sky['dithered'].sum())} dithered; "
            f"{len(designs)} designs, {science} science"
        )
    where = {"obslog": "on visits", "obslog_sequence": "on sequences", "sps_annotation": "on cameras"}
    notes = frames["notes"]["source"].value_counts()
    detail = ", ".join(f"{count} {where.get(source, source)}" for source, count in notes.items())
    lines.append(f"  notes: {int(notes.sum())}" + (f" ({detail})" if detail else ""))
    return "\n".join(lines)


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


#: Paths that can't change what the pipeline writes: they don't name the output collection.
RESULT_NEUTRAL_PATHS = (
    "python/pfs/drp/qa/comparison",
    "python/pfs/drp/qa/plotting/comparison.py",
    "bin.src/qaComparison.py",
    "tests",
    "docs",
    "examples",
    ".github",
    "*.md",
)


def drpQaVersion(directory: Path | str | None = None) -> str:
    """Return the drp_qa version that names a comparison's output collection.

    It is ``git describe`` of the last commit that could change what the
    pipeline writes: every path but `RESULT_NEUTRAL_PATHS`. Changing the
    comparison driver or its report therefore keeps the collection, and its
    finished quanta, while a change to a task or a threshold starts a new one.

    Parameters
    ----------
    directory : `pathlib.Path` or `str`, optional
        The checkout. Defaults to ``$DRP_QA_DIR``, else the one this module
        was imported from.

    Returns
    -------
    `str`
        E.g. ``w.2026.41-3-gabc1234``, with ``-dirty`` when those paths have
        uncommitted changes.
    """
    directory = str(directory or os.environ.get("DRP_QA_DIR") or Path(__file__).resolve().parents[5])
    pathspec = [".", *(f":(exclude){path}" for path in RESULT_NEUTRAL_PATHS)]

    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", directory, *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    commit = git("log", "-1", "--format=%H", "--", *pathspec)
    version = git("describe", "--tags", "--always", commit)
    if git("status", "--porcelain", "--untracked-files=no", "--", *pathspec):
        version += "-dirty"
    return version


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
        paths = fetch(opdb, period, dataDir)
        frames, stamp = loadFetched(period, dataDir)
        print(fetchSummary(frames, stamp, period, periods))
        print(f"written to {paths['listing'].parent}")
        return 0

    visits, frames, stamp = _classified(period, periods, dataDir)
    try:
        args.reductions, args.inputs = resolveCollections(args.fresh, args.raw, args.reductions, args.inputs)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    version = args.version or drpQaVersion()
    args.variant = ""
    if args.fresh:
        try:
            args.variant = f"drp_stella-{args.pipeline_version or pipelineVersion()}"
        except RuntimeError as error:
            raise SystemExit(str(error)) from None
    try:
        output = outputCollection(args.prefix, period.name, version, args.variant)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    workDir = dataDir / period.run / "qa" / version / args.variant
    workDir.mkdir(parents=True, exist_ok=True)
    print(f"{period.name}: {len(visits)} visits read from the opdb up to {stamp['readUntil']}")
    print(f"output collection: {output}")

    if args.command in ("plan", "run"):
        return _plan(args, period, visits, output, workDir)
    return _report(args, period, periods, visits, frames, stamp, output, version, workDir, dataDir)


def pipelineVersion(environ: dict[str, str] | None = None) -> str:
    """Return the version of the drp_stella that reduces, which names fresh reductions.

    Parameters
    ----------
    environ : `dict` [`str`, `str`], optional
        The environment. Default `os.environ`.

    Returns
    -------
    `str`
        The EUPS version of the drp_stella set up (``SETUP_DRP_STELLA``), e.g.
        ``w.2026.40``; for one set up from a checkout (``setup -r``), ``git
        describe`` of ``DRP_STELLA_DIR``, with ``-dirty`` if it has changes.

    Raises
    ------
    RuntimeError
        If drp_stella isn't set up, or its checkout can't be described.
    """
    environ = os.environ if environ is None else environ
    setup = environ.get("SETUP_DRP_STELLA", "").split()
    if len(setup) > 1 and not setup[1].startswith("LOCAL:"):
        return setup[1]
    directory = environ.get("DRP_STELLA_DIR")
    if not directory:
        raise RuntimeError("drp_stella isn't set up: set it up, or name the pipeline with --pipeline-version")
    result = subprocess.run(
        ["git", "-C", directory, "describe", "--tags", "--always", "--dirty"], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(f"Can't describe drp_stella in {directory}: name it with --pipeline-version")
    return result.stdout.strip()


def resolveCollections(
    fresh: bool, raw: str, reductions: Sequence[str] | None, inputs: Sequence[str] | None
) -> tuple[list[str], list[str]]:
    """Return the collections whose reductions are reused, and the pipeline's inputs.

    Parameters
    ----------
    fresh : `bool`
        Whether everything is reduced afresh.
    raw : `str`
        The collection with the raw data and calibrations, e.g. ``PFS/defaults``.
    reductions, inputs : sequence of `str`, or `None`
        As given on the command line; `None` for the default.

    Returns
    -------
    reductions : `list` [`str`]
        ``drpActor/reductions`` by default; none when ``fresh``.
    inputs : `list` [`str`]
        ``reductions`` then ``raw`` by default.

    Raises
    ------
    ValueError
        If ``fresh`` and reductions are named, or the inputs include
        ``drpActor/reductions``: they would be reused.
    """
    if fresh:
        if reductions or (inputs and DRPACTOR_REDUCTIONS in inputs):
            raise ValueError(f"--fresh reuses no reductions: drop --reductions and {DRPACTOR_REDUCTIONS}")
        return [], list(inputs or [raw])
    reductions = list(reductions) if reductions is not None else [DRPACTOR_REDUCTIONS]
    return reductions, list(inputs) if inputs is not None else [*reductions, raw]


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
    from pfs.drp.qa.comparison.report import ReportInputs, buildReport, failedSummary, populations
    from pfs.drp.qa.metrics.readers import readMetrics

    butler = Butler(args.butler, writeable=False)
    judgedVisits = visits.loc[visits["judged"], "pfs_visit_id"]
    earlier = _earlier(butler, args, period, output)
    holdings = detectorHoldings(
        butler,
        judgedVisits,
        raw=args.raw,
        reductions=[*args.reductions, *earlier],
        output=output,
        inputs=[*earlier, *args.inputs],
    )
    failed = _failedImages(workDir, period)
    detectors = coverage(visits, holdings, failed)
    done = detectors.loc[detectors["status"] == "judged", "visit"].unique()
    if not len(done):
        raise SystemExit(f"Nothing judged yet in {output}: run 'run' first")
    pending = detectors["status"].isin(["to judge", "to reduce"])
    if pending.any():
        byType = detectors[pending].groupby("sequence_type").size().sort_values(ascending=False)
        listed = ", ".join(f"{n} {kind}" for kind, n in byType.items())
        print(
            f"!! INCOMPLETE: {pending.sum()} of {len(detectors)} detector images not judged yet ({listed});"
            " the report covers the rest. Run 'run' (and check its failures) to complete it.",
            file=sys.stderr,
        )
    metrics = readMetrics(butler, done, collections=[output])
    metrics.to_parquet(workDir / f"iqQaMetrics-{period.name}.parquet")
    config = butler.get("imageQualityQa_config", collections=[output])
    judged = judgeImages(metrics, visits, taskThresholds(config))
    from pfs.drp.qa.metrics.validationVisits import loadValidationVisits

    found = findings(metrics, judged, visits, frames["notes"], visitSet=loadValidationVisits())
    found.to_csv(workDir / f"findings-{period.name}.csv", index=False)
    from pfs.drp.qa.comparison.persistence import armTimeline, lastLitBefore

    lastLit = lastLitBefore(visits)
    lastLit.to_csv(workDir / f"darksLastLit-{period.name}.csv", index=False)

    reference = None
    if args.reference and args.reference != period.name:
        referencePeriod = periods[args.reference]
        referenceVersion = args.reference_version or version
        cache = (
            dataDir
            / referencePeriod.run
            / "qa"
            / referenceVersion
            / args.variant
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
            lastLit=lastLit,
            timeline=armTimeline(visits),
            failed=failedSummary(detectors, failed),
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
    earlier = _earlier(butler, args, period, output)
    if earlier:
        print(f"reusing the fresh reductions of {', '.join(earlier)}")
    holdings = detectorHoldings(
        butler,
        judged,
        raw=args.raw,
        reductions=[*args.reductions, *earlier],
        output=output,
        inputs=[*earlier, *args.inputs],
    )
    failed = None if args.retry_failed else _failedImages(workDir, period)
    detectors = coverage(visits, holdings, failed)
    detectors.to_parquet(workDir / f"coverage-{period.name}.parquet")
    summary = summarize(visits, detectors)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False))
    print("detector images:", ", ".join(f"{(detectors['status'] == s).sum()} {s}" for s in STATUS_ORDER))
    if (detectors["status"] == "failed").any():
        print("failed images are not rescheduled (they fail the same way each time); --retry-failed to retry")

    outputExists = collectionExists(butler, output)
    skip = [*args.reductions, *([output] if outputExists else [])]
    commands = []
    for item in passes(visits, detectors):
        if (
            args.passes
            and item.name.removesuffix("-judge") not in args.passes
            and item.name not in args.passes
        ):
            print(f"\n# {item.name} ({', '.join(item.types)}): {len(item.visits)} visits, skipped (--pass)")
            continue
        configFile = None
        if not item.judgeOnly:
            configFile = workDir / f"cosmicray-{period.name}-{item.name}.py"
            configFile.write_text(item.cosmicrayConfig())
        command = pipetaskCommand(
            item,
            butler=args.butler,
            pipeline=args.pipeline,
            # Earlier fresh collections are inputs but not skipped: their imageQualityQa outputs are
            # an older drp_qa's, to be judged again.
            inputs=[*earlier, *args.inputs],
            output=output,
            skipExistingIn=skip,
            cosmicrayConfigFile=None if configFile is None else str(configFile),
            jobs=args.jobs,
            rebase=outputExists,
        )
        if args.dry_run:
            command = _qgraph(command)
        commands.append((item.name, command))
        print(f"\n# {item.name} ({', '.join(item.types)}): {len(item.visits)} visits")
        print(f"cmd = {command!r}")
    if not commands:
        print("Nothing to do: every detector image is judged or has no raw data.")
    if args.command == "plan":
        return 0

    # A failed quantum fails its pass but not the others: run every pass, then say what failed.
    failed = []
    for name, command in commands:
        step = "qgraph" if args.dry_run else "pipetask"
        log = workDir / f"{step}-{period.name}-{name}-{datetime.datetime.now():%Y%m%dT%H%M%S}.log"
        print(f"running {name}; log: {log}")
        with log.open("w") as stream:
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        text = log.read_text(errors="replace")
        if args.dry_run:
            print("\n".join(line for line in text.splitlines() if "quanta" in line))
        summary, lines = pipetaskFailures(text)
        if summary:
            print(f"{name}: {summary}")
        newlyFailed = failedQuanta(text)
        if not newlyFailed.empty:
            print(f"{name}: {len(newlyFailed)} detector images failed, now marked failed")
        if result.returncode:
            failed.append((name, result.returncode, log, lines))
    for name, code, log, lines in failed:
        print(f"\n{'!' * 80}\n!! {name} FAILED (exit {code}); log: {log}", file=sys.stderr)
        for line in lines:
            print(f"!!   {line}", file=sys.stderr)
    if failed:
        names = ", ".join(name for name, *_ in failed)
        print(f"{'!' * 80}\n!! {len(failed)} of {len(commands)} passes failed: {names}", file=sys.stderr)
        return 1
    return 0


def _failedImages(workDir: Path, period: Period) -> pd.DataFrame:
    """Return the detector images that failed in any of a period's ``pipetask run`` logs."""
    logs = sorted(workDir.glob(f"pipetask-{period.name}-*.log"))
    found = [failedQuanta(log.read_text(errors="replace")) for log in logs]
    if not found:
        return failedQuanta("")
    failed = pd.concat(found, ignore_index=True)
    return failed.drop_duplicates(["visit", "arm", "spectrograph"], keep="last", ignore_index=True)


#: How many failure lines of a pipetask log `pipetaskFailures` keeps.
MAX_FAILURE_LINES = 20


def pipetaskFailures(text: str, limit: int = MAX_FAILURE_LINES) -> tuple[str | None, list[str]]:
    """Return a ``pipetask run`` log's outcome and the lines saying what failed.

    Parameters
    ----------
    text : `str`
        The log.
    limit : `int`, optional
        How many failure lines to keep; the last ones, where pipetask lists the failed quanta.

    Returns
    -------
    summary : `str` or `None`
        pipetask's closing ``Executed N quanta successfully, M failed ...`` line, without its log
        prefix; `None` when the log has none (a dry run, or pipetask stopped before executing).
    lines : `list` [`str`]
        The distinct ``ERROR`` lines, lines saying something failed and the exceptions ending
        tracebacks, in order, at most ``limit`` (the last), each cut to 300 characters.
    """
    summary = None
    lines = []
    for line in text.splitlines():
        if "Executed" in line and "quanta successfully" in line:
            summary = line[line.index("Executed") :].strip()
            continue
        isException = re.match(r"[A-Za-z_.]*(Error|Exception): ", line) is not None
        if ("ERROR" in line or " failed" in line.lower() or isException) and line.strip() not in lines:
            lines.append(line.strip())
    return summary, [line[:300] for line in lines[-limit:]]


def _earlier(butler, args, period: Period, output: str) -> list[str]:
    """Return the earlier collections whose fresh reductions this one reuses; none unless --fresh."""
    from pfs.drp.qa.comparison.butlerQueries import earlierReductions

    if not args.fresh:
        return []
    return earlierReductions(butler, args.prefix, period.name, args.variant, output)


def _qgraph(command: list[str]) -> list[str]:
    """Turn a ``pipetask run`` command into the ``pipetask qgraph`` that builds its graph only."""
    index = command.index("run")
    rest = command[index + 1 :]
    if rest[:1] == ["-j"]:
        rest = rest[2:]
    return [*command[:index], "qgraph", *rest]


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
        default=None,
        help=f"collections whose reductions are reused (default: {DRPACTOR_REDUCTIONS})",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="pipetask input collections (default: the reductions, then --raw)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="reduce everything with the current pipeline, reusing nothing of drpActor's; the collection and"
        " files are named after the pipeline (drp_stella-<version>)",
    )
    parser.add_argument(
        "--pipeline-version",
        default=None,
        help="with --fresh: the drp_stella version naming the reductions (default: the one set up)",
    )
    parser.add_argument(
        "--pipeline", default=f"{drpQaDir}/pipelines/qaThresholds.yaml", help="the reduction and QA pipeline"
    )
    parser.add_argument("--reference", default="run25", help="the period to compare with, for report")
    parser.add_argument(
        "--reference-version", default=None, help="the drp_qa version of the reference (default: --version)"
    )
    parser.add_argument("-j", "--jobs", type=int, default=8, help="pipetask processes")
    parser.add_argument(
        "--pass",
        dest="passes",
        action="append",
        default=[],
        help="only this pass, as plan names it: calibration, sky, unvalidated-calibration, ... (with its"
        " -judge pass), or sky-judge, ...; repeatable",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="with plan or run: reschedule the images whose quanta failed in an earlier run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with run: build each pass's graph (pipetask qgraph), write nothing",
    )
    return parser
