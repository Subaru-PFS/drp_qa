"""``drpQa.py redo``: redo an observing run with an updated pipeline, end to end.

For one run it reads the run's visits from the opdb, reduces the raw frames with drp_stella's
``reduceExposure`` and ``drpActor``'s per-sequence-type configuration (unless ``--reductions`` names
a collection already reduced), measures them with the QA tasks of ``pipelines/drpQA.yaml``, loads the
numbers into qadb, makes the collection's normal values from its own calibration frames and prints
each beside the same run's value under the previous pipeline, judges every frame, and points at the
dashboard for the run report.

The reductions and QA go to ``<prefix>/<run>/<drp_qa version>/<pipeline>`` (the pipeline from
drp_stella's ``getPfsVersions``, so a different stack starts a fresh collection); with
``--reductions`` the reductions are reused and the collection is ``<prefix>/<run>/<drp_qa version>``.
Nothing is written to the opdb, and no file is left behind but the ``pipetask`` logs and cosmic-ray
configs under ``--work-dir``. The QA tasks give no verdict until the numbers are loaded: the gate
judges them (`pfs.drp.qa.metrics.gate`).

The opdb (``--opdb-host``), the Butler (``--butler``) and qadb (``--qadb-host``/``--qadb-port``) must
all be reachable, so this runs on the processing host. ``--only`` runs a subset of the stages
(``process``, ``load``, ``values``, ``judge``) to resume after an interruption.
"""

import argparse
import datetime
import getpass
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
    selectCadences,
    summarize,
)
from pfs.drp.qa.comparison.runs import Period, loadPeriods

__all__ = [
    "DRPACTOR_REDUCTIONS",
    "STAGES",
    "classifiedVisits",
    "drpQaVersion",
    "fetch",
    "fetchSummary",
    "main",
    "pipelineVersion",
    "pipetaskFailures",
    "resolveCollections",
    "unexpectedPipeline",
]

#: Where drpActor writes its reductions.
DRPACTOR_REDUCTIONS = "drpActor/reductions"

#: The default collection whose normal values a redo compares against: the previous pipeline's.
PREVIOUS_COLLECTION = "qaActor/reductions"

#: The stages ``redo`` runs, in order; ``--only`` picks a subset. ``process`` is reduce and measure.
STAGES = ("process", "load", "values", "judge")


def fetch(opdb, period: Period, now: datetime.datetime | None = None) -> tuple[dict[str, pd.DataFrame], str]:
    """Read a period's visits, notes, telescope status and design categories from the opdb.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The opdb, read-only.
    period : `Period`
        The period.
    now : `datetime.datetime`, optional
        The current time (HST); a period still under way is read up to it.

    Returns
    -------
    frames : `dict` [`str`, `pandas.DataFrame`]
        ``listing``, ``notes``, ``telStatus`` and ``designs``.
    readUntil : `str`
        The time the listing was read up to (HST).
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
    return frames, str(end)


def fetchSummary(
    frames: dict[str, pd.DataFrame], readUntil: str, period: Period, periods: dict[str, Period]
) -> str:
    """Return what a fetch read, in a few lines.

    Parameters
    ----------
    frames : `dict` [`str`, `pandas.DataFrame`]
        From `fetch`.
    readUntil : `str`
        The time the listing was read up to (HST), from `fetch`.
    period : `Period`
        The period fetched.
    periods : `dict` [`str`, `Period`]
        Every period, to classify against.

    Returns
    -------
    `str`
        Visits, sequences and nights; what the QA tasks will measure, by
        sequence type and category; what they won't, and why; the sky visits'
        labels; the designs and the notes.
    """
    visits = classifyVisits(
        frames["listing"], periods.values(), designKinds(frames["designs"]), frames["telStatus"]
    )
    visits = visits[visits["period"] == period.name]
    lines = [
        f"{period.name}: nights {period.firstNight} to {period.lastNight}, read until {readUntil}",
        f"  {len(visits):,} visits in {visits['iic_sequence_id'].nunique():,} sequences,"
        f" on {visits['night'].nunique()} nights",
    ]
    for label, subset in (
        ("gated", visits[visits["validated"]]),
        ("unvalidated", visits[visits["judged"] & ~visits["validated"]]),
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


def classifiedVisits(
    frames: dict[str, pd.DataFrame], period: Period, periods: dict[str, Period]
) -> pd.DataFrame:
    """Classify a fetch's visits and keep the period's.

    Parameters
    ----------
    frames : `dict` [`str`, `pandas.DataFrame`]
        From `fetch`.
    period : `Period`
        The period.
    periods : `dict` [`str`, `Period`]
        Every period, to classify against.

    Returns
    -------
    `pandas.DataFrame`
        `pfs.drp.qa.comparison.classify.classifyVisits` restricted to the period.
    """
    visits = classifyVisits(
        frames["listing"], periods.values(), designKinds(frames["designs"]), frames["telStatus"]
    )
    return visits[visits["period"] == period.name].reset_index(drop=True)


#: Paths that can't change what the pipeline writes: they don't name the output collection.
RESULT_NEUTRAL_PATHS = (
    # The comparison driver, module by module: comparison/reduction.py configures the reduction
    # and counts, as would any new module until listed here.
    *(
        f"python/pfs/drp/qa/comparison/{name}"
        for name in (
            "__init__.py",
            "butlerQueries.py",
            "calibInputs.py",
            "calibInputsCli.py",
            "calibValidate.py",
            "calibValidateCli.py",
            "classify.py",
            "data",
            "plan.py",
            "queries.py",
            "redo.py",
            "runs.py",
        )
    ),
    "python/pfs/drp/qa/plotting/calibInputs.py",
    # The tasks write numbers only: the limits and the judging cannot change them.
    "python/pfs/drp/qa/metrics/registry.py",
    "python/pfs/drp/qa/metrics/gate.py",
    "python/pfs/drp/qa/metrics/normalValues.py",
    "python/pfs/drp/qa/metrics/visitRollup.py",
    "python/pfs/drp/qa/qadb",
    "python/pfs/drp/qa/cli.py",
    "bin.src/drpQa.py",
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
    comparison driver therefore keeps the collection, and its finished quanta,
    while a change to a task or the reduction's configuration
    (`pfs.drp.qa.comparison.reduction`) starts a new one.

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


def say(line: str) -> None:
    """Print a line, flushed: the stages stream their progress into a log over ssh."""
    print(line, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Redo a run end to end: reduce, measure, load, make normal values, judge, point at the report."""
    args = _parser().parse_args(argv)
    periods = loadPeriods(args.runs)
    if args.run not in periods:
        raise SystemExit(f"Unknown run {args.run!r}; known: {', '.join(periods)}")
    period = periods[args.run]
    stages = _stages(args.only)

    opdb = _opdb(args)
    frames, readUntil = fetch(opdb, period)
    visits = selectCadences(classifiedVisits(frames, period, periods), args.cadence)
    say(fetchSummary(frames, readUntil, period, periods))

    # Redo reduces afresh with the current pipeline unless given an already-reduced collection.
    fresh = not args.reductions
    try:
        args.reductions, args.inputs = resolveCollections(fresh, args.raw, args.reductions or None, None)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    args.fresh = fresh
    args.command = "run"  # redo always runs the passes; --dry-run builds the graphs only
    args.passes = []  # redo processes every pass; --cadence subsets the arcs and traces
    version = args.version or drpQaVersion()
    args.variant = ""
    if fresh:
        try:
            args.variant = args.pipeline_version or pipelineVersion()
        except RuntimeError as error:
            raise SystemExit(str(error)) from None
    try:
        output = outputCollection(args.prefix, period.name, version, args.variant)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    workDir = Path(args.work_dir).expanduser() / period.run / version / args.variant
    workDir.mkdir(parents=True, exist_ok=True)
    say(f"output collection: {output}")
    say(f"stages: {', '.join(stage for stage in STAGES if stage in stages)}")

    if "process" in stages:
        rc = _plan(args, period, visits, output, workDir)
        if rc or args.dry_run:
            return rc

    if stages & {"load", "values", "judge"}:
        qadb = _qadb(args)
        if "load" in stages:
            _load(args, output, visits, opdb, qadb)
        if "values" in stages:
            _values(args, output, visits, qadb)
        if "judge" in stages:
            _judge(args, output, visits, qadb)

    _reportPointer(args, output, visits)
    return 0


def _stages(only: Sequence[str] | None) -> set[str]:
    """Return the stages to run: all of `STAGES`, or the ``--only`` subset."""
    if not only:
        return set(STAGES)
    chosen = {stage.strip() for item in only for stage in item.split(",") if stage.strip()}
    unknown = chosen - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s) {', '.join(sorted(unknown))}; known: {', '.join(STAGES)}")
    return chosen


def _opdb(args):
    """Return the read-only opdb."""
    from pfs.utils.database.opdb import OpDB

    return OpDB(host=args.opdb_host, port=args.opdb_port, dbname=args.opdb_dbname, user="public_user")


def _qadb(args):
    """Return the qadb, with a user that may write."""
    from pfs.utils.database.qadb import QaDB

    return QaDB(host=args.qadb_host, port=args.qadb_port, user=args.qadb_user, dbname=args.qadb_dbname)


def _load(args, output: str, visits: pd.DataFrame, opdb, qadb) -> None:
    """Load the collection's numbers into qadb and judge them."""
    from lsst.daf.butler import Butler

    from pfs.drp.qa.qadb.loader import loadCollection

    judged = sorted(int(v) for v in visits.loc[visits["judged"], "pfs_visit_id"])
    say(f"\nloading {len(judged)} visits of {output} into qadb")
    counts = loadCollection(
        Butler(args.butler, writeable=False),
        qadb,
        output,
        visits=judged,
        opdb=opdb,
        mode="comparison",
        log=say,
    )
    for table, count in counts.items():
        say(f"{table}: {count}")


def _values(args, output: str, visits: pd.DataFrame, qadb) -> None:
    """Make the collection's normal values from its calibration frames, against the previous pipeline."""
    from pfs.drp.qa.metrics.normalValues import formatComparison
    from pfs.drp.qa.qadb.judge import makeValues

    ranges = _calibRanges(args, visits)
    if not ranges:
        say("\nno calibration-set arcs or traces in the run; no normal values made")
        return
    say(f"\nnormal values from the calibration set, each beside its value in {args.previous_collection}")
    compared, stored, refused = makeValues(
        qadb,
        output,
        ranges,
        baseline=args.previous_collection,
        arms=args.arm or None,
        store=args.accept is not None,
        acceptedBy=args.accept or "",
        reason=args.reason,
        log=say,
    )
    say(formatComparison(compared, onlyShifted=not args.all_values))
    failed = refused[refused["status"] == "FAIL"]
    if len(failed):
        say(f"{len(failed)} values shifted beyond FAIL and were refused; give --reason to accept them")
    if args.accept is None:
        say(f"nothing stored: {len(stored)} values would be, with --accept")
    else:
        say(f"{len(stored)} values stored for {output}, accepted by {args.accept}")


def _judge(args, output: str, visits: pd.DataFrame, qadb) -> None:
    """Judge every frame of the collection again, from the numbers in qadb."""
    from pfs.drp.qa.qadb.judge import judgeStored

    judged = visits.loc[visits["judged"], "pfs_visit_id"]
    if judged.empty:
        say("\nno judged visits to judge")
        return
    say("")
    judgeStored(qadb, output, int(judged.min()), int(judged.max()), mode="comparison", log=say)


def _calibRanges(args, visits: pd.DataFrame) -> list[tuple[int, int]]:
    """Return the run's calibration-frame visit ranges: ``--calib-visits``, or its set arcs and traces."""
    if args.calib_visits:
        return [tuple(args.calib_visits)]
    calib = visits.loc[visits["validated"] & (visits["cadence"] == "set"), "pfs_visit_id"]
    return _ranges(int(v) for v in calib)


def _ranges(values) -> list[tuple[int, int]]:
    """Return sorted distinct ints as contiguous ``(first, last)`` ranges."""
    ordered = sorted({int(value) for value in values})
    ranges: list[list[int]] = []
    for value in ordered:
        if ranges and value == ranges[-1][1] + 1:
            ranges[-1][1] = value
        else:
            ranges.append([value, value])
    return [(first, last) for first, last in ranges]


def _reportPointer(args, output: str, visits: pd.DataFrame) -> None:
    """Say how to render the run report from qadb with the dashboard (which reads qadb, judging nothing)."""
    judged = visits.loc[visits["judged"], "pfs_visit_id"]
    first = int(judged.min()) if not judged.empty else 0
    say("")
    say(f"Run report: {output} is in qadb. Render it with pfs_qa_dashboard, e.g.")
    say(
        f"  qaDashboard.py report --collection {output} --first-visit {first}"
        f" --qadb-host {args.qadb_host} --qadb-port {args.qadb_port} --name {args.run} --out <dir>"
    )
    say(f"then serve the page it writes ({args.run}.html).")


def pipelineVersion(versions: dict[str, str] | None = None) -> str:
    """Return the label of the pipeline that reduces, which names fresh reductions.

    Parameters
    ----------
    versions : `dict` [`str`, `str`], optional
        `pfs.drp.stella.utils.sysUtils.getPfsVersions` with ``prefix=""``
        (``DATAMODEL``, ``OBS_PFS``, ``DRP_STELLA``); read from it by default.

    Returns
    -------
    `str`
        ``pfs-<version>`` when every product has the same version, as a
        weekly does (``pfs-w.2026.40``); otherwise each product's,
        ``datamodel-<v>+obs_pfs-<v>+drp_stella-<v>``.

    Raises
    ------
    RuntimeError
        If the versions can't be read (drp_stella isn't set up), or one is
        unknown.
    """
    if versions is None:
        try:
            from pfs.drp.stella.utils.sysUtils import getPfsVersions
        except ImportError as error:
            raise RuntimeError(
                f"Can't read the pipeline's versions ({error}): use --pipeline-version"
            ) from None
        versions = getPfsVersions(prefix="")
    versions = {key.replace("HIERARCH ", "").lower(): str(value) for key, value in versions.items()}
    if not versions or any(value in ("", "unknown") for value in versions.values()):
        raise RuntimeError(f"Unknown pipeline versions {versions}: use --pipeline-version")
    if len(set(versions.values())) == 1:
        return f"pfs-{next(iter(versions.values()))}"
    return "+".join(f"{name}-{value}" for name, value in versions.items())


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


def _plan(args, period: Period, visits: pd.DataFrame, output: str, workDir: Path) -> int:
    """Plan the reduce and measure passes and run them (the ``process`` stage)."""
    from lsst.daf.butler import Butler

    from pfs.drp.qa.comparison.butlerQueries import (
        collectionExists,
        detectorHoldings,
        freshPipelines,
        ownRuns,
    )

    butler = Butler(args.butler, writeable=False)
    judged = visits.loc[visits["judged"], "pfs_visit_id"]
    earlier = _earlier(butler, args, period, output)
    if earlier:
        print(f"reusing the reductions of {', '.join(earlier)}")
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
    if args.arm:
        detectors = detectors[detectors["arm"].isin(args.arm)]
    if args.spectrograph:
        detectors = detectors[detectors["spectrograph"].isin(args.spectrograph)]
    if args.arm or args.spectrograph:
        detectors = detectors.reset_index(drop=True)
        print(f"detectors restricted to arm {args.arm or 'any'}, spectrograph {args.spectrograph or 'any'}")
    summary = summarize(visits, detectors)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False))
    print("detector images:", ", ".join(f"{(detectors['status'] == s).sum()} {s}" for s in STATUS_ORDER))
    if (detectors["status"] == "failed").any():
        print("failed images are not rescheduled (they fail the same way each time); --retry-failed to retry")
    if args.fresh:
        # This label's own collections include the output itself, which `earlier` leaves out: a run
        # resumed after an interruption, or retried, must not be told its label has nothing yet.
        labels = freshPipelines(butler, args.prefix, period.name)
        known = [*earlier, *labels.get(args.variant, [])]
        others = [label for label in labels if label != args.variant]
        warning = unexpectedPipeline(
            args.variant, known, others, int((detectors["status"] == "to reduce").sum())
        )
        if warning and args.command == "run" and not args.dry_run and not args.new_pipeline:
            raise SystemExit(
                f"!! {warning}\n!! Set up that stack, or pass --new-pipeline to reduce with this one."
            )
        if warning:
            print(f"!! {warning}", file=sys.stderr)

    outputExists = collectionExists(butler, output)
    # Skip what this version has measured, in its own runs: skipping the output chain would also skip
    # every image an earlier version in its inputs measured.
    skip = [*args.reductions, *(ownRuns(butler, output) if outputExists else [])]
    commands = []
    for item in passes(visits, detectors):
        if (
            args.passes
            and item.name.removesuffix("-judge") not in args.passes
            and item.name not in args.passes
        ):
            print(f"\n# {item.name} ({', '.join(item.types)}): {len(item.visits)} visits, skipped (--pass)")
            continue
        print(f"\n# {item.name} ({', '.join(item.types)}): {len(item.visits)} visits")
        # A reducing pass runs two pipetask commands, reduce then measure; a judge-only pass measures only.
        phases = []
        if not item.judgeOnly:
            configFile = workDir / f"cosmicray-{period.name}-{item.name}.py"
            configFile.write_text(item.cosmicrayConfig())
            phases.append((f"{item.name}-reduce", "reduce", str(configFile)))
        phases.append((item.name, "measure", None))
        for name, phase, configFile in phases:
            command = pipetaskCommand(
                item,
                phase=phase,
                butler=args.butler,
                # Earlier collections are inputs but not skipped: their QA outputs are an older drp_qa's,
                # to be measured again. The measure phase reads this run's reductions from the output chain.
                inputs=[*earlier, *args.inputs],
                output=output,
                skipExistingIn=skip,
                cosmicrayConfigFile=configFile,
                jobs=args.jobs,
                # The first command creates the output chain when it doesn't exist yet; every later one
                # finds it, and must rebase in case drpActor added a reduction RUN meanwhile (daily).
                rebase=outputExists or bool(commands),
                arms=args.arm or None,
                spectrographs=args.spectrograph or None,
            )
            if args.dry_run:
                command = _qgraph(command)
            commands.append((name, command))
            print(f"cmd = {command!r}")
    if not commands:
        print("Nothing to do: every detector image is judged or has no raw data.")
    if args.command == "plan":
        return 0

    # A failed quantum fails its pass but not the others: run every pass, then say what failed.
    failed = []
    for index, (name, command) in enumerate(commands):
        if not args.dry_run and (outputExists or (index > 0 and collectionExists(butler, output))):
            # The chain must hold today's inputs before pipetask searches it for the graph. A
            # first pass that failed before making the chain leaves nothing to rebase.
            from lsst.daf.butler import Butler

            from pfs.drp.qa.comparison.butlerQueries import rebaseOutputChain

            inputs = command[command.index("-i") + 1].split(",")
            rebaseOutputChain(Butler(args.butler, writeable=True), output, inputs)
        step = "qgraph" if args.dry_run else "pipetask"
        log = workDir / f"{step}-{period.name}-{name}-{datetime.datetime.now():%Y%m%dT%H%M%S}.log"
        print(f"running {name}; log: {log}", flush=True)
        with log.open("w") as stream:
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        text = log.read_text(errors="replace")
        if args.dry_run:
            print("\n".join(line for line in text.splitlines() if "quanta" in line))
        summary, lines = pipetaskFailures(text)
        if summary:
            print(f"{name}: {summary}", flush=True)
        newlyFailed = failedQuanta(text)
        if not newlyFailed.empty:
            print(f"{name}: {len(newlyFailed)} detector images failed, now marked failed", flush=True)
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


def unexpectedPipeline(
    variant: str, earlier: Sequence[str], others: Sequence[str], toReduce: int
) -> str | None:
    """Say when a ``--fresh`` run would reduce a period again under a new pipeline label.

    Reductions are reused only under the same label, so a shell with another
    stack set up plans the whole reduction again, silently, where a pass of
    the QA tasks alone was meant.

    Parameters
    ----------
    variant : `str`
        This run's pipeline label; empty when not ``--fresh``.
    earlier : sequence of `str`
        The period's earlier collections under this label (reused).
    others : sequence of `str`
        The period's fresh pipeline labels other than this one.
    toReduce : `int`
        Detector images the plan would reduce.

    Returns
    -------
    `str` or `None`
        The warning, or `None` when nothing is amiss: no other label, this
        label already has reductions, or nothing is to reduce.
    """
    if not variant or earlier or not others or toReduce <= 0:
        return None
    return (
        f"--fresh would reduce {toReduce} detector images under pipeline {variant!r}, which has no "
        f"reductions of this period yet, while {', '.join(repr(o) for o in others)} "
        f"{'has' if len(others) == 1 else 'have'}: is this the stack you meant?"
    )


def _earlier(butler, args, period: Period, output: str) -> list[str]:
    """Return the runs of this period's earlier comparisons (same mode) whose reductions this one reuses."""
    from pfs.drp.qa.comparison.butlerQueries import earlierRuns

    return earlierRuns(butler, args.prefix, period.name, args.variant, output)


def _qgraph(command: list[str]) -> list[str]:
    """Turn a ``pipetask run`` command into the ``pipetask qgraph`` that builds its graph only.

    Drops the run-only options ``-j`` (with its value) and ``--register-dataset-types``, which
    ``pipetask qgraph`` does not accept.
    """
    index = command.index("run")
    rest = command[index + 1 :]
    out: list[str] = []
    i = 0
    while i < len(rest):
        if rest[i] == "-j":
            i += 2
        elif rest[i] == "--register-dataset-types":
            i += 1
        else:
            out.append(rest[i])
            i += 1
    return [*command[:index], "qgraph", *out]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drpQa.py redo", description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="the run to redo, e.g. run31 or run31-pre")
    parser.add_argument("--runs", default=None, help="the run table (default: the packaged one)")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="STAGE[,STAGE]",
        help=f"run only these stages ({', '.join(STAGES)}), to resume; repeatable (default: all)",
    )
    parser.add_argument(
        "--reductions",
        nargs="+",
        default=None,
        metavar="COLLECTION",
        help="reuse these collections' reductions instead of reducing afresh (e.g. drpActor/reductions)",
    )
    parser.add_argument(
        "--version", default=None, help="drp_qa version naming the collection (default: git describe)"
    )
    parser.add_argument(
        "--pipeline-version",
        default=None,
        help="the pipeline label naming fresh reductions (default: from drp_stella's getPfsVersions)",
    )
    parser.add_argument("--new-pipeline", action="store_true", help=argparse.SUPPRESS)
    # Butler / collections.
    parser.add_argument("--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument(
        "--prefix",
        default=f"u/{os.environ.get('USER', 'unknown')}/redo",
        help="output collection prefix (the collection is <prefix>/<run>/<version>[/<pipeline>])",
    )
    parser.add_argument("--raw", default="PFS/defaults", help="collection holding the raw data")
    parser.add_argument(
        "--work-dir",
        default="~/.cache/drp_qa/redo",
        help="where the pipetask logs and cosmic-ray configs go",
    )
    parser.add_argument("-j", "--jobs", type=int, default=8, help="pipetask processes")
    parser.add_argument(
        "--cadence",
        action="append",
        default=[],
        choices=["set", "daily", "single"],
        help="process only the arcs and traces of this cadence, e.g. set for the calibration sets; repeatable",
    )
    parser.add_argument(
        "--arm",
        nargs="+",
        choices=["b", "r", "m", "n"],
        help="process only these arms, e.g. b r m to skip the slow n arm (default: every arm)",
    )
    parser.add_argument(
        "--spectrograph",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        help="process only these spectrographs (default: every spectrograph)",
    )
    parser.add_argument("--retry-failed", action="store_true", help="retry images whose quanta failed before")
    parser.add_argument(
        "--dry-run", action="store_true", help="process stage: build each pass's graph, write nothing"
    )
    # opdb (read-only).
    parser.add_argument("--opdb-host", default="pfsa-db", help="opdb host")
    parser.add_argument("--opdb-port", type=int, default=5432, help="opdb port")
    parser.add_argument("--opdb-dbname", default="opdb", help="opdb database name")
    # qadb (writable), for load, values and judge.
    parser.add_argument("--qadb-host", default="pfsa-db", help="qadb host")
    parser.add_argument("--qadb-port", type=int, default=5436, help="qadb port")
    parser.add_argument("--qadb-user", default="pfs", help="qadb user; must be allowed to write")
    parser.add_argument("--qadb-dbname", default="qadb", help="qadb database name")
    # Normal values (the values stage).
    parser.add_argument(
        "--previous-collection",
        default=PREVIOUS_COLLECTION,
        help="the collection whose normal values the new ones are compared with (the previous pipeline's)",
    )
    parser.add_argument(
        "--calib-visits",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="the calibration frames for the normal values (default: the run's set-cadence arcs and traces)",
    )
    parser.add_argument(
        "--accept",
        nargs="?",
        const=getpass.getuser(),
        metavar="NAME",
        help="store the normal values, accepted by NAME (default: the user); without it they are only compared",
    )
    parser.add_argument("--reason", default="", help="why normal values shifted beyond FAIL are accepted")
    parser.add_argument(
        "--all-values", action="store_true", help="list every normal value, not only those moved"
    )
    return parser
