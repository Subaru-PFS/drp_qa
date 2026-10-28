"""The end-of-run AG report: how the guide cameras and the guide loop did over an observing run.

Two steps, as ``bin.src/agRunReport.py``::

    agRunReport.py fetch --period run30     # read the run from the opdb, once
    agRunReport.py report --period run30    # flags, plots and tables, from what was read

The runs are those of comparison mode (`pfs.drp.qa.comparison.runs`). `fetch`
is the only step that touches the opdb (through
`pfs.drp.qa.guiders.queries.readAgRun`); `report` reads its parquet files, so
a report can be redrawn, or its thresholds changed, without the database.

The report holds science visit IDs and guiding statistics, so it stays local.
"""

import argparse
import datetime
import json
import zoneinfo
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pfs.drp.qa.comparison.runs import Period, loadPeriods
from pfs.drp.qa.guiders import runPlotting
from pfs.drp.qa.guiders.queries import AG_RUN_TABLES
from pfs.drp.qa.guiders.runAnalysis import (
    AgRun,
    RunConfig,
    cameraSummary,
    describePeriod,
    findDropouts,
    findFlags,
    findRejectionClusters,
    fitModelErrors,
    groupSummary,
    prepareRun,
    settledJitter,
)
from pfs.drp.qa.guiders.runPlotting import CAMERA_ORDER

__all__ = ["fetchRun", "loadRun", "main", "writeReport"]

#: The run-wide figures of a report, by file name; each camera's follows, as ``camera-AG<n>``.
FIGURES = {
    "cameraPresence": ("Camera presence", lambda run, jitter: runPlotting.plotCameraPresence(run)),
    "guiding": ("Guiding", runPlotting.plotGuiding),
    "focus": ("Focus", lambda run, jitter: runPlotting.plotFocus(run)),
    "acquisition": ("Acquisition", lambda run, jitter: runPlotting.plotAcquisition(run)),
}


def _paths(period: Period, dataDir: Path) -> tuple[dict[str, Path], Path]:
    directory = dataDir / "opdb"
    return (
        {name: directory / f"agRun-{period.name}-{name}.parquet" for name in AG_RUN_TABLES},
        directory / f"agRun-{period.name}-fetched.json",
    )


def fetchRun(opdb, period: Period, dataDir: Path, magLimit: float = 20.0, now=None) -> dict[str, Path]:
    """Read a period's AG data from the opdb and keep it as parquet.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The opdb, read-only.
    period : `pfs.drp.qa.comparison.runs.Period`
        The period.
    dataDir : `pathlib.Path`
        Where to write, under ``opdb/``.
    magLimit : `float`
        Guide stars brighter than this count as bright (`readAgRun`).
    now : `datetime.datetime`, optional
        The current time (HST); a period still under way is read up to it.

    Returns
    -------
    paths : `dict` [`str`, `pathlib.Path`]
        The files written, by table.
    """
    from pfs.drp.qa.guiders.queries import readAgRun

    # Periods and opdb times are HST, whatever the machine's time zone.
    now = now or datetime.datetime.now(zoneinfo.ZoneInfo("Pacific/Honolulu")).replace(tzinfo=None)
    end = min(period.end, now)
    tables = readAgRun(opdb, period.start, end, magLimit=magLimit)
    paths, stampPath = _paths(period, dataDir)
    stampPath.parent.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(paths[name])
    stamp = {
        "period": period.name,
        "fetchedAt": now.isoformat(timespec="seconds"),
        "readUntil": str(end),
        "magLimit": magLimit,
        "rows": {name: len(table) for name, table in tables.items()},
    }
    stampPath.write_text(json.dumps(stamp, indent=1) + "\n")
    return paths


def loadRun(period: Period, dataDir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Read back what `fetchRun` wrote.

    Returns
    -------
    tables : `dict` [`str`, `pandas.DataFrame`]
        As `readAgRun` returned them.
    stamp : `dict`
        When and how they were read.

    Raises
    ------
    FileNotFoundError
        If the period hasn't been fetched.
    """
    paths, stampPath = _paths(period, dataDir)
    if not stampPath.exists():
        raise FileNotFoundError(f"{period.name} hasn't been fetched into {stampPath.parent}: run fetch first")
    return {name: pd.read_parquet(path) for name, path in paths.items()}, json.loads(stampPath.read_text())


def _table(frame: pd.DataFrame, fmt: str = "{:.2f}") -> list[str]:
    lines = ["| | " + " | ".join(str(c) for c in frame.columns) + " |", "|---|" + "---|" * len(frame.columns)]
    for index, row in frame.iterrows():
        cells = [
            str(int(v))
            if isinstance(v, int | float)
            and pd.notna(v)
            and float(v).is_integer()
            and "exposures with" in str(index)
            else fmt.format(v)
            if isinstance(v, int | float) and pd.notna(v)
            else str(v)
            for v in row
        ]
        lines.append(f"| {index} | " + " | ".join(cells) + " |")
    return lines


def writeReport(run: AgRun, outDir: Path, title: str, notes: str | None = None) -> Path:
    """Write a run's report: Markdown, its figures (PNG) and its numbers (JSON).

    Parameters
    ----------
    run : `pfs.drp.qa.guiders.runAnalysis.AgRun`
        The run.
    outDir : `pathlib.Path`
        Where to write ``report.md``, ``<figure>.png`` and ``metrics.json``.
    title : `str`
        The run's name, for the heading.
    notes : `str`, optional
        Markdown appended to the report.

    Returns
    -------
    path : `pathlib.Path`
        The report.
    """
    outDir.mkdir(parents=True, exist_ok=True)
    jitter = settledJitter(run)
    dropouts = findDropouts(run)
    clusters = findRejectionClusters(run)
    modelErrors = fitModelErrors(run)
    flags = findFlags(run, jitter, dropouts, clusters, modelErrors)
    groups = groupSummary(run)
    cameras = cameraSummary(run, clusters, modelErrors)
    for name, (_, plot) in FIGURES.items():
        plot(run, jitter).savefig(outDir / f"{name}.png", dpi=150)
    for cid in CAMERA_ORDER:
        runPlotting.plotCamera(run, cid, clusters, modelErrors).savefig(
            outDir / f"camera-AG{cid + 1}.png", dpi=150
        )

    config = run.config
    skipped = run.nights[run.nights < config.minFramesNight]
    populations = run.frames.groupby("population").agg(
        visits=("pfs_visit_id", "nunique"), exposures=("agc_exposure_id", "size")
    )
    md = [
        f"# {title}: AG run report",
        "",
        "1 AG pixel = 13 µm = 0.14″. Offsets in arcsec, ΔRA on the sky (ΔRA cos Dec). Times HST. "
        f"{describePeriod(run)}.",
        "",
        "## Flags",
        "",
        *([f"- {flag.text}" for flag in flags] or ["- None."]),
        "",
    ]
    for name, (heading, _) in FIGURES.items():
        md += [f"## {heading}", "", f"![{heading}]({name}.png)", ""]
        if name == "cameraPresence":
            md += ["### AG1/3/5 against AG2/4/6", "", *_table(groups), ""]
        if name == "guiding":
            md += [
                f"Settled: science visits' accepted guide offsets after their first {config.nSettle} guide frames, "
                f"in visits with at least {config.minSettled} ({len(jitter)} visits). Jitter: the standard "
                "deviation of consecutive differences / √2.",
                "",
            ]
    md += [
        "## Cameras",
        "",
        "In their two groups, AG1/3/5 then AG2/4/6. A rejected match is a detection paired with a guide star "
        "but not counted as a good match; when a camera's position model is off, those far from where it fits "
        "best go first, so they cluster on one side of its detector, always the same side. The position-model "
        "error is fitted from the good matches' residuals (measured minus expected position, less the camera's "
        "mean in each AG exposure, which the pointing moves together) across the detector; directions are on "
        "the detector, 0° = +x, 90° = +y.",
        "",
        *_table(cameras),
        "",
    ]
    for cid in CAMERA_ORDER:
        own = [flag for flag in flags if flag.agc_camera_id == cid]
        md += [f"### AG{cid + 1}", "", *([f"- {flag.text}" for flag in own] or ["- No flags."]), ""]
        md += [f"![AG{cid + 1}](camera-AG{cid + 1}.png)", ""]
    md += ["## AG exposures", "", *_table(populations.rename_axis(None), "{:.0f}"), ""]
    if len(skipped):
        md += [
            f"Not plotted (fewer than {config.minFramesNight} AG exposures): "
            + ", ".join(f"{night:%m-%d} ({n})" for night, n in skipped.items())
            + ".",
            "",
        ]
    if notes:
        md += [notes.rstrip(), ""]
    path = outDir / "report.md"
    path.write_text("\n".join(md))

    metrics = {
        "title": title,
        "period": describePeriod(run),
        "flags": [flag.__dict__ for flag in flags],
        "jitterMedian": {axis: float(jitter[axis].median()) for axis in ("ra", "dec", "inr")}
        if len(jitter)
        else {},
        "groups": json.loads(groups.to_json()),
        "cameras": json.loads(cameras.to_json()),
        "rejectionClusters": json.loads(
            clusters.groupby(["agc_camera_id", "alone"])
            .size()
            .rename("n")
            .reset_index()
            .to_json(orient="records")
        ),
        "modelErrors": json.loads(
            modelErrors.groupby("agc_camera_id")[["error_um", "angle_deg"]].median().to_json()
        ),
        "dropouts": json.loads(dropouts.to_json(orient="records", date_format="iso")),
    }
    (outDir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agRunReport.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["fetch", "report"])
    parser.add_argument("--period", required=True, help="e.g. run30")
    parser.add_argument(
        "--data-dir", default="~/.cache/drp_qa/agRun", help="where opdb reads and reports are kept"
    )
    parser.add_argument("--runs", default=None, help="the run table (default: comparison mode's)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="opdb host, for fetch (default: an ssh port-forward to pfsa-db; give pfsa-db at the summit)",
    )
    parser.add_argument("--port", type=int, default=5432, help="opdb port, for fetch")
    parser.add_argument("--user", default="public_user", help="opdb user (read-only), for fetch")
    parser.add_argument("--mag-limit", type=float, default=20.0, help="bright guide stars, for fetch")
    parser.add_argument("--title", default=None, help="the report's heading (default: the period)")
    parser.add_argument("--notes", default=None, help="a Markdown file to append to the report")
    parser.add_argument("--out", default=None, help="report directory (default: <data-dir>/<period>)")
    return parser


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

        opdb = OpDB(host=args.host, port=args.port, dbname="opdb", user=args.user)
        print(
            f"reading {period.name} ({period.start} to {period.end} HST) from {args.host}:{args.port} ...",
            flush=True,
        )
        paths = fetchRun(opdb, period, dataDir, magLimit=args.mag_limit)
        _, stamp = loadRun(period, dataDir)
        print(
            ", ".join(f"{name} {n}" for name, n in stamp["rows"].items())
            + f" rows, in {paths['cameras'].parent}"
        )
        return 0

    try:
        tables, stamp = loadRun(period, dataDir)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    run = prepareRun(tables, RunConfig())
    if run.frames.empty:
        raise SystemExit(f"{period.name}: no AG exposures were read")
    notes = Path(args.notes).read_text() if args.notes else None
    outDir = Path(args.out).expanduser() if args.out else dataDir / period.name
    path = writeReport(run, outDir, args.title or period.name, notes)
    print(f"{period.name}: {describePeriod(run)}; read {stamp['fetchedAt']}")
    print(f"wrote {path}")
    return 0
