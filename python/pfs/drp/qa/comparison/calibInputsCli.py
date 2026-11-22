r"""``drpQa.py calib inputs``: report which exposures of a calibration set to build the calibrations from, per detector.

Reads the period's visits from the opdb (``--opdb-host``), takes the set's arcs and traces (cadence
``set``, or ``--visits``), reads what a redo (or any collection) measured on them (``iqQaMetrics``,
``dmLineResidualMetrics``) from its Butler output collection, and writes the Markdown report of
`pfs.drp.qa.comparison.calibInputs`: a go/no-go for every detector and visit, as a grid, a table and a
CSV, with the reasons and the lamp trends behind ``--detail``. Blind: it uses no calibration built from the set.

The stored tables hold numbers, not verdicts: the gate (`pfs.drp.qa.metrics.gate`) judges them here, against
the table of normal values ``--normal-values`` gives (those in force at each visit). Every gate check of a
calibration frame is relative to a normal value, so without the table the gate gives no verdict: the report
says so, the go/no-go rests on the sequence rules alone, and the figures show no gate limit.

    drpQa.py calib inputs --period run31 --opdb-host pfsa-db \\
        --collection u/wtg/redo/run31/w.2026.35-259-g83ef4f3 -o calibInputs-run31.md
"""

import argparse
import datetime
from pathlib import Path

import pandas as pd

from pfs.drp.qa.comparison.calibInputs import (
    Rule,
    Thresholds,
    buildReport,
    detectorTable,
    gateNumbers,
    goNoGo,
    judge,
    scatterLimits,
    setVisits,
    speciesTable,
)
from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.runs import loadPeriods
from pfs.drp.qa.metrics.normalValues import NORMAL_COLUMNS
from pfs.drp.qa.metrics.registry import ARC, IIS_ARC

TABLES = ("iqQaMetrics", "dmLineResidualMetrics")


def readNormalValues(path: str | None) -> pd.DataFrame | None:
    """Read a table of normal values (`NORMAL_COLUMNS`) from parquet or CSV; None without a path."""
    if not path:
        return None
    table = pd.read_csv(path) if Path(path).suffix == ".csv" else pd.read_parquet(path)
    missing = [column for column in NORMAL_COLUMNS if column not in table.columns]
    if missing:
        raise SystemExit(f"{path} is not a table of normal values: no {', '.join(missing)}")
    return table


def readTables(args, visits: list[int]) -> dict[str, pd.DataFrame]:
    """Read the per-detector tables (`TABLES`) for ``visits`` from the Butler collection."""
    from lsst.daf.butler import Butler

    from pfs.drp.qa.metrics.readers import readMetrics

    butler = Butler(args.butler, writeable=False)
    return {
        name: readMetrics(butler, visits, datasetType=name, collections=[args.collection], progress=True)
        for name in TABLES
    }


def readResiduals(args, images: pd.DataFrame) -> pd.DataFrame | None:
    """Read ``dmLineResidualData`` for the no-go images from the Butler; None when there are none."""
    if images.empty:
        return None
    from lsst.daf.butler import Butler

    from pfs.drp.qa.metrics.readers import readMetrics

    butler = Butler(args.butler, writeable=False)
    visits = sorted({int(v) for v in images["visit"]})
    data = readMetrics(
        butler, visits, datasetType="dmLineResidualData", collections=[args.collection], progress=True
    )
    return data.merge(
        images[["visit", "arm", "spectrograph"]].drop_duplicates(), on=["visit", "arm", "spectrograph"]
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--period", required=True, help="e.g. run31")
    parser.add_argument("--runs", default=None, help="the run table (default: the packaged one)")
    parser.add_argument(
        "--visits",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="the set's visit range (default: every set-cadence arc and trace of the period)",
    )
    parser.add_argument("--collection", required=True, help="the Butler output collection to read")
    parser.add_argument("--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument("--opdb-host", default="pfsa-db", help="opdb host, for the visit listing")
    parser.add_argument("--opdb-port", type=int, default=5432, help="opdb port")
    parser.add_argument("--opdb-dbname", default="opdb", help="opdb database name")
    parser.add_argument("--title", default=None, help="report title (default: the period)")
    parser.add_argument("--weak-lines", type=float, default=Thresholds.weakLines)
    parser.add_argument("--fwhm-check", type=float, default=Thresholds.fwhmCheck)
    parser.add_argument("--fwhm-out", type=float, default=Thresholds.fwhmOut)
    parser.add_argument("--rms-check", type=float, default=Thresholds.rmsCheck)
    parser.add_argument("--rms-out", type=float, default=Thresholds.rmsOut)
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="the Markdown report; its tables (CSV) and figures (PNG) go beside it",
    )
    parser.add_argument("--no-plots", action="store_true", help="write no figures")
    parser.add_argument(
        "--normal-values",
        default=None,
        help="the table of normal values (parquet or CSV, the columns of qadb's qa_normal_value) the gate "
        "judges by; without it the gate gives no verdict, and the go/no-go rests on the sequence rules alone",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="add the reasons, weak exposures, lamp trends and the detail figures",
    )
    args = parser.parse_args(argv)

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
        chosen = classified[classified["cadence"] == "set"]
    visits = setVisits(chosen)
    if visits.empty:
        raise SystemExit("no set arcs or traces among the chosen visits")
    print(
        f"{len(visits)} set visits in {len(visits['iic_sequence_id'].unique())} sequences: "
        f"{visits['visit'].min()}–{visits['visit'].max()}",
        flush=True,
    )

    tables = readTables(args, [int(v) for v in visits["visit"]])
    normalValues = readNormalValues(args.normal_values)
    thresholds = Thresholds(args.weak_lines, args.fwhm_check, args.fwhm_out, args.rms_check, args.rms_out)
    if normalValues is None:
        print(
            "no --normal-values: the gate gives no verdict; judging on the sequence rules alone", flush=True
        )
    judged = judge(
        detectorTable(visits, tables["iqQaMetrics"], tables["dmLineResidualMetrics"], normalValues),
        thresholds,
    )
    numbers = gateNumbers(tables["iqQaMetrics"], tables["dmLineResidualMetrics"], normalValues)
    species = speciesTable(visits, tables["dmLineResidualMetrics"])
    out = Path(args.output)
    figures, detectorFigures, imageFigures = {}, {}, {}
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        from pfs.drp.qa.comparison.calibInputs import trends
        from pfs.drp.qa.plotting.calibInputs import (
            plotDetectorSeries,
            plotGoNoGoGrid,
            plotLampTrends,
            plotVerdictGrid,
        )

        plots = [("gonogo", "Go / no-go per detector and visit", lambda: plotGoNoGoGrid(judged, visits))]
        if args.detail:
            plots += [
                ("verdicts", "Verdict per detector image", lambda: plotVerdictGrid(judged, visits)),
                (
                    "detectors",
                    "FWHM and reserved RMS of each detector through the set",
                    lambda: plotDetectorSeries(judged),
                ),
                ("lamps", "Lamps through their sequences", lambda: plotLampTrends(trends(judged, species))),
            ]
        for name, caption, make in plots:
            path = out.with_name(f"{out.stem}-{name}.png")
            make().savefig(path, dpi=130, bbox_inches="tight")
            figures[caption] = path.name
        from pfs.drp.qa.comparison.calibInputs import Verdict
        from pfs.drp.qa.plotting.calibInputs import plotDetectorDetail

        noGo = judged.loc[judged["verdict"] == Verdict.OUT, "detector"].unique()
        for det in sorted(noGo, key=lambda d: ("brmn".index(d[0]) if d[0] in "brmn" else 9, d)):
            path = out.with_name(f"{out.stem}-{det}.png")
            plotDetectorDetail(judged, species, visits, det, numbers=numbers).savefig(
                path, dpi=130, bbox_inches="tight"
            )
            detectorFigures[det] = path.name
        residuals = readResiduals(args, judged[judged["verdict"] == Verdict.OUT])
        if residuals is not None:
            from pfs.drp.qa.plotting.calibInputs import plotLineResiduals

            for r in (
                judged[judged["verdict"] == Verdict.OUT]
                .sort_values(["arm", "spectrograph", "visit"])
                .itertuples()
            ):
                onScatter = {"scatterX", "scatterY"} & set(r.gateChecks.split(","))
                onSequenceRms = Rule.RMS in r.rules.split(", ")
                if not onScatter and not onSequenceRms:
                    continue  # out on FWHM or lines per fiber alone: the residuals aren't the reason
                lines = residuals[
                    (residuals["visit"] == r.visit)
                    & (residuals["arm"] == r.arm)
                    & (residuals["spectrograph"] == r.spectrograph)
                ]
                if lines.empty:
                    continue
                path = out.with_name(f"{out.stem}-{r.detector}-{r.visit}-lines.png")
                # The gate's scatter limits, relative to the normal values in force at the visit; none
                # without the table, or on a frame lit through the engineering fibers.
                limits = scatterLimits(
                    r.arm,
                    r.spectrograph,
                    sorted(lines["description"].unique()),
                    r.visit,
                    normalValues,
                    frameType=IIS_ARC if "(IIS)" in str(r.lamp) else ARC,
                )
                if onSequenceRms:
                    # Out on the sequence rule: show its limits, the detector's median fit RMS over the
                    # sequence times the check and out factors, per axis.
                    same = judged[
                        (judged["detector"] == r.detector) & (judged["iic_sequence_id"] == r.iic_sequence_id)
                    ]
                    medianX = float(same["fitReservedXRms"].median())
                    medianY = float(same["fitReservedYRms"].median())
                    limits["sequence"] = {
                        "robustRmsDeltaX": (medianX * thresholds.rmsCheck, medianX * thresholds.rmsOut),
                        "robustRmsDeltaY": (medianY * thresholds.rmsCheck, medianY * thresholds.rmsOut),
                    }
                plotLineResiduals(
                    lines,
                    title=f"{r.detector} {r.visit} ({r.lamp} {r.k}/{r.n}): line residuals",
                    limits=limits,
                ).savefig(path, dpi=120, bbox_inches="tight")
                imageFigures[(r.detector, int(r.visit))] = path.name
    report = buildReport(
        visits,
        judged,
        species,
        args.title or period.name,
        source=args.collection,
        exports=False,
        detectorFigures=detectorFigures,
        imageFigures=imageFigures,
        thresholds=thresholds,
        figures=figures,
        detail=args.detail,
        normalValues=Path(args.normal_values).name if args.normal_values else "",
    )
    out.write_text(report)
    goNoGo(judged, visits).to_csv(out.with_name(out.stem + "-gonogo.csv"))
    judged.to_csv(out.with_name(out.stem + "-detectors.csv"), index=False)
    numbers.to_csv(out.with_name(out.stem + "-gate.csv"), index=False)
    species.to_csv(out.with_name(out.stem + "-species.csv"), index=False)
    counts = judged["verdict"].value_counts().to_dict()
    print(f"wrote {out}: {len(judged)} detector images, {counts}", flush=True)
    return 0
