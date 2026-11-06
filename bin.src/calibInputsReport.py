#!/usr/bin/env python3
r"""Report which exposures of a calibration set to build the calibrations from, per detector.

Reads comparison mode's visit listing for the period (``fetch``), takes the set's arcs and traces
(cadence ``set``, or ``--visits``), reads what comparison mode measured on them (``iqQaMetrics``,
``dmLineResidualMetrics``, ``dmLineResidualSummary``) from its output collection, or from
``exportMetrics.py`` exports with ``--exports``, and writes the Markdown report of
`pfs.drp.qa.comparison.calibInputs`: a go/no-go for every detector and visit, as a grid, a table and a
CSV, with the reasons and the lamp trends behind ``--detail``. Blind: it uses no calibration built from the set.

    python bin.src/calibInputsReport.py --period run31 --data-dir ~/qa-comparison \\
        --collection u/wtg/comparison/run31/w.2026.35-259-g83ef4f3 -o calibInputs-run31.md
    python bin.src/calibInputsReport.py --period run31 --data-dir ~/qa-comparison --exports ./export \\
        --visits 152652 152705 -o calibInputs-run31.md
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from pfs.drp.qa.comparison.calibInputs import (
    Thresholds,
    buildReport,
    detectorTable,
    goNoGo,
    judge,
    lineThresholds,
    setVisits,
    speciesTable,
)
from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.runs import loadPeriods

TABLES = ("iqQaMetrics", "dmLineResidualMetrics", "dmLineResidualSummary")


def readTables(args, visits: list[int]) -> dict[str, pd.DataFrame]:
    """Read the three per-detector tables for ``visits``, from the exports or the Butler."""
    if args.exports:
        tables = {}
        for name in TABLES:
            files = sorted(Path(args.exports).glob(f"{name}-*.parquet"))
            if not files:
                raise SystemExit(f"no {name}-*.parquet in {args.exports}")
            frame = pd.read_parquet(files[-1])
            tables[name] = frame[frame["visit"].isin(visits)]
            # exportMetrics.py names the file <datasetType>-<collection with / as _>.parquet
            tables["source"] = files[-1].stem[len(name) + 1 :]
        return tables
    from lsst.daf.butler import Butler

    from pfs.drp.qa.metrics.readers import readMetrics

    butler = Butler(args.butler, writeable=False)
    return {
        name: readMetrics(butler, visits, datasetType=name, collections=[args.collection], progress=True)
        for name in TABLES
    }


def readResiduals(args, images: pd.DataFrame) -> pd.DataFrame | None:
    """Read ``dmLineResidualData`` for the no-go images: from ``--residuals`` parquet files, or the Butler.

    None when there is no source (exports without ``--residuals``).
    """
    if args.residuals:
        frames = [pd.read_parquet(f) for f in args.residuals]
        return pd.concat(frames, ignore_index=True) if frames else None
    if not args.collection or images.empty:
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
    parser.add_argument(
        "--data-dir", required=True, help="comparison mode's data directory (fetch wrote opdb/ there)"
    )
    parser.add_argument("--runs", default=None, help="the run table (default: the packaged one)")
    parser.add_argument(
        "--visits",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="the set's visit range (default: every set-cadence arc and trace of the period)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collection", help="comparison mode's output collection to read")
    source.add_argument("--exports", help="directory of exportMetrics.py parquet files instead of a Butler")
    parser.add_argument("--butler", default="/work/datastore", help="Butler repository")
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
        "--residuals",
        nargs="*",
        default=None,
        help="dmLineResidualData parquet file(s) for the no-go images' line-residual figures (with --exports; "
        "with --collection they are read from the Butler)",
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
    listingPath = Path(args.data_dir) / "opdb" / f"comparison-{period.name}-listing.parquet"
    if not listingPath.exists():
        raise SystemExit(f"{listingPath} not found: run qaComparison.py fetch --period {period.name} first")
    listing = pd.read_parquet(listingPath)
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
    thresholds = Thresholds(args.weak_lines, args.fwhm_check, args.fwhm_out, args.rms_check, args.rms_out)
    judged = judge(
        detectorTable(
            visits, tables["iqQaMetrics"], tables["dmLineResidualMetrics"], tables["dmLineResidualSummary"]
        ),
        thresholds,
    )
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
            plotDetectorDetail(judged, species, visits, det).savefig(path, dpi=130, bbox_inches="tight")
            detectorFigures[det] = path.name
        residuals = readResiduals(args, judged[judged["verdict"] == Verdict.OUT])
        if residuals is not None:
            from pfs.drp.qa.plotting.calibInputs import plotLineResiduals

            for r in (
                judged[judged["verdict"] == Verdict.OUT]
                .sort_values(["arm", "spectrograph", "visit"])
                .itertuples()
            ):
                if "line residuals" not in r.reason and "reserved RMS" not in r.reason:
                    continue  # out on FWHM or lines per fiber alone: the residuals aren't the reason
                lines = residuals[
                    (residuals["visit"] == r.visit)
                    & (residuals["arm"] == r.arm)
                    & (residuals["spectrograph"] == r.spectrograph)
                ]
                if lines.empty:
                    continue
                path = out.with_name(f"{out.stem}-{r.detector}-{r.visit}-lines.png")
                summaryRow = tables["dmLineResidualSummary"]
                summaryRow = summaryRow[
                    (summaryRow["visit"] == r.visit)
                    & (summaryRow["arm"] == r.arm)
                    & (summaryRow["spectrograph"] == r.spectrograph)
                ]
                limits = {}
                if len(summaryRow):
                    sr = summaryRow.iloc[0]
                    named = (
                        str(sr["thresholdsFiles"])
                        if "thresholdsFiles" in sr.index and sr["thresholdsFiles"]
                        else ""
                    )
                    limits = lineThresholds(
                        r.arm,
                        r.spectrograph,
                        sorted(lines["description"].unique()),
                        sr.get("cameraSet"),
                        sr.get("obsType"),
                        named.split(",")[0].strip() or None,
                    )
                if "reserved RMS" in r.reason:
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
                    thresholds=limits,
                ).savefig(path, dpi=120, bbox_inches="tight")
                imageFigures[(r.detector, int(r.visit))] = path.name
    report = buildReport(
        visits,
        judged,
        species,
        args.title or period.name,
        source=args.collection or tables.get("source", args.exports),
        exports=bool(args.exports),
        detectorFigures=detectorFigures,
        imageFigures=imageFigures,
        thresholds=thresholds,
        figures=figures,
        detail=args.detail,
    )
    out.write_text(report)
    goNoGo(judged, visits).to_csv(out.with_name(out.stem + "-gonogo.csv"))
    judged.to_csv(out.with_name(out.stem + "-detectors.csv"), index=False)
    species.to_csv(out.with_name(out.stem + "-species.csv"), index=False)
    counts = judged["verdict"].value_counts().to_dict()
    print(f"wrote {out}: {len(judged)} detector images, {counts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
