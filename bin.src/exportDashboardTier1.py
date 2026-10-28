#!/usr/bin/env python3
r"""Build the QA dashboard's Tier 1 table from one collection's exported metrics.

Stack-free: reads the per-detector tables ``bin.src/exportMetrics.py`` wrote
(``iqQaMetrics``, ``dmLineResidualSummary``, ``dmLineResidualMetrics``) and,
optionally, opdb's visit listing for times and sequences. Writes one Parquet
file for ``pfs.drp.qa.dashboard`` (PIPE2D-1924) with its provenance.

Example::

    python bin.src/exportDashboardTier1.py --collection u/wtg/qa-thresholds/010 \\
        --dir ~/Projects/Subaru/PFS/reports/PIPE2D-1938 -o tier1-qa-thresholds-010.parquet

reads ``<datasetType>-u_wtg_qa-thresholds_010.parquet`` from ``--dir``. Several collections, with
distinct images, are combined into one table.
"""

import argparse
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pfs.drp.qa.comparison.cli import drpQaVersion
from pfs.drp.qa.dashboard.serving import (
    buildTier1,
    firstThresholdsFile,
    taskThresholds,
    thresholdsProvenance,
    verdictDisagreements,
    writeTier1,
)
from pfs.drp.qa.metrics.gate import thresholdsPath


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the QA dashboard's Tier 1 table.")
    parser.add_argument(
        "-c", "--collection", nargs="+", required=True, help="the collections the tables came from, combined"
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        nargs="+",
        default=[Path(".")],
        help="directories holding the exported tables",
    )
    parser.add_argument(
        "-t",
        "--thresholds",
        help="a thresholds file judged before imageQualityQa's own: a path, else a name in the package's "
        "thresholds data (default: none)",
    )
    parser.add_argument("--visit-times", type=Path, nargs="+", help="opdb visit listings (parquet)")
    parser.add_argument(
        "--notes", type=Path, nargs="+", help="opdb notes, comparison.queries.readNotes (parquet)"
    )
    parser.add_argument("-o", "--out", type=Path, required=True, help="file to write")
    args = parser.parse_args(argv)

    def read(datasetType: str, required: bool = True) -> pd.DataFrame | None:
        tables = []
        for collection in args.collection:
            name = f"{datasetType}-{collection.replace('/', '_')}.parquet"
            path = next((directory / name for directory in args.dir if (directory / name).exists()), None)
            if path is None:
                if required:
                    raise SystemExit(f"{name} not in {args.dir}: export it with bin.src/exportMetrics.py")
                print(f"{datasetType}: none for {collection}", flush=True)
                continue
            tables.append(pd.read_parquet(path))
            print(f"{datasetType}: {len(tables[-1])} rows from {path}", flush=True)
        return pd.concat(tables, ignore_index=True) if tables else None

    def readAll(paths: list[Path] | None) -> pd.DataFrame | None:
        return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True) if paths else None

    start = time.monotonic()
    iq = read("iqQaMetrics")
    summary, metrics = read("dmLineResidualSummary", False), read("dmLineResidualMetrics", False)
    visitTimes = readAll(args.visit_times)
    if visitTimes is not None:
        visitTimes = visitTimes.drop_duplicates("pfs_visit_id")
    notes = readAll(args.notes)
    if iq.duplicated(["visit", "arm", "spectrograph"]).any():
        raise SystemExit("An image is in more than one collection: give collections with distinct visits")

    layers, sources = taskThresholds()
    if args.thresholds:
        given = Path(args.thresholds)
        layers["iqQaMetrics"][0:0] = [given.resolve() if given.exists() else thresholdsPath(args.thresholds)]
        sources["iqQaMetrics"] = f"{args.thresholds} + " + sources["iqQaMetrics"]
    for key, source in sources.items():
        print(f"{key} thresholds: {source}", flush=True)

    table = buildTier1(
        iq, layers["iqQaMetrics"], summary, metrics, visitTimes, layers["dmLineResidualMetrics"], notes
    )
    disagree = verdictDisagreements(table)
    provenance = {
        "collection": " + ".join(args.collection),
        "collections": args.collection,
        "thresholds": thresholdsProvenance(firstThresholdsFile(layers["iqQaMetrics"])),
        "thresholdLayers": [f"{key}: {source}" for key, source in sources.items()],
        "drpQaVersion": drpQaVersion(),
        "builtAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "visitTimes": [str(path) for path in args.visit_times or []],
        "notes": [str(path) for path in args.notes or []],
        "nImages": len(iq),
        "nVerdictDisagreements": len(disagree),
    }
    writeTier1(table, args.out, provenance)
    print(
        f"{len(table)} rows, {len(disagree)} stored verdicts differ from the thresholds -> {args.out} "
        f"({args.out.stat().st_size / 1e6:.2f} MB) in {time.monotonic() - start:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
