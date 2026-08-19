#!/usr/bin/env python3
"""Plot imageQualityQa metrics across visits.

Reads ``iqQaMetrics`` datasets from a butler collection and produces a
multi-panel time-series figure showing FWHM, flexure (dxCenter), and
pass/fail status.

Example::

    python bin.src/plotIqQaTimeSeries.py \\
        -b /path/to/butler \\
        -c u/wtg/output/collection \\
        -o iq_timeseries.png

    # Filter to a single arm:
    python bin.src/plotIqQaTimeSeries.py \\
        -b /path/to/butler \\
        -c u/wtg/output/collection \\
        -w "arm='r'"

Requires the LSST stack (``lsst.daf.butler``) and ``matplotlib``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Plot imageQualityQa metrics across visits.",
    )
    parser.add_argument(
        "-b", "--butler",
        help="Path to the butler repository (required unless --csv is provided).",
    )
    parser.add_argument(
        "-c", "--collection",
        help="Butler collection containing iqQaMetrics datasets (required unless --csv is provided).",
    )
    parser.add_argument(
        "--csv",
        help="Path to a CSV file containing concatenated iqQaMetrics rows (bypasses Butler query).",
    )
    parser.add_argument(
        "--arm",
        help="Filter to specific arms (comma-separated, e.g. 'b,r').",
    )
    parser.add_argument(
        "--spectrograph", "--spec",
        help="Filter to specific spectrographs (comma-separated integers, e.g. '1,3').",
    )
    parser.add_argument(
        "--obs-type", "--obsType",
        help="Filter to specific observation types (comma-separated, e.g. 'arc,trace').",
    )
    parser.add_argument(
        "-o", "--output", default="iq_qa_timeseries.png",
        help="Output file path (default: iq_qa_timeseries.png).",
    )
    parser.add_argument(
        "-w", "--where", default="",
        help=(
            "Optional butler query expression to filter quanta"
            " (e.g. \"arm='r'\" or \"visit > 140000\"). Only used with Butler query."
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output resolution in DPI (default: 150).",
    )
    args = parser.parse_args()

    if not args.csv and (not args.butler or not args.collection):
        parser.error("Either --csv or both --butler and --collection must be specified.")

    if args.csv:
        print(f"Loading metrics from CSV: {args.csv}")
        metrics = pd.read_csv(args.csv)
        print(
            f"Loaded {len(metrics)} rows spanning"
            f" {metrics['visit'].nunique()} visits."
        )
    else:
        try:
            from lsst.daf.butler import Butler
        except ImportError:
            print(
                "Error: lsst.daf.butler not available."
                "  Run this script within the LSST stack environment.",
                file=sys.stderr,
            )
            sys.exit(1)

        butler = Butler(args.butler, collections=[args.collection])
        refs = set(
            butler.registry.queryDatasets("iqQaMetrics", where=args.where)
        )
        if not refs:
            print(
                f"No iqQaMetrics datasets found in collection"
                f" '{args.collection}'"
                + (f" with where='{args.where}'" if args.where else "")
                + ".",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Found {len(refs)} iqQaMetrics datasets.  Loading...")
        frames = []
        for ref in refs:
            df = butler.get(ref)
            frames.append(df)
        metrics = pd.concat(frames, ignore_index=True)
        print(
            f"Loaded {len(metrics)} rows spanning"
            f" {metrics['visit'].nunique()} visits."
        )

    # Apply command-line filters (arm / spectrograph)
    if args.arm:
        arms_to_keep = [a.strip() for a in args.arm.split(",")]
        metrics = metrics[metrics["arm"].isin(arms_to_keep)]
        print(f"Filtered to arm(s): {arms_to_keep} ({len(metrics)} rows remaining)")

    if args.spectrograph:
        try:
            specs_to_keep = [int(s.strip()) for s in args.spectrograph.split(",")]
            metrics = metrics[metrics["spectrograph"].isin(specs_to_keep)]
            print(f"Filtered to spectrograph(s): {specs_to_keep} ({len(metrics)} rows remaining)")
        except ValueError:
            parser.error("--spectrograph must be a comma-separated list of integers.")

    if args.obs_type:
        obs_types_to_keep = [o.strip() for o in args.obs_type.split(",")]
        metrics = metrics[metrics["obsType"].isin(obs_types_to_keep)]
        print(f"Filtered to obsType(s): {obs_types_to_keep} ({len(metrics)} rows remaining)")

    from pfs.drp.qa.iqQaPlots import plotIqTimeSeries

    # Construct identifying title
    source = args.csv if args.csv else args.collection
    title_parts = [f"Image Quality QA: {source}"]
    filters = []
    if args.arm:
        filters.append(f"arms=[{args.arm}]")
    if args.spectrograph:
        filters.append(f"specs=[{args.spectrograph}]")
    if args.obs_type:
        filters.append(f"obsType=[{args.obs_type}]")
    if args.where:
        filters.append(f"query=[{args.where}]")
    if filters:
        title_parts.append(f"({', '.join(filters)})")
    title_str = " ".join(title_parts)

    fig = plotIqTimeSeries(metrics, title=title_str)
    outPath = Path(args.output)
    fig.savefig(outPath, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved to {outPath}")


if __name__ == "__main__":
    main()
