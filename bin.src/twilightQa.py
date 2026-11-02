#!/usr/bin/env python3
r"""Judge the twilight sequences of a collection: which exposures are good flats (PIPE2D-1973).

Reads the ``twilightFiberFlux`` tables ``drpQA.yaml#twilightFluxQa`` stored (or a parquet export of
them, ``bin.src/exportMetrics.py -t twilightFiberFlux``), compares each sequence's exposures with
`pfs.drp.qa.metrics.twilight.sequenceConsistency`, and judges them against the provisional thresholds.

Example::

    python bin.src/twilightQa.py -c u/wtg/twilight/run30 --visits 148039 150896 -o .
    python bin.src/twilightQa.py --parquet twilightFiberFlux-u_wtg_twilight_run30.parquet -o .

writes ``twilightMetrics.parquet`` (one row per detector image) and ``twilightVerdict.csv`` (one row per
exposure and arm: ``PASS``, ``WARN``, ``FAIL`` or ``TOO_FAINT``, with the reasons), and prints the verdicts
per sequence. Reading from a Butler requires the LSST stack (``lsst.daf.butler``); a parquet does not.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pfs.drp.qa.metrics.twilight import sequenceConsistency, twilightVerdict


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Judge twilight sequences: which exposures are good flats.")
    parser.add_argument("-b", "--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument("-c", "--collection", help="collection holding twilightFiberFlux")
    parser.add_argument("--visits", type=int, nargs=2, metavar=("FIRST", "LAST"), help="visit range to read")
    parser.add_argument("--parquet", type=Path, help="read twilightFiberFlux from this file instead")
    parser.add_argument("--min-snr", type=float, default=15.0, help="brightness floor, b-arm S/N per pixel")
    parser.add_argument("-o", "--output", type=Path, default=Path("."), help="output directory")
    args = parser.parse_args(argv)

    if args.parquet is not None:
        fibers = pd.read_parquet(args.parquet)
    else:
        if args.collection is None or args.visits is None:
            parser.error("give --collection and --visits, or --parquet")
        from lsst.daf.butler import Butler

        from pfs.drp.qa.metrics.readers import readMetrics

        butler = Butler(args.butler, collections=args.collection, writeable=False)
        fibers = readMetrics(
            butler, datasetType="twilightFiberFlux", visitRange=tuple(args.visits), progress=True
        )
    unknown = fibers["sequence"] < 0
    if unknown.any():
        print(f"Skipping visits without W_SEQID: {sorted(fibers.loc[unknown, 'visit'].unique())}", flush=True)
        fibers = fibers[~unknown]
    print(
        f"Judging {fibers['visit'].nunique()} exposures in {fibers['sequence'].nunique()} sequences",
        flush=True,
    )

    metrics = sequenceConsistency(fibers, minSnr=args.min_snr)
    verdict = twilightVerdict(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.output / "twilightMetrics.parquet", index=False)
    verdict.to_csv(args.output / "twilightVerdict.csv", index=False)

    counts = verdict.groupby(["sequence", "arm"])["status"].value_counts().unstack(fill_value=0)
    print(counts.to_string())
    bad = verdict[verdict["status"].isin(["WARN", "FAIL"])]
    if len(bad):
        print(bad[["visit", "arm", "status", "reason"]].to_string(index=False))
    print(f"Wrote {args.output / 'twilightMetrics.parquet'} and {args.output / 'twilightVerdict.csv'}")


if __name__ == "__main__":
    main()
