#!/usr/bin/env python3
r"""Judge the twilight sequences of a collection: which exposures are good flats (PIPE2D-1973).

Reads the ``twilightFiberFlux`` tables ``drpQA.yaml#twilightFluxQa`` stored (or a parquet export of
them, ``bin.src/exportMetrics.py -t twilightFiberFlux``), compares each sequence's exposures with
`pfs.drp.qa.metrics.twilight.sequenceConsistency`, and judges them against the provisional thresholds.

Example::

    python bin.src/twilightQa.py -c u/wtg/twilight/run30 --visits 148039 150896 -o .
    python bin.src/twilightQa.py --parquet twilightFiberFlux-u_wtg_twilight_run30.parquet -o .
    python bin.src/twilightQa.py -c qaActor/reductions --sequence-of-visit 150844 --opdb-host pfsa-db \
        --qadb-host pfsa-db

prints the verdicts per sequence. With ``-o`` it writes ``twilightMetrics.parquet`` (one row per detector
image) and ``twilightVerdict.csv`` (one row per exposure and arm: ``PASS``, ``WARN``, ``FAIL`` or
``TOO_FAINT``, with the reasons); with ``--qadb-host`` it writes both to qadb's ``twilight_qa`` and
``twilight_qa_metric`` (`pfs.drp.qa.qadb.twilight`, INSTRM-3089), replacing the judged visits' rows.
Reading from a Butler requires the LSST stack (``lsst.daf.butler``); a parquet does not.

``--sequence-of-visit`` is for running after every visit (``ics_qaActor``, INSTRM-3089): it looks the
visit's IIC sequence up in the opdb (read-only user) and judges it only when it is a twilight, it has
finished (``iic_sequence_status``), and every visit of it has a ``twilightFiberFlux`` for every camera it
has a pfsArm for; otherwise it says what is missing and does nothing. The sequence id comes from the opdb,
so raws without ``W_SEQID`` need no ``--sequences``. Its files, with ``-o``, are named by sequence:
``twilightMetrics-<sequence>.parquet`` and ``twilightVerdict-<sequence>.csv``.

The exit status says what happened, for the caller's log:

- `EXIT_JUDGED` (0): the sequence was judged, and written to qadb when asked;
- `EXIT_PENDING` (3): not yet: the sequence has not finished, or a visit's tables are missing;
- `EXIT_NOT_TWILIGHT` (4): the visit is in no IIC sequence, or its sequence is not a twilight;
- `EXIT_NOT_WRITTEN` (5): judged, but the qadb write failed: the verdicts and metrics are then printed
  in full, so the caller's log holds them.

The other modes exit `EXIT_JUDGED` or `EXIT_NOT_WRITTEN`.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pfs.drp.qa.metrics.twilight import (
    isTwilightSequence,
    sequenceConsistency,
    sequenceReadiness,
    twilightVerdict,
)

#: Exit statuses; see the module docstring.
EXIT_JUDGED = 0
EXIT_PENDING = 3
EXIT_NOT_TWILIGHT = 4
EXIT_NOT_WRITTEN = 5


def writeToQadb(
    args: argparse.Namespace, verdict: pd.DataFrame, metrics: pd.DataFrame, collection, opdb
) -> int:
    """Write the judgement to qadb; on any failure print it in full and return `EXIT_NOT_WRITTEN`."""
    from pfs.drp.qa.qadb.twilight import writeTwilight
    from pfs.utils.database.qadb import QaDB

    try:
        qadb = QaDB(host=args.qadb_host, port=args.qadb_port, user=args.qadb_user, dbname=args.qadb_dbname)
        writeTwilight(qadb, verdict, metrics, collection, opdb=opdb, log=lambda line: print(line, flush=True))
    except Exception as error:
        print(
            f"qadb write failed ({args.qadb_host}:{args.qadb_port}/{args.qadb_dbname}): {error!r}", flush=True
        )
        with pd.option_context(
            "display.width", 10_000, "display.max_columns", None, "display.max_rows", None
        ):
            print("Not written, twilightVerdict:", flush=True)
            print(verdict.to_string(index=False), flush=True)
            print("Not written, twilightMetrics:", flush=True)
            print(metrics.to_string(index=False), flush=True)
        return EXIT_NOT_WRITTEN
    return EXIT_JUDGED


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge twilight sequences: which exposures are good flats.")
    parser.add_argument("-b", "--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument("-c", "--collection", help="collection holding twilightFiberFlux")
    parser.add_argument("--visits", type=int, nargs=2, metavar=("FIRST", "LAST"), help="visit range to read")
    parser.add_argument("--parquet", type=Path, help="read twilightFiberFlux from this file instead")
    parser.add_argument(
        "--sequences",
        type=Path,
        help="CSV of visit, sequence: for visits whose headers lack W_SEQID (raws before 2025-11)",
    )
    parser.add_argument(
        "--sequence-of-visit",
        type=int,
        metavar="VISIT",
        help="judge VISIT's sequence if it is a finished twilight with every visit's tables stored",
    )
    parser.add_argument("--opdb-host", help="opdb host (read-only user): needed by --sequence-of-visit")
    parser.add_argument("--opdb-port", type=int, default=5432, help="opdb port (default 5432)")
    parser.add_argument("--qadb-host", help="write the verdicts and metrics to the qadb on this host")
    parser.add_argument("--qadb-port", type=int, default=5436, help="qadb port (default 5436)")
    parser.add_argument(
        "--qadb-user", default="pfs", help="qadb user; must be allowed to write (default pfs)"
    )
    parser.add_argument("--qadb-dbname", default="qadb", help="qadb database name (default qadb)")
    parser.add_argument("--min-snr", type=float, default=15.0, help="brightness floor, b-arm S/N per pixel")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="write the metrics and verdict files here"
    )
    args = parser.parse_args(argv)

    suffix = ""
    opdb = None
    collection = args.collection
    if args.opdb_host is not None:
        from pfs.utils.database.opdb import OpDB

        opdb = OpDB(host=args.opdb_host, port=args.opdb_port, dbname="opdb", user="public_user")
    if args.sequence_of_visit is not None:
        if args.collection is None or args.opdb_host is None:
            parser.error("--sequence-of-visit needs --collection and --opdb-host")
        from pfs.drp.qa.comparison.queries import readSequenceOfVisit

        sequence = readSequenceOfVisit(opdb, args.sequence_of_visit)
        if not isTwilightSequence(sequence):
            _, why = sequenceReadiness(sequence, {}, {})
            print(f"Visit {args.sequence_of_visit}: {why}", flush=True)
            return EXIT_NOT_TWILIGHT
        from lsst.daf.butler import Butler

        from pfs.drp.qa.metrics.readers import datasetDetectors, readMetrics

        visits = [int(visit) for visit in sequence["pfs_visit_id"]]
        butler = Butler(args.butler, collections=args.collection, writeable=False)
        reduced = datasetDetectors(butler, "pfsArm", visits)
        stored = datasetDetectors(butler, "twilightFiberFlux", visits)
        ready, why = sequenceReadiness(sequence, reduced, stored)
        print(f"Visit {args.sequence_of_visit}: {why}", flush=True)
        if not ready:
            return EXIT_PENDING
        fibers = readMetrics(butler, visits=visits, datasetType="twilightFiberFlux", progress=True)
        sequenceId = int(sequence["iic_sequence_id"].iloc[0])
        fibers["sequence"] = sequenceId
        suffix = f"-{sequenceId}"
    elif args.parquet is not None:
        fibers = pd.read_parquet(args.parquet)
        collection = collection or args.parquet.name
    else:
        if args.collection is None or args.visits is None:
            parser.error("give --collection and --visits, or --parquet")
        from lsst.daf.butler import Butler

        from pfs.drp.qa.metrics.readers import readMetrics

        butler = Butler(args.butler, collections=args.collection, writeable=False)
        fibers = readMetrics(
            butler, datasetType="twilightFiberFlux", visitRange=tuple(args.visits), progress=True
        )
    if args.sequences is not None:
        known = pd.read_csv(args.sequences).set_index("visit")["sequence"]
        missing = fibers["sequence"] < 0
        fibers.loc[missing, "sequence"] = fibers.loc[missing, "visit"].map(known).fillna(-1).astype(int)
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
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(args.output / f"twilightMetrics{suffix}.parquet", index=False)
        verdict.to_csv(args.output / f"twilightVerdict{suffix}.csv", index=False)
        print(
            f"Wrote {args.output / f'twilightMetrics{suffix}.parquet'} and"
            f" {args.output / f'twilightVerdict{suffix}.csv'}",
            flush=True,
        )

    if verdict.empty:
        print("Nothing measured: no sequence has enough exposures in an arm.", flush=True)
    else:
        counts = verdict.groupby(["sequence", "arm"])["status"].value_counts().unstack(fill_value=0)
        print(counts.to_string(), flush=True)
        bad = verdict[verdict["status"].isin(["WARN", "FAIL"])]
        if len(bad):
            print(bad[["visit", "arm", "status", "reason"]].to_string(index=False), flush=True)

    if args.qadb_host is not None:
        return writeToQadb(args, verdict, metrics, collection, opdb)
    return EXIT_JUDGED


if __name__ == "__main__":
    sys.exit(main())
