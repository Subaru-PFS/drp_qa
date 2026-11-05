#!/usr/bin/env python3
r"""Judge the twilight sequences of a collection: which exposures are good flats (PIPE2D-1973).

Reads the ``twilightFiberFlux`` tables ``drpQA.yaml#twilightFluxQa`` stored (or a parquet export of
them, ``bin.src/exportMetrics.py -t twilightFiberFlux``), compares each sequence's exposures with
`pfs.drp.qa.metrics.twilight.sequenceConsistency`, and judges them against the shipped thresholds
(``twilightThresholds.yaml``).

Example::

    python bin.src/twilightQa.py -c u/wtg/twilight/run30 --visits 148039 150896 -o .
    python bin.src/twilightQa.py --parquet twilightFiberFlux-u_wtg_twilight_run30.parquet -o .
    python bin.src/twilightQa.py -c qaActor/reductions --sequence-of-visit 150844 --opdb-host pfsa-db \
        --qadb-host pfsa-db

prints the verdicts per sequence, one per exposure and arm (``PASS``, ``WARN``, ``FAIL`` or ``TOO_FAINT``, with
the reasons). Read from a Butler, it also stores the verdict per detector image as ``twilightQaMetrics`` in a new
RUN prepended to the collection (`pfs.drp.qa.qadb.store`; ``--no-store`` skips it): the Butler is the record,
and the qadb loader folds the stored verdict into ``qa_gate`` as a third task, ``twilightQa``, on every load. With
``--qadb-host`` it then loads the judged visits into qadb itself (``twilight_qa`` and ``qa_gate``,
`pfs.drp.qa.qadb.loadCollection`; INSTRM-3089). With ``-o`` it writes ``twilightMetrics.parquet`` (one row per
detector image) and ``twilightVerdict.csv`` (one row per exposure and arm). Reading or storing in a Butler
requires the LSST stack (``lsst.daf.butler``); a parquet does not.

``--sequence-of-visit`` is for running after every visit (``ics_qaActor``, INSTRM-3089): it looks the
visit's IIC sequence up in the opdb (read-only user) and judges it only when it is a twilight, it has
finished (``iic_sequence_status``), and every visit of it has a ``twilightFiberFlux`` for every camera it
has a pfsArm for; otherwise it says what is missing and does nothing. The sequence id comes from the opdb,
so raws without ``W_SEQID`` need no ``--sequences``. Its files, with ``-o``, are named by sequence:
``twilightMetrics-<sequence>.parquet`` and ``twilightVerdict-<sequence>.csv``.

The exit status says what happened, for the caller's log:

- `EXIT_JUDGED` (0): the sequence was judged, stored, and loaded into qadb when asked, and every exposure is
  ``PASS`` or ``TOO_FAINT``;
- `EXIT_JUDGED_BAD` (6): as `EXIT_JUDGED`, but some exposure is ``WARN`` or ``FAIL``: the caller may want to raise
  it in its log;
- `EXIT_PENDING` (3): not yet: the sequence has not finished, or a visit's tables are missing;
- `EXIT_NOT_TWILIGHT` (4): the visit is in no IIC sequence, or its sequence is not a twilight;
- `EXIT_NOT_WRITTEN` (5): judged, but storing it or loading qadb failed: the verdicts and metrics are then
  printed in full, so the caller's log holds them.

The other modes exit `EXIT_JUDGED`, `EXIT_JUDGED_BAD` or `EXIT_NOT_WRITTEN`.
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
    twilightImageVerdict,
    twilightVerdict,
)

#: Exit statuses; see the module docstring.
EXIT_JUDGED = 0
EXIT_PENDING = 3
EXIT_NOT_TWILIGHT = 4
EXIT_NOT_WRITTEN = 5
EXIT_JUDGED_BAD = 6

#: The per-image verdict's dataset type.
TWILIGHT_DATASET = "twilightQaMetrics"


def say(line: str) -> None:
    """Print a line, flushed: the caller may be streaming the output into a log."""
    print(line, flush=True)


def storeAndLoad(
    args: argparse.Namespace, images: pd.DataFrame, visits: list[int], collection: str, opdb
) -> None:
    """Store the per-image verdicts in the Butler, then load the judged visits into qadb when asked."""
    from lsst.daf.butler import Butler

    if not args.no_store:
        from pfs.drp.qa.qadb.store import storeImageTable

        writer = Butler(args.butler, writeable=True)
        storeImageTable(writer, collection, TWILIGHT_DATASET, images, log=say)
    if args.qadb_host is not None:
        from pfs.drp.qa.qadb.loader import loadCollection
        from pfs.utils.database.qadb import QaDB

        qadb = QaDB(host=args.qadb_host, port=args.qadb_port, user=args.qadb_user, dbname=args.qadb_dbname)
        # A fresh Butler: the one that read the fibers may hold the chain as it was before the store. The
        # reduction's versions are not reread from the pfsArm headers: the per-visit loads recorded them,
        # and the loader keeps a recorded version when a load does not read one.
        reader = Butler(args.butler, writeable=False)
        loadCollection(
            reader,
            qadb,
            collection,
            visits=visits,
            opdb=opdb,
            mode="realtime",
            pipelineVersions=False,
            log=say,
        )


def printInFull(verdict: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Print the whole judgement, for a log to keep when it could not be stored."""
    with pd.option_context("display.width", 10_000, "display.max_columns", None, "display.max_rows", None):
        say("Not written, twilightVerdict:")
        say(verdict.to_string(index=False))
        say("Not written, twilightMetrics:")
        say(metrics.to_string(index=False))


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
    parser.add_argument(
        "--no-store", action="store_true", help=f"don't store {TWILIGHT_DATASET} in the Butler collection"
    )
    parser.add_argument("--qadb-host", help="load the judged visits into the qadb on this host")
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
    opdb = butler = None
    collection = args.collection
    if args.opdb_host is not None:
        from pfs.utils.database.opdb import OpDB

        opdb = OpDB(host=args.opdb_host, port=args.opdb_port, dbname="opdb", user="public_user")
    if args.no_store and args.qadb_host is not None:
        parser.error(
            "--no-store with --qadb-host would load what the collection already holds, not this judgement"
        )
    if args.sequence_of_visit is not None:
        if args.collection is None or args.opdb_host is None:
            parser.error("--sequence-of-visit needs --collection and --opdb-host")
        from pfs.drp.qa.comparison.queries import readSequenceOfVisit

        sequence = readSequenceOfVisit(opdb, args.sequence_of_visit)
        if not isTwilightSequence(sequence):
            _, why = sequenceReadiness(sequence, {}, {})
            say(f"Visit {args.sequence_of_visit}: {why}")
            return EXIT_NOT_TWILIGHT
        from lsst.daf.butler import Butler

        from pfs.drp.qa.metrics.readers import datasetDetectors, readMetrics

        visits = [int(visit) for visit in sequence["pfs_visit_id"]]
        butler = Butler(args.butler, collections=args.collection, writeable=False)
        reduced = datasetDetectors(butler, "pfsArm", visits)
        stored = datasetDetectors(butler, "twilightFiberFlux", visits)
        ready, why = sequenceReadiness(sequence, reduced, stored)
        say(f"Visit {args.sequence_of_visit}: {why}")
        if not ready:
            return EXIT_PENDING
        fibers = readMetrics(butler, visits=visits, datasetType="twilightFiberFlux", progress=True)
        sequenceId = int(sequence["iic_sequence_id"].iloc[0])
        fibers["sequence"] = sequenceId
        suffix = f"-{sequenceId}"
    elif args.parquet is not None:
        if args.qadb_host is not None:
            parser.error("--qadb-host needs a Butler collection, not a parquet")
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
    if args.sequences is not None:
        known = pd.read_csv(args.sequences).set_index("visit")["sequence"]
        missing = fibers["sequence"] < 0
        fibers.loc[missing, "sequence"] = fibers.loc[missing, "visit"].map(known).fillna(-1).astype(int)
    unknown = fibers["sequence"] < 0
    if unknown.any():
        say(f"Skipping visits without W_SEQID: {sorted(fibers.loc[unknown, 'visit'].unique())}")
        fibers = fibers[~unknown]
    say(f"Judging {fibers['visit'].nunique()} exposures in {fibers['sequence'].nunique()} sequences")

    metrics = sequenceConsistency(fibers, minSnr=args.min_snr)
    verdict = twilightVerdict(metrics)
    images = twilightImageVerdict(metrics)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(args.output / f"twilightMetrics{suffix}.parquet", index=False)
        verdict.to_csv(args.output / f"twilightVerdict{suffix}.csv", index=False)
        say(
            f"Wrote {args.output / f'twilightMetrics{suffix}.parquet'} and"
            f" {args.output / f'twilightVerdict{suffix}.csv'}"
        )

    bad = verdict[verdict["status"].isin(["WARN", "FAIL"])] if not verdict.empty else verdict
    if verdict.empty:
        say("Nothing measured: no sequence has enough exposures in an arm.")
    else:
        counts = verdict.groupby(["sequence", "arm"])["status"].value_counts().unstack(fill_value=0)
        say(counts.to_string())
        if len(bad):
            say(bad[["visit", "arm", "status", "reason"]].to_string(index=False))
    outcome = EXIT_JUDGED_BAD if len(bad) else EXIT_JUDGED

    if butler is None or images.empty or args.no_store:
        return outcome
    judged = sorted(int(visit) for visit in images["visit"].unique())
    try:
        storeAndLoad(args, images, judged, collection, opdb)
    except Exception as error:
        say(f"Storing or loading the judgement failed: {error!r}")
        printInFull(verdict, metrics)
        return EXIT_NOT_WRITTEN
    return outcome


if __name__ == "__main__":
    sys.exit(main())
