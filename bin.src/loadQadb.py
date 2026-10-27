#!/usr/bin/env python3
r"""Load a collection's QA tables, thresholds and gate verdicts from the Butler into qadb.

The tables ``imageQualityQa`` and ``dmLineResiduals`` wrote for the visits
(``iqQaMetrics``, ``iqQaSpeciesMetrics``, ``dmLineResidualSummary``,
``dmLineResidualMetrics``) go to ``image_quality_qa``, ``image_quality_qa_species``,
``dm_line_residual_qa`` and ``dm_line_residual_qa_metric``, keyed by the
collection and Butler RUN (``data_processing_pipe2d``), the visit and the camera;
the thresholds files their configs name to ``qa_thresholds``; and each image's
worst verdict to ``qa_gate``. Loading is idempotent: run it again after a
re-judgement, or after a load that stopped.

Examples::

    # What would be written, as parquet, touching no database
    python bin.src/loadQadb.py -c u/wtg/comparison/run30/w.2026.35-153-g92d72a8 \\
        --visit-range 148302 148348 --dry-run ~/reports/PIPE2D-1385/dry

    # Load it, with the opdb's visit descriptions
    python bin.src/loadQadb.py -c qaActor/reductions --visits 150001 150002 --opdb-host pfsa-db

Requires the LSST stack (``lsst.daf.butler``) and a qadb user that may write
(credentials in ``~/.pgpass``). Prints progress as it reads and writes.
"""

import argparse
from collections.abc import Sequence

from pfs.drp.qa.qadb.loader import MODES, collectionExists, loadCollection


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load QA tables from a Butler collection into qadb.")
    parser.add_argument("-b", "--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument("-c", "--collection", required=True, help="collection the QA tasks wrote to")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--visits", type=int, nargs="+", help="the visits to load")
    which.add_argument("--visit-range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="a visit range")
    parser.add_argument("--host", default="pfsa-db", help="qadb host (default pfsa-db)")
    parser.add_argument("--port", type=int, default=5436, help="qadb port (default 5436)")
    parser.add_argument("--user", default="pfs", help="qadb user; must be allowed to write (default pfs)")
    parser.add_argument("--dbname", default="qadb", help="database name (default qadb)")
    parser.add_argument(
        "--opdb-host",
        default=None,
        help="copy visit descriptions from the opdb on this host (read-only user)",
    )
    parser.add_argument("--mode", choices=MODES, default="backfill", help="recorded in qa_gate.mode")
    parser.add_argument(
        "--no-pipeline-versions",
        action="store_true",
        help="don't read the reducing pipeline's versions from the pfsArm headers (one read per image)",
    )
    parser.add_argument("--dry-run", metavar="DIR", default=None, help="write parquet files here instead")
    args = parser.parse_args(argv)

    from lsst.daf.butler import Butler

    butler = Butler(args.butler, writeable=False)
    if not collectionExists(butler, args.collection):
        raise SystemExit(f"No collection {args.collection!r} in {args.butler}")
    qadb = opdb = None
    if args.dry_run is None:
        from pfs.utils.database.qadb import QaDB

        qadb = QaDB(host=args.host, port=args.port, user=args.user, dbname=args.dbname)
    if args.opdb_host and args.dry_run is None:
        from pfs.utils.database.opdb import OpDB

        opdb = OpDB(host=args.opdb_host, user="public_user")

    counts = loadCollection(
        butler,
        qadb,
        args.collection,
        visits=args.visits,
        visitRange=tuple(args.visit_range) if args.visit_range else None,
        opdb=opdb,
        mode=args.mode,
        pipelineVersions=not args.no_pipeline_versions,
        dryRun=args.dry_run,
    )
    for table, count in counts.items():
        print(f"{table}: {count}", flush=True)


if __name__ == "__main__":
    main()
