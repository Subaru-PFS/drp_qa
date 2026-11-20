"""The ``drpQa.py`` subcommands that work on qadb: ``load``, ``values`` and ``judge``.

``load`` reads a Butler, so it needs the LSST stack; it is imported only when
``load`` runs. ``values`` and ``judge`` read and write qadb alone.
"""

import argparse
import getpass
from collections.abc import Sequence

__all__ = ["EXIT_REFUSED", "judgeMain", "loadMain", "valuesMain"]

#: ``values``: exit status when a candidate was refused, so the values in force stayed for it.
EXIT_REFUSED = 1


def say(line: str) -> None:
    """Print a line, flushed: the caller may be streaming the output into a log."""
    print(line, flush=True)


def _qadbOptions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="pfsa-db", help="qadb host (default pfsa-db)")
    parser.add_argument("--port", type=int, default=5436, help="qadb port (default 5436)")
    parser.add_argument("--user", default="pfs", help="qadb user; must be allowed to write (default pfs)")
    parser.add_argument("--dbname", default="qadb", help="database name (default qadb)")


def _qadb(args: argparse.Namespace):
    from pfs.utils.database.qadb import QaDB

    return QaDB(host=args.host, port=args.port, user=args.user, dbname=args.dbname)


def _visitRange(text: str) -> tuple[int, int]:
    """Return ``"133025-133041"`` (or one visit) as first and last visit."""
    first, _, last = text.partition("-")
    try:
        visits = int(first), int(last or first)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a visit or FIRST-LAST") from None
    if visits[0] > visits[1]:
        raise argparse.ArgumentTypeError(f"{text!r}: the first visit is after the last")
    return visits


def loadMain(argv: Sequence[str] | None = None) -> int:
    """Copy a collection's QA numbers from the Butler to qadb, and judge them.

    The tables ``imageQualityQa`` and ``dmLineResiduals`` wrote for the visits go
    to ``image_quality_qa``, ``image_quality_qa_species``, ``dm_line_residual_qa``
    and ``dm_line_residual_qa_metric``, keyed by the collection and Butler RUN
    (``data_processing_pipe2d``), the visit and the camera; a twilight sequence
    that is complete is measured into ``twilight_qa``; and each image's verdict
    goes to ``qa_gate``. Loading is idempotent: run it again after a load that
    stopped, or when more of a sequence has been reduced.
    """
    from pfs.drp.qa.qadb.loader import MODES

    parser = argparse.ArgumentParser(
        description="Load QA numbers from a Butler collection into qadb and judge them."
    )
    parser.add_argument("-b", "--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument("-c", "--collection", required=True, help="collection the QA tasks wrote to")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--visits", type=int, nargs="+", help="the visits to load")
    which.add_argument("--visit-range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="a visit range")
    _qadbOptions(parser)
    parser.add_argument(
        "--opdb-host",
        default=None,
        help="the opdb's host: visit descriptions are copied from it, and a twilight visit's sequence is "
        "found in it; without it no twilight is judged",
    )
    parser.add_argument("--opdb-port", type=int, default=5432, help="opdb port (default 5432)")
    parser.add_argument(
        "--opdb-user", default="public_user", help="opdb user; reading suffices (default public_user)"
    )
    parser.add_argument("--opdb-dbname", default="opdb", help="opdb database name (default opdb)")
    parser.add_argument("--mode", choices=MODES, default="backfill", help="recorded in qa_gate.mode")
    parser.add_argument(
        "--no-pipeline-versions",
        action="store_true",
        help="don't read the reducing pipeline's versions from the pfsArm headers (one read per image)",
    )
    parser.add_argument("--dry-run", metavar="DIR", default=None, help="write parquet files here instead")
    args = parser.parse_args(argv)

    from lsst.daf.butler import Butler

    from pfs.drp.qa.qadb.loader import collectionExists, loadCollection

    butler = Butler(args.butler, writeable=False)
    if not collectionExists(butler, args.collection):
        raise SystemExit(f"No collection {args.collection!r} in {args.butler}")
    qadb = opdb = None
    if args.dry_run is None:
        qadb = _qadb(args)
    if args.opdb_host:
        from pfs.utils.database.opdb import OpDB

        opdb = OpDB(host=args.opdb_host, port=args.opdb_port, dbname=args.opdb_dbname, user=args.opdb_user)
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
        log=say,
    )
    for table, count in counts.items():
        say(f"{table}: {count}")
    return 0


def valuesMain(argv: Sequence[str] | None = None) -> int:
    """Make a run's normal values from its calibration frames, compare them, and store those accepted.

    Without ``--accept`` it only prints each candidate beside the value in force,
    with its shift. With ``--accept NAME`` the values are stored, valid from the
    visit after the calibration frames, and every frame from there on is judged
    again. A value shifted beyond its check's FAIL is refused, and the value in
    force stays, unless ``--reason`` says why it is accepted.
    """
    parser = argparse.ArgumentParser(
        description="Make, compare and accept a collection's normal values, from qadb."
    )
    parser.add_argument(
        "-c", "--collection", required=True, help="collection the QA tasks wrote to, as loaded"
    )
    parser.add_argument(
        "--visits",
        type=_visitRange,
        nargs="+",
        required=True,
        metavar="FIRST-LAST",
        help="the calibration frames: the visits the run's calibrations are built from",
    )
    parser.add_argument(
        "--valid-from", type=int, help="first visit the values judge (default: after --visits)"
    )
    parser.add_argument(
        "--arms", help="only these arms, e.g. brn or m, for one camera group's calibration set"
    )
    parser.add_argument(
        "--accept",
        nargs="?",
        const=getpass.getuser(),
        metavar="NAME",
        help="store the values, accepted by NAME (default: the user running this)",
    )
    parser.add_argument("--reason", default="", help="why values shifted beyond FAIL are accepted")
    parser.add_argument("--all", action="store_true", help="list every value, not only those that moved")
    _qadbOptions(parser)
    args = parser.parse_args(argv)

    from pfs.drp.qa.metrics.normalValues import formatComparison
    from pfs.drp.qa.qadb.judge import makeValues

    compared, stored, refused = makeValues(
        _qadb(args),
        args.collection,
        args.visits,
        validFrom=args.valid_from,
        arms=list(args.arms) if args.arms else None,
        store=args.accept is not None,
        acceptedBy=args.accept or "",
        reason=args.reason,
        log=say,
    )
    say(formatComparison(compared, onlyShifted=not args.all))
    failed = refused[refused["status"] == "FAIL"]
    tooFew = refused[refused["status"] == "TOO_FEW"]
    if len(tooFew):
        say(
            f"{len(tooFew)} values rest on too few frames and are left out; the values in force stay for them"
        )
    if len(failed):
        say(f"{len(failed)} values are shifted beyond FAIL and refused; give --reason to accept them")
    if args.accept is None:
        say(f"nothing stored: {len(stored)} values would be, with --accept")
    else:
        say(f"{len(stored)} values stored, accepted by {args.accept}")
    return EXIT_REFUSED if len(failed) else 0


def judgeMain(argv: Sequence[str] | None = None) -> int:
    """Judge again the numbers qadb holds for a visit range, with the limits and normal values as they are now."""
    parser = argparse.ArgumentParser(description="Judge stored QA numbers again, from qadb.")
    parser.add_argument(
        "-c", "--collection", required=True, help="collection the QA tasks wrote to, as loaded"
    )
    parser.add_argument(
        "--visits", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"), help="visit range"
    )
    parser.add_argument("--dry-run", action="store_true", help="judge and count, and write nothing")
    _qadbOptions(parser)
    args = parser.parse_args(argv)

    from pfs.drp.qa.qadb.judge import judgeStored

    judgeStored(_qadb(args), args.collection, *args.visits, dryRun=args.dry_run, log=say)
    return 0
