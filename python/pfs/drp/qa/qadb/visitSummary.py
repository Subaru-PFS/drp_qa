"""Summarize a collection's QA verdicts per visit, from qadb: ``bin.src/qaVisitSummary.py``.

Reads each image's gate verdict (`pfs.drp.qa.qadb.reader.readGate`) and rolls
them up to one row per visit (`pfs.drp.qa.metrics.visitRollup.rollupVisits`).
Given an opdb as well, it says which cameras exposed on a judged visit but have
no verdict, and how many visits of the range have none at all.

The exit status is the worst visit verdict (`pfs.drp.qa.metrics.visitRollup.exitCode`):
0 ``PASS``, 1 ``WARN``, 2 ``FAIL``, 3 when nothing was judged. A question that
could not be answered is none of those: `EXIT_ERROR` (4) for a collection qadb
holds no processing of or a database that could not be read, `EXIT_USAGE` (64)
for bad arguments, so a nightly job never takes a failure to ask for a verdict.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from pfs.drp.qa.comparison.queries import readVisitCameras
from pfs.drp.qa.metrics.visitRollup import exitCode, rollupVisits
from pfs.drp.qa.qadb.reader import FIRST_VISIT, LAST_VISIT, collectionLoaded, readGate

__all__ = ["EXIT_ERROR", "EXIT_USAGE", "formatSummary", "main", "summaryDocument", "visitSummary"]

#: Exit status when the summary could not be made: an unknown collection, or a database error.
EXIT_ERROR = 4

#: Exit status for bad arguments (argparse's own, 2, is ``FAIL`` here).
EXIT_USAGE = 64


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def visitSummary(
    qadb: Any,
    collection: str,
    first: int = FIRST_VISIT,
    last: int = LAST_VISIT,
    opdb: Any = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Return a collection's verdict per visit, and the visits of the range with none.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, read-only.
    collection : `str`
        The collection the QA tasks wrote to, as loaded.
    first, last : `int`
        The visit range, inclusive.
    opdb : `pfs.utils.database.opdb.OpDB`, optional
        The opdb, read-only: which cameras exposed. It is read over the range
        given, an open end taken from the judged visits' span; not at all when
        an end is open and no visit was judged.

    Returns
    -------
    rollup : `pandas.DataFrame`
        `pfs.drp.qa.metrics.visitRollup.ROLLUP_COLUMNS`, one row per judged
        visit; ``nExpected`` and ``missingCameras`` only with ``opdb``.
    unjudged : `list` [`int`]
        Visits of the range with an SpS exposure and no verdict on any camera;
        empty without ``opdb``. Not every exposure is one the QA tasks judge
        (a bias, a dark), so these are listed, not counted against the run.

    Raises
    ------
    LookupError
        If qadb holds no processing of ``collection``.
    """
    if not collectionLoaded(qadb, collection):
        raise LookupError(f"qadb holds no processing of collection {collection!r}")
    gate = readGate(qadb, collection, first, last)
    if opdb is None:
        return rollupVisits(gate), []
    if gate.empty and (first == FIRST_VISIT or last == LAST_VISIT):
        # Nothing judged bounds an open end, and the whole opdb is not read to find out.
        return rollupVisits(gate), []
    low = int(gate["visit"].min()) if first == FIRST_VISIT else int(first)
    high = int(gate["visit"].max()) if last == LAST_VISIT else int(last)
    expected = readVisitCameras(opdb, low, high)
    judged = set(gate["visit"])
    unjudged = sorted(set(expected["visit"]) - judged)
    return rollupVisits(gate, expected[expected["visit"].isin(judged)]), unjudged


def formatSummary(rollup: pd.DataFrame, unjudged: Sequence[int] = ()) -> str:
    """Return a rollup as text: a line of counts, then a line per visit."""
    counts = rollup["qaStatus"].value_counts()
    head = ", ".join(
        f"{int(counts.get(status, 0))} {status}" for status in ("FAIL", "WARN", "PASS", "UNKNOWN")
    )
    lines = [f"{len(rollup)} visits judged: {head}"]
    if unjudged:
        lines.append(f"{len(unjudged)} visits with SpS exposures have no verdict")
    for row in rollup.itertuples(index=False):
        text = f"{row.visit} {row.qaStatus:7s} {row.nImages:2d} images"
        if row.qaStatus in ("WARN", "FAIL"):
            text += f" (F{row.nFail} W{row.nWarn} P{row.nPass} U{row.nUnknown})"
            text += f" {row.decidedByCamera} {row.decidedByTask}: {row.qaReason or row.qaDecidedBy}"
        if row.provisional:
            text += " [provisional]"
        if isinstance(row.missingCameras, str) and row.missingCameras:
            text += f" missing {row.missingCameras}"
        lines.append(text)
    return "\n".join(lines)


def summaryDocument(collection: str, rollup: pd.DataFrame, unjudged: Sequence[int] = ()) -> dict[str, Any]:
    """Return a rollup as a JSON-serializable document, for a nightly job."""
    visits = json.loads(rollup.to_json(orient="records", date_format="iso"))
    counts = rollup["qaStatus"].value_counts()
    return {
        "collection": collection,
        "exitCode": exitCode(rollup),
        "counts": {status: int(count) for status, count in counts.items()},
        "unjudgedVisits": [int(visit) for visit in unjudged],
        "visits": visits,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print a collection's verdict per visit; return the exit status."""
    parser = _Parser(description="Summarize a collection's QA verdicts per visit, from qadb.")
    parser.add_argument("-c", "--collection", required=True, help="collection the QA tasks wrote to")
    parser.add_argument("--visit-range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="a visit range")
    parser.add_argument("--host", default="pfsa-db", help="qadb host (default pfsa-db)")
    parser.add_argument("--port", type=int, default=5436, help="qadb port (default 5436)")
    parser.add_argument("--user", default="pfs", help="qadb user (default pfs)")
    parser.add_argument("--dbname", default="qadb", help="database name (default qadb)")
    parser.add_argument("--opdb-host", default=None, help="check the cameras that exposed in the opdb here")
    parser.add_argument("--opdb-port", type=int, default=5432, help="opdb port (default 5432)")
    parser.add_argument("--opdb-user", default="public_user", help="opdb user; reading suffices")
    parser.add_argument("--opdb-dbname", default="opdb", help="opdb database name (default opdb)")
    parser.add_argument("--json", metavar="FILE", default=None, help="also write the summary here as JSON")
    args = parser.parse_args(argv)

    first, last = args.visit_range or (FIRST_VISIT, LAST_VISIT)
    if last < first:
        parser.error(f"--visit-range {first} {last}: the range ends before it starts")
    try:
        from pfs.utils.database.qadb import QaDB

        qadb = QaDB(host=args.host, port=args.port, user=args.user, dbname=args.dbname)
        opdb = None
        if args.opdb_host:
            from pfs.utils.database.opdb import OpDB

            opdb = OpDB(
                host=args.opdb_host, port=args.opdb_port, dbname=args.opdb_dbname, user=args.opdb_user
            )
        rollup, unjudged = visitSummary(qadb, args.collection, first, last, opdb=opdb)
        text = formatSummary(rollup, unjudged)
        if args.json:
            document = summaryDocument(args.collection, rollup, unjudged)
            Path(args.json).write_text(json.dumps(document, indent=2))
    except Exception as error:
        # Whatever stopped the read or the write, the caller must not take the status for a verdict.
        print(f"qaVisitSummary: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return EXIT_ERROR
    print(text, flush=True)
    return exitCode(rollup)
