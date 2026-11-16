"""Compare a run's QA metrics with reference visits', from qadb: ``bin.src/qaCompareRuns.py``.

Reads the metrics of a visit range and of each reference
(`pfs.drp.qa.qadb.reader.readQadb`) and compares their medians per camera and
population (`pfs.drp.qa.metrics.runComparison.compareRuns`). Nothing is stored:
the comparison is made when asked for, here and by the dashboard, which calls
`compareCollections` too.

The exit status is the worst status (`pfs.drp.qa.metrics.runComparison.exitCode`):
0 ``PASS``, 1 ``WARN``, 2 ``FAIL``, 3 when nothing was judged; `EXIT_ERROR` (4)
when a collection is not in qadb or the database could not be read, `EXIT_USAGE`
(64) for bad arguments.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pfs.drp.qa.metrics.runComparison import COMPARED_METRICS, compareRuns, exitCode
from pfs.drp.qa.qadb.reader import collectionLoaded, readQadb
from pfs.drp.qa.qadb.tables import IMAGE_KEYS
from pfs.drp.qa.qadb.visitSummary import EXIT_ERROR, EXIT_USAGE

__all__ = [
    "Reference",
    "compareCollections",
    "comparisonDocument",
    "comparisonTables",
    "formatComparison",
    "main",
    "metricTables",
    "parseReference",
]


@dataclass(frozen=True)
class Reference:
    """The visits a run is compared with.

    Parameters
    ----------
    name : `str`
        What to call it: ``run30``, ``firstNight``.
    first, last : `int`
        The visit range, inclusive.
    collection : `str` or `None`
        The collection its QA results were loaded from; `None` for the run's own.
    """

    name: str
    first: int
    last: int
    collection: str | None = None


def parseReference(text: str) -> Reference:
    """Parse ``NAME=FIRST-LAST`` or ``NAME=COLLECTION:FIRST-LAST``.

    Raises
    ------
    ValueError
        If ``text`` is neither.
    """
    name, equals, rest = text.partition("=")
    collection, _, span = rest.rpartition(":")
    first, dash, last = span.partition("-")
    if not (name and equals and dash and first.isdigit() and last.isdigit()):
        raise ValueError(f"expected NAME=[COLLECTION:]FIRST-LAST, not {text!r}")
    if int(last) < int(first):
        raise ValueError(f"{text!r}: the range ends before it starts")
    return Reference(name, int(first), int(last), collection or None)


def comparisonTables(tables: Mapping[str, pd.DataFrame]) -> list[pd.DataFrame]:
    """Return the tables `compareRuns` takes from what `pfs.drp.qa.qadb.reader.readQadb` read.

    ``iqQaMetrics``, and ``dmLineResidualMetrics`` with the ``obsType`` of its
    image (from ``dmLineResidualSummary``), which qadb does not repeat per metric.
    """
    residuals = tables["dmLineResidualMetrics"]
    if "obsType" not in residuals:
        obsType = tables["dmLineResidualSummary"].reindex(columns=[*IMAGE_KEYS, "obsType"])
        residuals = residuals.merge(obsType, on=list(IMAGE_KEYS), how="left")
    return [tables["iqQaMetrics"], residuals]


def metricTables(qadb: Any, collection: str, first: int, last: int) -> list[pd.DataFrame]:
    """Return a visit range's metrics tables, read from qadb, as `compareRuns` takes them.

    Raises
    ------
    LookupError
        If qadb holds no processing of ``collection``.
    """
    if not collectionLoaded(qadb, collection):
        raise LookupError(f"qadb holds no processing of collection {collection!r}")
    return comparisonTables(readQadb(qadb, collection, first, last))


def compareCollections(
    qadb: Any,
    collection: str,
    first: int,
    last: int,
    references: Sequence[Reference],
    metrics: Sequence[str] = COMPARED_METRICS,
    knownGood: bool = True,
) -> pd.DataFrame:
    """Compare a visit range's metrics with each reference's, read from qadb.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, read-only.
    collection : `str`
        The collection the QA tasks wrote to, as loaded.
    first, last : `int`
        The visits in question, inclusive.
    references : sequence of `Reference`
        What to compare them with.
    metrics, knownGood
        As `pfs.drp.qa.metrics.runComparison.compareRuns`.

    Returns
    -------
    `pandas.DataFrame`
        `pfs.drp.qa.metrics.runComparison.COMPARISON_COLUMNS`.

    Raises
    ------
    LookupError
        If qadb holds no processing of a collection.
    ValueError
        If two references share a name.
    """
    names = [reference.name for reference in references]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"reference names must differ: {', '.join(repeated)} given more than once")
    run = metricTables(qadb, collection, first, last)
    others = {
        reference.name: metricTables(
            qadb, reference.collection or collection, reference.first, reference.last
        )
        for reference in references
    }
    return compareRuns(run, others, metrics=metrics, knownGood=knownGood)


def _population(row: Any) -> str:
    """Return a row's population as text: the columns that say something."""
    parts = (row.illumination, row.obsType, row.species, row.reference, row.description, row.statusType)
    return " ".join(part for part in parts if part)


def formatComparison(comparison: pd.DataFrame, everything: bool = False) -> str:
    """Return a comparison as text: counts per reference, then a line per ``WARN`` or ``FAIL``.

    With ``everything``, a line per row.
    """
    lines = []
    for name, rows in comparison.groupby("referenceName", sort=False):
        counts = rows["status"].value_counts()
        head = ", ".join(f"{int(counts.get(s, 0))} {s}" for s in ("FAIL", "WARN", "PASS", "UNKNOWN"))
        lines.append(f"against {name}: {len(rows)} compared: {head}")
        for row in rows.itertuples(index=False):
            if not everything and row.status not in ("WARN", "FAIL"):
                continue
            relative = f"{row.relative:+.1%}" if pd.notna(row.relative) else "n/a"
            lines.append(
                f"  {row.status:7s} {row.metric} {row.camera} {_population(row)}: "
                f"{row.runMedian:.4g} (n={row.nRun}) vs {row.referenceMedian:.4g} (n={row.nReference}), "
                f"{relative}"
            )
    if not lines:
        lines.append("nothing to compare")
    return "\n".join(lines)


def comparisonDocument(
    collection: str, visits: tuple[int, int], references: Sequence[Reference], comparison: pd.DataFrame
) -> dict[str, Any]:
    """Return a comparison as a JSON-serializable document, for a nightly job."""
    return {
        "collection": collection,
        "visits": [int(visits[0]), int(visits[1])],
        "references": [
            {
                "name": r.name,
                "collection": r.collection or collection,
                "visits": [r.first, r.last],
            }
            for r in references
        ],
        "exitCode": exitCode(comparison),
        "counts": {status: int(count) for status, count in comparison["status"].value_counts().items()},
        "rows": json.loads(comparison.to_json(orient="records")),
    }


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _reference(text: str) -> Reference:
    try:
        return parseReference(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def main(argv: Sequence[str] | None = None) -> int:
    """Print a visit range's comparison with its references; return the exit status."""
    parser = _Parser(description="Compare a run's QA metrics with reference visits', from qadb.")
    parser.add_argument("-c", "--collection", required=True, help="collection the QA tasks wrote to")
    parser.add_argument(
        "--visit-range", type=int, nargs=2, metavar=("FIRST", "LAST"), required=True, help="the visits"
    )
    parser.add_argument(
        "--reference",
        type=_reference,
        action="append",
        required=True,
        metavar="NAME=[COLLECTION:]FIRST-LAST",
        help="visits to compare with, in the same collection unless one is given; repeat for more",
    )
    parser.add_argument(
        "--all-reference-values",
        action="store_true",
        help="use the references' WARN and FAIL values as well",
    )
    parser.add_argument("--all", action="store_true", help="print every row, not only WARN and FAIL")
    parser.add_argument("--host", default="pfsa-db", help="qadb host (default pfsa-db)")
    parser.add_argument("--port", type=int, default=5436, help="qadb port (default 5436)")
    parser.add_argument("--user", default="pfs", help="qadb user (default pfs)")
    parser.add_argument("--dbname", default="qadb", help="database name (default qadb)")
    parser.add_argument("--json", metavar="FILE", default=None, help="also write the comparison here as JSON")
    args = parser.parse_args(argv)

    first, last = args.visit_range
    if last < first:
        parser.error(f"--visit-range {first} {last}: the range ends before it starts")
    names = [reference.name for reference in args.reference]
    if len(set(names)) < len(names):
        parser.error("--reference: each needs its own name")
    try:
        from pfs.utils.database.qadb import QaDB

        qadb = QaDB(host=args.host, port=args.port, user=args.user, dbname=args.dbname)
        comparison = compareCollections(
            qadb,
            args.collection,
            first,
            last,
            args.reference,
            knownGood=not args.all_reference_values,
        )
        text = formatComparison(comparison, everything=args.all)
        if args.json:
            document = comparisonDocument(args.collection, (first, last), args.reference, comparison)
            Path(args.json).write_text(json.dumps(document, indent=2))
    except Exception as error:
        # Whatever stopped the read or the write, the caller must not take the status for a verdict.
        print(f"qaCompareRuns: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return EXIT_ERROR
    print(text, flush=True)
    return exitCode(comparison)
