#!/usr/bin/env python3
r"""Export per-detector QA tables from a Butler collection to parquet.

For work away from the Butler: analyses, and the test fixtures made from them
(``tests/metrics/data/makeDriftFixture.py``). One registry query per dataset
type, then one ``get`` per detector, with progress. Reads the validation
visits by default, or every visit of a range with ``--visits``.

Example::

    python bin.src/exportMetrics.py -c u/wtg/qa-thresholds/009 \\
        -t dmLineResidualSummary dmLineResidualMetrics -o ~/reports/PIPE2D-1956
    python bin.src/exportMetrics.py -c u/wtg/lineResiduals/run25 \\
        -t dmLineResidualSummary dmLineResidualMetrics --visits 0 999999 -o .

writes ``<datasetType>-<collection>.parquet``, ``/`` in the collection made ``_``.
Requires the LSST stack (``lsst.daf.butler``).
"""

import argparse
from collections.abc import Sequence

from pfs.drp.qa.metrics.readers import exportMetrics
from pfs.drp.qa.metrics.validationVisits import loadValidationVisits


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export per-detector QA tables to parquet.")
    parser.add_argument("-b", "--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument("-c", "--collection", required=True, help="collection to read")
    parser.add_argument("-t", "--dataset-types", nargs="+", required=True, help="per-detector tables")
    parser.add_argument("-o", "--out", default=".", help="directory to write to")
    parser.add_argument(
        "--visits", type=int, nargs=2, metavar=("FIRST", "LAST"), help="a visit range, not the validation set"
    )
    args = parser.parse_args(argv)

    from lsst.daf.butler import Butler

    butler = Butler(args.butler, writeable=False)
    visits = None if args.visits else loadValidationVisits().visits
    exportMetrics(
        butler, args.collection, args.dataset_types, args.out, visits=visits, visitRange=args.visits
    )


if __name__ == "__main__":
    main()
