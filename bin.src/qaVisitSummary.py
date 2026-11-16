#!/usr/bin/env python3
r"""Summarize a collection's QA verdicts per visit, from qadb.

One line per visit: the worst of its images' gate verdicts, how many images,
and for a ``WARN`` or ``FAIL`` the camera, task and reason that decided it.
With ``--opdb-host`` it also names the cameras that exposed on a visit but have
no verdict, and counts the visits of the range with none. ``--json`` writes the
same as a document for a nightly job.

Examples::

    python bin.src/qaVisitSummary.py -c qaActor/reductions --visit-range 153000 153534
    python bin.src/qaVisitSummary.py -c qaActor/reductions --visit-range 153000 153534 \\
        --opdb-host pfsa-db --json summary.json

The exit status is the worst visit verdict: 0 ``PASS``, 1 ``WARN``, 2 ``FAIL``,
3 when no visit was judged. A collection qadb holds no processing of, or a
database that can't be read, is an error (4, with a message on stderr), not
missing data; bad arguments exit 64. Stack-free; reads only (credentials in
``~/.pgpass``).
"""

import sys

from pfs.drp.qa.qadb.visitSummary import main

if __name__ == "__main__":
    sys.exit(main())
