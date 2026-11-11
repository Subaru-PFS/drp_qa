#!/usr/bin/env python3
r"""Compare a run's QA metrics with reference visits', from qadb.

Per reference, metric, camera and population (illumination, frame type, lamp):
the median over the visits, the median over the reference's, and their
difference. A metric with a tolerance (``medFwhm``, the line-residual scatter)
gets a status; the others are reported unjudged. A reference's ``WARN`` and
``FAIL`` values are left out of its medians unless ``--all-reference-values``.

Examples::

    python bin.src/qaCompareRuns.py -c qaActor/reductions --visit-range 154000 154074 \\
        --reference firstNight=153534-153600
    python bin.src/qaCompareRuns.py -c qaActor/reductions --visit-range 154000 154074 \\
        --reference firstNight=153534-153600 --reference run30=u/me/run30:148302-148348 \\
        --all --json comparison.json

Only ``WARN`` and ``FAIL`` rows are printed unless ``--all``; the JSON holds
every row. The exit status is the worst status: 0 ``PASS``, 1 ``WARN``, 2
``FAIL``, 3 when nothing was judged; 4 (with a message on stderr) for a
collection qadb holds no processing of or a database that can't be read; 64 for
bad arguments. Stack-free; reads only (credentials in ``~/.pgpass``).
"""

import sys

from pfs.drp.qa.qadb.compareRuns import main

if __name__ == "__main__":
    sys.exit(main())
