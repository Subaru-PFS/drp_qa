#!/usr/bin/env python3
r"""Report the saturation of a visit's detector images: a figure and a Markdown file each.

Example::

    python bin.src/saturationReport.py -c u/wtg/qa,drpActor/reductions --visit 153432 -o ~/reports/saturation
    python bin.src/saturationReport.py -c drpActor/reductions --visit 153432 --cameras r1 r2 -o .

Without ``--cameras``, the detectors whose ``iqQaMetrics`` in the collections report saturation.
Requires the LSST stack. See `pfs.drp.qa.saturationReport`.
"""

import sys

from pfs.drp.qa.saturationReport import main

if __name__ == "__main__":
    sys.exit(main())
