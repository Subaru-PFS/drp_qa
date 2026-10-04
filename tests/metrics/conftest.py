"""The ``pfs.drp.qa.metrics`` tests need pandas but not the LSST stack.

CI runs them in the ``deps`` job; the standard-library job ignores this
directory.
"""

import pandas  # noqa: F401 (listed so the directory is skipped without it)
