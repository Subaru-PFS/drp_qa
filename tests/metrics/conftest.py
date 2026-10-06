"""The ``pfs.drp.qa.metrics`` tests need pandas, scipy and pyyaml but not the LSST stack.

CI runs them in the ``deps`` job; the standard-library job ignores this
directory.
"""

# Listed so the directory is skipped without them.
import pandas  # noqa: F401
import scipy  # noqa: F401
import yaml  # noqa: F401
