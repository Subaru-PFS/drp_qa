"""The ``pfs.drp.qa.dashboard`` tests need pandas, pyarrow, scipy, pyyaml, sqlalchemy and duckdb, not the stack.

The app itself (Panel, HoloViews) is not tested here. CI does not install duckdb yet, so both
of its jobs skip this directory.
"""

# Listed so the directory is skipped without them.
import duckdb  # noqa: F401
import pandas  # noqa: F401
import pyarrow  # noqa: F401
import scipy  # noqa: F401
import sqlalchemy  # noqa: F401 (pfs.drp.qa.qadb, which qadbSource reads the table specs from)
import yaml  # noqa: F401
