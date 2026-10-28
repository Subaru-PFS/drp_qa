"""The dashboard app's tests need Panel and HoloViews, as well as `tests/dashboard`'s packages.

CI installs neither, so both of its jobs skip this directory.
"""

# Listed so the directory is skipped without them.
import duckdb  # noqa: F401
import holoviews  # noqa: F401
import pandas  # noqa: F401
import panel  # noqa: F401
import pyarrow  # noqa: F401
import scipy  # noqa: F401
import sqlalchemy  # noqa: F401
import yaml  # noqa: F401
