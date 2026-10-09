"""Fixtures for the ``pfs.drp.qa.comparison`` tests: pandas, pyyaml, pyarrow, scipy and matplotlib; no stack.

The visits are synthetic. Where a test needs realistic values (focus offsets, camera lists,
command strings) they follow Run28's engineering visits, which are public.
"""

import re
from types import SimpleNamespace

import matplotlib
import pandas as pd
import pyarrow  # noqa: F401  (parquet, for the cli tests)
import pytest
import scipy  # noqa: F401  (pfs.drp.qa.metrics needs it)
import seaborn  # noqa: F401  (pfs.drp.qa.plotting needs it)
import yaml

from pfs.drp.qa.comparison.runs import loadPeriods

matplotlib.use("Agg")

ALL_CAMERAS = "b1,b2,b3,b4,n1,n2,n3,n4,r1,r2,r3,r4"


@pytest.fixture
def periods(tmp_path):
    """Two runs, the second with a pre-run period."""
    path = tmp_path / "runs.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {"name": "run1", "nights": ["2026-01-02", "2026-01-05"]},
                    {
                        "name": "run2",
                        "nights": ["2026-02-10", "2026-02-12"],
                        "preRun": ["2026-02-01", "2026-02-03"],
                    },
                ]
            }
        )
    )
    return loadPeriods(path)


def visit(
    visitId,
    when,
    sequenceType,
    *,
    sequence=None,
    expType=None,
    name="",
    design=None,
    cameras=ALL_CAMERAS,
    exptime=60.0,
    group=None,
):
    """Return one row of a visit listing."""
    defaultExpType = {"scienceArc": "arc", "scienceTrace": "flat", "darks": "dark"}.get(
        sequenceType, "object"
    )
    return {
        "pfs_visit_id": visitId,
        "time_exp_start": pd.Timestamp(when),
        "exptime": exptime,
        "exp_type": expType or defaultExpType,
        "iic_sequence_id": sequence if sequence is not None else (visitId if sequenceType else None),
        "sequence_type": sequenceType,
        "sequence_name": name,
        "group_id": group,
        "group_name": None,
        "cmd_str": "",
        "sequence_comments": None,
        "pfs_design_id": design,
        "cameras": cameras,
    }


def listing(*rows):
    """Return a visit listing from `visit` rows."""
    frame = pd.DataFrame(list(rows))
    for column in ("iic_sequence_id", "group_id", "pfs_design_id"):
        frame[column] = frame[column].astype("Int64")
    return frame


class FakeOpDB:
    """Stand-in for `pfs.utils.database.opdb.OpDB`: answers each query from the table whose pattern matches.

    Parameters
    ----------
    tables : `dict` [`str`, `pandas.DataFrame` or callable]
        Answers (or functions of the bound parameters), by a regular
        expression matched against the SQL.
    """

    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def query_dataframe(self, sql, /, *, params=None, conn=None):
        self.calls.append(SimpleNamespace(sql=sql, params=params))
        for pattern, table in self.tables.items():
            if re.search(pattern, sql, re.DOTALL):
                return table(params) if callable(table) else table.copy()
        raise AssertionError(f"Unexpected query:\n{sql}")


@pytest.fixture(name="visit")
def visitFixture():
    """Return `visit`: test modules can't import this file."""
    return visit


@pytest.fixture(name="listing")
def listingFixture():
    """Return `listing`."""
    return listing


@pytest.fixture(name="FakeOpDB")
def fakeOpDBFixture():
    """Return `FakeOpDB`."""
    return FakeOpDB
