"""The ``pfs.drp.qa.qadb`` tests need pandas, pyyaml, pyarrow and SQLAlchemy, but not the stack.

The database is SQLite in memory through SQLAlchemy, behind the same
``commit``/``query_scalar``/``query_dataframe`` surface as
`pfs.utils.database.db.DB`: the ``INSERT ... ON CONFLICT DO UPDATE ...
RETURNING`` statements the writer builds run unchanged on both. ``DDL`` below
mirrors the tables of ``spt_qa_database`` 0.2.0 (migration ``a7c3e5d91b24``);
change both together. Set ``PFS_QADB_TEST_DSN`` (``dbname=... user=... host=...
port=...``) to run the same tests against a scratch PostgreSQL through
`pfs.utils.database.qadb.QaDB` as well.

CI runs this directory in the ``deps`` job; the standard-library job ignores it.
"""

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow  # noqa: F401
import pytest
import sqlalchemy
import yaml  # noqa: F401
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

DATA = Path(__file__).parent / "data"

#: The qadb tables the loader writes, as ``spt_qa_database`` 0.2.0 defines them.
DDL = """
CREATE TABLE pfs_visit (
    pfs_visit_id INTEGER PRIMARY KEY, pfs_visit_description TEXT, pfs_design_id BIGINT, issued_at TIMESTAMP);
CREATE TABLE data_processing_pipe2d (
    processing_id INTEGER PRIMARY KEY AUTOINCREMENT, collection_name TEXT, run_name TEXT,
    pipe2d_version TEXT, drp_qa_version TEXT, datamodel_version TEXT, obs_pfs_version TEXT,
    CONSTRAINT uq_collection_run UNIQUE (collection_name, run_name));
CREATE TABLE image_quality_qa (
    processing_id INTEGER REFERENCES data_processing_pipe2d ON DELETE CASCADE,
    pfs_visit_id INTEGER REFERENCES pfs_visit,
    camera_name VARCHAR(2), spectrograph INTEGER, arm VARCHAR(1), obs_type TEXT, seq_name TEXT,
    illumination TEXT, trace_only BOOLEAN, med_fwhm REAL, med_dx_center REAL, dx_center_rms REAL, pct_flagged REAL,
    pct_low_sn REAL, pct_meas_fail REAL, n_lines INTEGER, lines_per_fiber REAL, n_lit_fibers INTEGER,
    n_fibers INTEGER, n_trace_rows INTEGER, qa_status TEXT, qa_decided_by TEXT, qa_reason TEXT,
    calexp_run TEXT, isr_bad_pixels INTEGER, isr_time REAL, cosmic_ray_count INTEGER,
    cosmic_ray_pixels INTEGER, cosmic_ray_time REAL, reduce_exposure_time REAL, fit_chi2 REAL,
    fit_dof INTEGER, fit_x_rms REAL, fit_y_rms REAL, fit_x_soften REAL, fit_y_soften REAL,
    fit_n_lines INTEGER, fit_total_lines INTEGER, fit_reserved_chi2 REAL, fit_reserved_x_rms REAL,
    fit_reserved_y_rms REAL, fit_reserved_x_soften REAL, fit_reserved_y_soften REAL,
    fit_reserved_n_lines INTEGER,
    PRIMARY KEY (processing_id, pfs_visit_id, camera_name));
CREATE TABLE image_quality_qa_species (
    processing_id INTEGER, pfs_visit_id INTEGER, camera_name VARCHAR(2), description TEXT, metric TEXT,
    value REAL,
    PRIMARY KEY (processing_id, pfs_visit_id, camera_name, description, metric),
    FOREIGN KEY (processing_id, pfs_visit_id, camera_name)
        REFERENCES image_quality_qa (processing_id, pfs_visit_id, camera_name) ON DELETE CASCADE);
CREATE TABLE dm_line_residual_qa (
    processing_id INTEGER REFERENCES data_processing_pipe2d ON DELETE CASCADE,
    pfs_visit_id INTEGER REFERENCES pfs_visit,
    camera_name VARCHAR(2), spectrograph INTEGER, arm VARCHAR(1), camera_set TEXT, obs_type TEXT,
    fitted BOOLEAN, n_species INTEGER, n_species_measured INTEGER, qa_status TEXT, qa_decided_by TEXT,
    qa_reason TEXT, recommended_action TEXT, calib_id TEXT, calib_run TEXT, calib_created TIMESTAMP,
    calib_first_visit INTEGER, calib_last_visit INTEGER,
    PRIMARY KEY (processing_id, pfs_visit_id, camera_name));
CREATE TABLE dm_line_residual_qa_metric (
    processing_id INTEGER, pfs_visit_id INTEGER, camera_name VARCHAR(2), reference TEXT, description TEXT,
    status_type TEXT, metric TEXT, value REAL, status TEXT,
    PRIMARY KEY (processing_id, pfs_visit_id, camera_name, reference, description, status_type, metric),
    FOREIGN KEY (processing_id, pfs_visit_id, camera_name)
        REFERENCES dm_line_residual_qa (processing_id, pfs_visit_id, camera_name) ON DELETE CASCADE);
CREATE TABLE qa_thresholds (
    thresholds_id INTEGER PRIMARY KEY AUTOINCREMENT, sha256 VARCHAR(64) NOT NULL UNIQUE, name TEXT,
    derived_on TIMESTAMP, collection TEXT, reference_runs TEXT, drp_qa_version TEXT, datamodel_version TEXT,
    obs_pfs_version TEXT, drp_stella_version TEXT, n_entries INTEGER, registered_at TIMESTAMP);
CREATE TABLE qa_threshold_entry (
    thresholds_id INTEGER REFERENCES qa_thresholds ON DELETE CASCADE, metric TEXT, population TEXT, arm VARCHAR(1),
    spectrograph INTEGER, obs_type TEXT, description TEXT, camera_set TEXT, reference TEXT,
    status_type TEXT, illumination TEXT, higher_is_worse BOOLEAN, absolute BOOLEAN, method TEXT, warn DOUBLE PRECISION,
    fail DOUBLE PRECISION,
    n_good INTEGER, n_good_visits INTEGER, visit_range TEXT, reliable BOOLEAN, fail_bounded BOOLEAN,
    degenerate BOOLEAN, provenance TEXT,
    PRIMARY KEY (thresholds_id, metric, population));
CREATE TABLE qa_gate (
    processing_id INTEGER REFERENCES data_processing_pipe2d ON DELETE CASCADE,
    pfs_visit_id INTEGER REFERENCES pfs_visit,
    camera_name VARCHAR(2), qa_status TEXT, decided_by_task TEXT, qa_decided_by TEXT, qa_reason TEXT,
    provisional BOOLEAN, mode TEXT, iq_thresholds_id INTEGER REFERENCES qa_thresholds,
    dm_thresholds_id INTEGER REFERENCES qa_thresholds, decided_at TIMESTAMP,
    PRIMARY KEY (processing_id, pfs_visit_id, camera_name));
"""

#: The tables `DDL` makes, in dependency order.
TABLES = re.findall(r"CREATE TABLE (\w+)", DDL)


class SqliteQaDB:
    """An in-memory SQLite database behind `pfs.utils.database.db.DB`'s query surface.

    Foreign keys are enforced, so a long table's rows need their image's row.
    """

    def __init__(self):
        self.engine = sqlalchemy.create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        with self.engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = ON")
            for statement in DDL.split(";"):
                if statement.strip():
                    conn.exec_driver_sql(statement)
        self.statements: list[str] = []

    def _bind(self, sql, params):
        self.statements.append(sql)
        statement = text(sql)
        if isinstance(params, dict):
            lists = [k for k, v in params.items() if isinstance(v, list | tuple)]
            if lists:
                # `= ANY(:x)` is PostgreSQL; SQLite takes `IN`.
                for key in lists:
                    statement = text(sql.replace(f"= ANY(:{key})", f"IN :{key}"))
                statement = statement.bindparams(*(sqlalchemy.bindparam(k, expanding=True) for k in lists))
        return statement

    @contextmanager
    def connection(self):
        """Yield one connection for several statements, committed together, as `DB.connection` does."""
        with self.engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = ON")
            yield conn

    def commit(self, query, params=None):
        with self.connection() as conn:
            conn.execute(self._bind(query, params), params)

    def query_scalar(self, sql, /, *, params=None, conn=None):
        with self.engine.begin() as c:
            row = c.execute(self._bind(sql, params), params).fetchone()
        return None if row is None else row[0]

    def query_scalars(self, sql, /, *, params=None, conn=None):
        with self.engine.begin() as c:
            return [row[0] for row in c.execute(self._bind(sql, params), params).fetchall()]

    def query_dataframe(self, sql, /, *, params=None, conn=None):
        with self.engine.begin() as c:
            return pd.read_sql(self._bind(sql, params), con=c, params=params)

    def count(self, table: str, where: str = "") -> int:
        return int(self.query_scalar(f"SELECT count(*) FROM {table}" + (f" WHERE {where}" if where else "")))


#: What the tests may create in a database that holds real rows too (a copy of qadb): processings of test
#: collections, and visits with small ids. The tests scope their counts with these, and the PostgreSQL
#: fixture deletes only these from the shared tables.
TEST_PROCESSINGS = "collection_name LIKE 'u/test/%'"
TEST_VISITS = "pfs_visit_id < 1000"

#: The new QA tables, which the PostgreSQL fixture empties (the dump has none of their rows).
QA_TABLES = [table for table in TABLES if table not in ("pfs_visit", "data_processing_pipe2d")]


def _postgres():
    """Return a `QaDB` on the scratch PostgreSQL ``PFS_QADB_TEST_DSN`` names, its test rows removed; or None."""
    dsn = os.environ.get("PFS_QADB_TEST_DSN")
    if not dsn:
        return None
    from pfs.utils.database.qadb import QaDB

    class TestQaDB(QaDB):
        def count(self, table, where=""):
            return int(
                self.query_scalar(f"SELECT count(*) FROM {table}" + (f" WHERE {where}" if where else ""))
            )

    db = TestQaDB(dsn)
    for table in reversed(QA_TABLES):
        db.commit(f"DELETE FROM {table}")
    db.commit(f"DELETE FROM data_processing_pipe2d WHERE {TEST_PROCESSINGS}")
    db.commit(f"DELETE FROM pfs_visit WHERE {TEST_VISITS}")
    return db


@pytest.fixture(params=["sqlite", "postgres"])
def qadb(request):
    """Return a fresh database: SQLite always, PostgreSQL when ``PFS_QADB_TEST_DSN`` is set."""
    if request.param == "sqlite":
        return SqliteQaDB()
    db = _postgres()
    if db is None:
        pytest.skip("PFS_QADB_TEST_DSN not set")
    return db


@dataclass(frozen=True)
class StubRef:
    datasetType: str
    dataId: dict = field(hash=False)
    run: str = "u/test/qa/20261026T000000Z"

    def __hash__(self):
        return hash((self.datasetType, tuple(sorted(self.dataId.items())), self.run))


class StubButler:
    """Answer the loader's Butler calls from tables keyed by dataset type and image.

    Parameters
    ----------
    tables : `dict` [`str`, `pandas.DataFrame`]
        Per-detector tables by dataset type, split per image on ``visit``,
        ``arm`` and ``spectrograph``.
    runs : `dict` [`tuple`, `str`], optional
        The RUN of an image's datasets, by ``(visit, arm, spectrograph)``;
        default one RUN.
    datasets : `dict` [`str`, object], optional
        What ``get(name, collections=...)`` returns for ``packages`` and the
        tasks' ``*_config``.
    """

    def __init__(self, tables, runs=None, datasets=None):
        self.tables = tables
        self.runs = runs or {}
        self.datasets = datasets or {}
        self.queries = []

    def query_datasets(self, datasetType, collections=None, where="", bind=None, **kwargs):
        self.queries.append((datasetType, collections))
        if datasetType not in self.tables:
            return []
        if "visits" in bind:
            wanted = set(bind["visits"]).__contains__
        else:
            wanted = range(bind["firstVisit"], bind["lastVisit"] + 1).__contains__
        table = self.tables[datasetType]
        refs = []
        for (v, a, s), _ in table.groupby(["visit", "arm", "spectrograph"], sort=True):
            if wanted(v):
                dataId = {
                    "instrument": bind["instrumentName"],
                    "visit": int(v),
                    "arm": a,
                    "spectrograph": int(s),
                }
                refs.append(StubRef(datasetType, dataId, self.runs.get((int(v), a, int(s)), StubRef.run)))
        return refs

    def get(self, ref, collections=None):
        if isinstance(ref, str):
            # A dataset keyed by (name, collection) answers for that collection alone; by name, for every one.
            for collection in collections or ():
                if (ref, collection) in self.datasets:
                    return self.datasets[(ref, collection)]
            if ref in self.datasets:
                return self.datasets[ref]
            raise LookupError(f"{ref} not in {collections}")
        table = self.tables[ref.datasetType]
        image = (table["visit"] == ref.dataId["visit"]) & (table["arm"] == ref.dataId["arm"])
        image &= table["spectrograph"] == ref.dataId["spectrograph"]
        return table[image].reset_index(drop=True)


@pytest.fixture
def fixtures():
    """Return the four-image fixtures by dataset type."""
    return {
        name: pd.read_parquet(DATA / f"{name}.parquet")
        for name in ("iqQaMetrics", "dmLineResidualSummary", "dmLineResidualMetrics")
    }


@pytest.fixture(name="StubButler")
def stubButlerFixture():
    return StubButler


@pytest.fixture(name="SqliteQaDB")
def sqliteFixture():
    return SqliteQaDB


@pytest.fixture
def configs():
    """Return the tasks' configs as the Butler stores them: the packaged thresholds files they name."""
    return {
        "imageQualityQa_config": SimpleNamespace(thresholdsFile="iqQaThresholds-run25.yaml"),
        "dmLineResiduals_config": SimpleNamespace(thresholdsFile="dmLineResidualThresholds-run25.yaml"),
    }


@pytest.fixture
def thresholdsFiles(configs):
    """Return the packaged thresholds file of each task, by task."""
    from pfs.drp.qa.metrics.gate import thresholdsPath

    return {
        task.removesuffix("_config"): thresholdsPath(config.thresholdsFile)
        for task, config in configs.items()
    }


@pytest.fixture
def ddlColumns():
    """Return the columns of each table in `DDL`, in order, by table."""
    columns = {}
    for name, body in re.findall(r"CREATE TABLE (\w+) \((.*?)\);", DDL, re.DOTALL):
        # Drop every parenthesised list (key and reference column lists, type lengths), innermost first.
        while re.search(r"\([^()]*\)", body):
            body = re.sub(r"\([^()]*\)", "", body)
        names = []
        for part in body.split(","):
            words = part.split()
            if words and words[0].upper() not in ("PRIMARY", "FOREIGN", "CONSTRAINT"):
                names.append(words[0])
        columns[name] = names
    return columns


@pytest.fixture
def scope():
    """Return the WHERE clauses that pick the tests' own rows of the shared tables."""
    return {"data_processing_pipe2d": TEST_PROCESSINGS, "pfs_visit": TEST_VISITS}
