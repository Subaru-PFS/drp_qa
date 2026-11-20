"""The ``pfs.drp.qa.qadb`` tests need pandas, scipy, pyarrow and SQLAlchemy, but not the stack.

The database is SQLite in memory through SQLAlchemy, behind the same
``commit``/``query_scalar``/``query_dataframe`` surface as
`pfs.utils.database.db.DB`: the ``INSERT ... ON CONFLICT DO UPDATE ...
RETURNING`` statements the writer builds run unchanged on both. ``DDL`` below
is the schema the code here writes to: numbers tables without a verdict, one
``qa_gate`` row per image, and ``qa_normal_value``. It is the proposal
``spt_qa_database``'s migration follows; change both together. Set
``PFS_QADB_TEST_DSN`` (``dbname=... user=... host=... port=...``) to run the
same tests against a scratch PostgreSQL through
`pfs.utils.database.qadb.QaDB` as well.

CI runs this directory in the ``deps`` job; the standard-library job ignores it.
"""

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import pytest
import scipy  # noqa: F401
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

DATA = Path(__file__).parent / "data"

#: The qadb tables the loader and the judge write: the schema the gate's redesign proposes for ``spt_qa_database``.
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
    n_fibers INTEGER, n_trace_rows INTEGER, n_saturated_lines INTEGER, pct_saturated_pixels REAL,
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
    fitted BOOLEAN, n_species INTEGER, n_species_measured INTEGER, calib_id TEXT, calib_run TEXT,
    calib_created TIMESTAMP, calib_first_visit INTEGER, calib_last_visit INTEGER,
    PRIMARY KEY (processing_id, pfs_visit_id, camera_name));
CREATE TABLE dm_line_residual_qa_metric (
    processing_id INTEGER, pfs_visit_id INTEGER, camera_name VARCHAR(2), reference TEXT, description TEXT,
    status_type TEXT, metric TEXT, value REAL,
    PRIMARY KEY (processing_id, pfs_visit_id, camera_name, reference, description, status_type, metric),
    FOREIGN KEY (processing_id, pfs_visit_id, camera_name)
        REFERENCES dm_line_residual_qa (processing_id, pfs_visit_id, camera_name) ON DELETE CASCADE);
CREATE TABLE qa_gate (
    processing_id INTEGER REFERENCES data_processing_pipe2d ON DELETE CASCADE,
    pfs_visit_id INTEGER REFERENCES pfs_visit,
    camera_name VARCHAR(2), frame_type TEXT, qa_status TEXT, qa_decided_by TEXT, qa_reason TEXT,
    mode TEXT, decided_at TIMESTAMP,
    PRIMARY KEY (processing_id, pfs_visit_id, camera_name));
CREATE TABLE twilight_qa (
    processing_id INTEGER REFERENCES data_processing_pipe2d ON DELETE CASCADE,
    pfs_visit_id INTEGER REFERENCES pfs_visit,
    camera_name VARCHAR(2), spectrograph INTEGER, arm VARCHAR(1), iic_sequence_id INTEGER, seq_name TEXT,
    n_exposures INTEGER, snr REAL, brightness REAL, too_faint BOOLEAN, n_reference INTEGER, noise REAL,
    sat_fraction REAL, n_measured INTEGER, twilight_level_offset REAL, twilight_resid_rms REAL,
    twilight_patch REAL, twilight_arm_mean REAL, twilight_arm_half_diff REAL,
    PRIMARY KEY (processing_id, pfs_visit_id, camera_name));
CREATE TABLE qa_normal_value (
    normal_value_id INTEGER PRIMARY KEY AUTOINCREMENT, collection_name TEXT NOT NULL, metric TEXT NOT NULL,
    camera TEXT NOT NULL, frame_type TEXT NOT NULL DEFAULT '', selector TEXT NOT NULL DEFAULT '',
    valid_from INTEGER NOT NULL, value DOUBLE PRECISION NOT NULL, n_frames INTEGER NOT NULL,
    source_visits TEXT, accepted_by TEXT, accepted_at TIMESTAMP, reason TEXT,
    CONSTRAINT uq_normal_value UNIQUE (collection_name, metric, camera, frame_type, selector, valid_from));
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

#: The QA tables, which the PostgreSQL fixture empties (the dump has none of their rows): all but the normal
#: values, which name their collection and lose the test collections' alone.
QA_TABLES = [
    table for table in TABLES if table not in ("pfs_visit", "data_processing_pipe2d", "qa_normal_value")
]


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
    db.commit(f"DELETE FROM qa_normal_value WHERE {TEST_PROCESSINGS}")
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
        The RUN of an image's datasets, by ``(visit, arm, spectrograph)``, or of
        one dataset type's by ``(datasetType, visit, arm, spectrograph)``;
        default one RUN.
    datasets : `dict` [`str`, object], optional
        What ``get(name, collections=...)`` returns for ``packages``.
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
                image = (int(v), a, int(s))
                run = self.runs.get((datasetType, *image), self.runs.get(image, StubRef.run))
                refs.append(StubRef(datasetType, dataId, run))
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


def quartzFrames(
    visits,
    cameras=("b1", "r1"),
    width=3.0,
    scatter=0.04,
    offset=0.0,
    linesPerFiber=4000.0,
    cameraSet="brn",
    obsType="trace",
):
    """Return the Butler tables of synthetic quartz frames: one image per visit and camera.

    Each image has what the gate reads on a quartz: its trace width and rows per
    lit fiber (``iqQaMetrics``), and the trace centroids' offset from the
    calibration in x and scatter about the visit's detectorMap
    (``dmLineResidualMetrics``). With ``obsType="science"`` the same numbers
    are a science frame's, of which the gate reads the offset alone.

    Parameters
    ----------
    visits : iterable of `int`
        The visits.
    cameras : iterable of `str`
        The cameras (``b1``) of each visit.
    width, scatter, offset, linesPerFiber : `float` or `dict`
        The value on every image; or a `dict` by ``(visit, camera)``, visit or
        camera, in that order, with the value for the others under `None`.
    cameraSet : `str`
        The arms the visits were read with.
    obsType : `str`
        The frames' observation type, as the tasks record it: ``trace`` for a quartz.

    Returns
    -------
    `dict` [`str`, `pandas.DataFrame`]
        ``iqQaMetrics``, ``dmLineResidualSummary`` and ``dmLineResidualMetrics``.
    """

    def pick(value, visit, camera):
        if not isinstance(value, dict):
            return float(value)
        for key in ((visit, camera), visit, camera, None):
            if key in value:
                return float(value[key])
        raise KeyError(f"no value for {visit} {camera}, and none under None")

    iq, summary, metrics = [], [], []
    for visit in visits:
        for camera in cameras:
            image = {"visit": int(visit), "arm": camera[0], "spectrograph": int(camera[1:])}
            iq.append(
                image
                | {
                    "obsType": obsType,
                    "seqName": "Trace" if obsType == "trace" else "",
                    "illumination": "full",
                    "traceOnly": False,
                    "medFwhm": pick(width, visit, camera),
                    "medDxCenter": 0.0,
                    "nLines": 2_400_000,
                    "linesPerFiber": pick(linesPerFiber, visit, camera),
                    "nLitFibers": 600,
                    "nFibers": 600,
                }
            )
            lines = image | {"cameraSet": cameraSet, "obsType": obsType}
            summary.append(lines | {"fitted": True, "nSpecies": 1, "nSpeciesMeasured": 1})
            for reference, statusType, metric, value in (
                ("calib", "BOTH", "medianDeltaX", pick(offset, visit, camera)),
                ("visit", "RESERVED", "robustRmsDeltaX", pick(scatter, visit, camera)),
            ):
                metrics.append(
                    lines
                    | {
                        "reference": reference,
                        "description": "Trace",
                        "statusType": statusType,
                        "metric": metric,
                        "value": value,
                    }
                )
    return {
        "iqQaMetrics": pd.DataFrame(iq),
        "dmLineResidualSummary": pd.DataFrame(summary),
        "dmLineResidualMetrics": pd.DataFrame(metrics),
    }


@pytest.fixture(name="quartzFrames")
def quartzFramesFixture():
    return quartzFrames


def storeValues(qadb, collection, values, validFrom, count=5, when=datetime(2026, 10, 1)):
    """Write normal values straight to ``qa_normal_value``, as accepted ones are stored.

    Parameters
    ----------
    qadb
        The database.
    collection : `str`
        The collection they judge.
    values : `dict` [`tuple`, `float`]
        The value by ``(metric, camera, frameType, selector)``.
    validFrom : `int`
        The first visit they judge.
    count : `int`
        The frames each rests on.
    when : `datetime.datetime`
        When they were accepted.
    """
    from pfs.drp.qa.metrics.normalValues import NORMAL_COLUMNS
    from pfs.drp.qa.qadb.tables import NORMAL_VALUE_KEYS, normalValueRows
    from pfs.drp.qa.qadb.writer import upsert

    rows = [
        {
            "metric": metric,
            "camera": camera,
            "frameType": frameType,
            "selector": selector,
            "value": value,
            "count": count,
            "collection": collection,
            "validFrom": validFrom,
            "sourceVisits": "",
            "acceptedBy": "tester",
            "acceptedAt": when,
            "reason": "",
        }
        for (metric, camera, frameType, selector), value in values.items()
    ]
    frame = pd.DataFrame(rows, columns=list(NORMAL_COLUMNS))
    upsert(qadb, "qa_normal_value", normalValueRows(frame), list(NORMAL_VALUE_KEYS))


@pytest.fixture(name="storeValues")
def storeValuesFixture():
    return storeValues


#: The synthetic twilight sequence: its IIC sequence, visits, arms and spectrographs.
TWILIGHT_SEQUENCE = 54787
TWILIGHT_VISITS = (501, 502, 503, 504)
TWILIGHT_ARMS = ("b", "r", "n")
TWILIGHT_SPECTROGRAPHS = (1, 2, 3, 4)

#: The fault in it: in the third exposure the r2 detector reads 10 % low.
TWILIGHT_STEP = (503, "r", 2, 0.90)


def twilightFibers(visits=TWILIGHT_VISITS, fibersPerDetector=40, seed=3) -> pd.DataFrame:
    """Return ``twilightFiberFlux`` rows of a twilight sequence, as ``twilightFluxQa`` stores them.

    Each fiber has a fixed throughput per arm, the sky a smooth gradient that
    changes through the sequence, and each fiber's median the photon noise of
    its S/N. `TWILIGHT_STEP` dims one detector in one exposure. The header's
    sequence id is -1, as on a frame whose header lacks ``W_SEQID``: the opdb
    names the sequence.

    Parameters
    ----------
    visits : iterable of `int`
        The visits of `TWILIGHT_VISITS` to return; each has the same numbers
        whichever others are asked for.
    fibersPerDetector : `int`
        Fibers on each spectrograph.
    seed : `int`
        Seed of the noise.
    """
    rng = np.random.default_rng(seed)
    numFibers = len(TWILIGHT_SPECTROGRAPHS) * fibersPerDetector
    fiberId = np.arange(1, numFibers + 1)
    spectrograph = np.repeat(TWILIGHT_SPECTROGRAPHS, fibersPerDetector)
    angle, radius = rng.uniform(0, 2 * np.pi, numFibers), 220 * np.sqrt(rng.uniform(0, 1, numFibers))
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    throughput = {arm: rng.normal(1, 0.05, numFibers) for arm in TWILIGHT_ARMS}
    numGood = 3000
    rows = []
    for index, visit in enumerate(TWILIGHT_VISITS):
        snr = 30.0 + 15.0 * index
        sky = 1 + 0.002 * index * x / 100 + 0.001 * y / 100
        for arm in TWILIGHT_ARMS:
            noise = rng.normal(0, np.sqrt(np.pi / 2) / (snr * np.sqrt(numGood)), numFibers)
            flux = 100.0 * (index + 1) * throughput[arm] * sky * (1 + noise)
            if (visit, arm) == TWILIGHT_STEP[:2]:
                flux = np.where(spectrograph == TWILIGHT_STEP[2], flux * TWILIGHT_STEP[3], flux)
            if visit not in visits:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "visit": visit,
                        "arm": arm,
                        "spectrograph": spectrograph,
                        "sequence": -1,
                        "seqName": "Twilight sky",
                        "fiberId": fiberId,
                        "x": x,
                        "y": y,
                        "flux": flux,
                        "snr": snr,
                        "nGood": numGood,
                        "satFraction": 0.0,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def twilightIq(visits=TWILIGHT_VISITS) -> pd.DataFrame:
    """Return the ``iqQaMetrics`` rows of the twilights: what marks a visit as one for the load."""
    rows = [
        {
            "visit": visit,
            "arm": arm,
            "spectrograph": spectrograph,
            "obsType": "twilight",
            "seqName": "Twilight sky",
            "illumination": "full",
            "traceOnly": True,
            "medFwhm": 3.0,
            "linesPerFiber": 4000.0,
        }
        for visit in visits
        for arm in TWILIGHT_ARMS
        for spectrograph in TWILIGHT_SPECTROGRAPHS
    ]
    return pd.DataFrame(rows)


def twilightButler(measured=TWILIGHT_VISITS, reduced=TWILIGHT_VISITS, flux=None, **kwargs) -> StubButler:
    """Return a Butler holding the QA tables of the visits ``measured``, and a pfsArm for those ``reduced``.

    ``flux`` replaces the per-fiber table of the visits measured.
    """
    fibers = twilightFibers(measured) if flux is None else flux
    tables = {
        "iqQaMetrics": twilightIq(measured),
        "pfsArm": twilightIq(reduced)[["visit", "arm", "spectrograph"]],
    }
    if len(fibers):
        tables["twilightFiberFlux"] = fibers
    return StubButler(tables, datasets={"packages": {"drp_qa": "w.2026.44"}}, **kwargs)


def twilightSequence(
    visits=TWILIGHT_VISITS, name="Twilight sky", finished="2026-10-26 05:15:00", sequence=TWILIGHT_SEQUENCE
) -> pd.DataFrame:
    """Rows as `pfs.drp.qa.comparison.queries.readSequenceOfVisit` returns them; none for a visit in no sequence."""
    return pd.DataFrame(
        {
            "pfs_visit_id": list(visits),
            "iic_sequence_id": sequence,
            "sequence_type": "scienceObject",
            "sequence_name": name,
            "finished_at": pd.to_datetime(finished),
            "sequence_status": "complete" if finished else None,
        }
    )


class Sequences:
    """Stand in for `readSequenceOfVisit`: one answer, and a record of the visits asked about."""

    def __init__(self, answer=None):
        self.answer = twilightSequence() if answer is None else answer
        self.calls = []

    def __call__(self, opdb, visit):
        self.calls.append(int(visit))
        return self.answer.copy()


class VisitsOnlyOpDB:
    """An opdb that knows no visit's description, and refuses any query but `ensureVisits`'."""

    def __init__(self):
        self.queries = []

    def query_dataframe(self, sql, /, *, params=None, conn=None):
        self.queries.append(sql)
        assert "FROM pfs_visit" in sql and "iic_sequence" not in sql, sql
        return pd.DataFrame(columns=["pfs_visit_id", "pfs_visit_description", "pfs_design_id", "issued_at"])


@pytest.fixture
def twilight():
    """Return the synthetic twilight sequence's makers and stand-ins, by name."""
    return SimpleNamespace(
        SEQUENCE=TWILIGHT_SEQUENCE,
        VISITS=list(TWILIGHT_VISITS),
        ARMS=TWILIGHT_ARMS,
        SPECTROGRAPHS=TWILIGHT_SPECTROGRAPHS,
        STEP=TWILIGHT_STEP,
        fibers=twilightFibers,
        iq=twilightIq,
        butler=twilightButler,
        sequence=twilightSequence,
        Sequences=Sequences,
        OpDB=VisitsOnlyOpDB,
    )


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
    return {
        "data_processing_pipe2d": TEST_PROCESSINGS,
        "pfs_visit": TEST_VISITS,
        "qa_normal_value": TEST_PROCESSINGS,
    }
