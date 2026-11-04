"""Tests for the twilight verdicts in qadb (INSTRM-3089): the rows, their write, and ``bin.src/twilightQa.py``.

The metrics are real ones, ``tests/metrics/data/twilightMetrics.parquet`` (engineering twilights, by visit
id), cut to one sequence; the Butler and the opdb are stubbed, the database is the ``qadb`` fixture.
"""

import importlib.util
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.twilight import METRIC_COLUMNS, VERDICT_COLUMNS, twilightVerdict
from pfs.drp.qa.qadb.tables import snakeCase
from pfs.drp.qa.qadb.twilight import (
    TWILIGHT_METRIC_KEYS,
    TWILIGHT_QA,
    TWILIGHT_QA_METRIC,
    TWILIGHT_VERDICT_KEYS,
    drpQaVersion,
    metricRows,
    verdictRows,
    writeTwilight,
)
from pfs.drp.qa.qadb.writer import records, upsert

METRICS = Path(__file__).parents[1] / "metrics" / "data" / "twilightMetrics.parquet"
SCRIPT = Path(__file__).parents[2] / "bin.src" / "twilightQa.py"
SEQUENCE = 54787
VISITS = [134334, 134335, 134336, 134337]
NOW = datetime(2026, 11, 3, 12, 0, 0)


@pytest.fixture(scope="module")
def metrics():
    table = pd.read_parquet(METRICS)
    return table[table["sequence"] == SEQUENCE].reset_index(drop=True)


@pytest.fixture(scope="module")
def verdict(metrics):
    return twilightVerdict(metrics)


class TestRows:
    def testVerdictColumns(self, verdict):
        rows = verdictRows(verdict, "u/test/twilight", "w.2026.44", NOW)
        assert list(rows.columns) == [
            "pfs_visit_id",
            "arm",
            "iic_sequence_id",
            "qa_status",
            "qa_reason",
            "twilight_arm_mean",
            "twilight_arm_half_diff",
            "collection_name",
            "drp_qa_version",
            "judged_at",
        ]
        assert list(rows.columns[:2]) == list(TWILIGHT_VERDICT_KEYS)
        assert len(rows) == len(verdict)
        assert sorted(rows["pfs_visit_id"].unique()) == VISITS
        assert set(rows["iic_sequence_id"]) == {SEQUENCE}
        assert (rows["collection_name"] == "u/test/twilight").all()
        assert (rows["drp_qa_version"] == "w.2026.44").all()
        assert (rows["judged_at"] == NOW).all()
        # The same verdict, by another name.
        first = verdict.iloc[0]
        assert rows.iloc[0]["qa_status"] == first["status"]
        assert rows.iloc[0]["qa_reason"] == first["reason"]

    def testMetricColumns(self, metrics):
        rows = metricRows(metrics)
        assert list(rows.columns[:2]) == list(TWILIGHT_METRIC_KEYS)
        expected = {"visit": "pfs_visit_id", "sequence": "iic_sequence_id"}
        assert list(rows.columns[2:]) == [
            snakeCase(expected.get(column, column)) for column in METRIC_COLUMNS if column != "visit"
        ]
        assert len(rows) == len(metrics)
        row = rows.iloc[0]
        assert row["camera_name"] == f"{row['arm']}{row['spectrograph']}"
        assert row["pfs_visit_id"] == metrics.iloc[0]["visit"]

    def testNonFiniteBecomesNull(self, metrics):
        """A detector with no measured fiber has an infinite level offset; qadb gets a null."""
        broken = metrics.copy()
        broken.loc[0, "twilightLevelOffset"] = np.inf
        bound = records(metricRows(broken))
        assert bound[0]["twilight_level_offset"] is None
        # Negative control: the value really was infinite, not already missing.
        assert np.isinf(broken.loc[0, "twilightLevelOffset"])
        assert bound[1]["twilight_level_offset"] is not None

    def testEveryVerdictColumnIsMapped(self):
        """Nothing in `VERDICT_COLUMNS` is dropped on the way to qadb."""
        empty = pd.DataFrame(columns=list(VERDICT_COLUMNS))
        rows = verdictRows(empty, None, None, NOW)
        assert len(rows.columns) == len(VERDICT_COLUMNS) + 3

    def testVersionIsAStringOrNone(self):
        assert drpQaVersion() is None or isinstance(drpQaVersion(), str)


class TestWriteTwilight:
    def testCountsAndContent(self, qadb, verdict, metrics):
        counts = writeTwilight(qadb, verdict, metrics, "u/test/twilight", version="v", now=lambda: NOW)
        assert counts == {"pfs_visit": 4, TWILIGHT_QA: len(verdict), TWILIGHT_QA_METRIC: len(metrics)}
        assert qadb.count(TWILIGHT_QA) == len(verdict)
        assert qadb.count(TWILIGHT_QA_METRIC) == len(metrics)
        assert qadb.count("pfs_visit", f"pfs_visit_id IN ({', '.join(map(str, VISITS))})") == 4
        stored = qadb.query_dataframe(
            f"SELECT pfs_visit_id, arm, qa_status, drp_qa_version FROM {TWILIGHT_QA} ORDER BY pfs_visit_id, arm"
        )
        expected = verdict.sort_values(["visit", "arm"])
        assert list(stored["qa_status"]) == list(expected["status"])
        assert set(stored["drp_qa_version"]) == {"v"}

    def testWritingTwiceReplaces(self, qadb, verdict, metrics):
        writeTwilight(qadb, verdict, metrics, "c", version="v")
        writeTwilight(qadb, verdict, metrics, "c", version="v2")
        assert qadb.count(TWILIGHT_QA) == len(verdict)
        assert qadb.count(TWILIGHT_QA_METRIC) == len(metrics)
        assert qadb.query_scalars(f"SELECT DISTINCT drp_qa_version FROM {TWILIGHT_QA}") == ["v2"]

    def testRewriteDropsWhatTheNewJudgementLacks(self, qadb, verdict, metrics):
        """An arm no longer judged loses its rows: the visits' rows are replaced, not merged."""
        writeTwilight(qadb, verdict, metrics, "c", version="v")
        noRed = verdict[verdict["arm"] != "r"]
        noRedMetrics = metrics[metrics["arm"] != "r"]
        # Negative control: an upsert alone keeps the r rows.
        upsert(qadb, TWILIGHT_QA, verdictRows(noRed, "c", "v", NOW), list(TWILIGHT_VERDICT_KEYS))
        assert qadb.count(TWILIGHT_QA, "arm = 'r'") == (verdict["arm"] == "r").sum()
        writeTwilight(qadb, noRed, noRedMetrics, "c", version="v")
        assert qadb.count(TWILIGHT_QA, "arm = 'r'") == 0
        assert qadb.count(TWILIGHT_QA) == len(noRed)
        assert qadb.count(TWILIGHT_QA_METRIC) == len(noRedMetrics)

    def testNothingToWrite(self, qadb, verdict, metrics):
        counts = writeTwilight(qadb, verdict.iloc[:0], metrics.iloc[:0], "c")
        assert counts == {}
        assert qadb.count(TWILIGHT_QA) == 0

    def testVisitsOfTheMetricsAloneAreEnsured(self, qadb, metrics):
        """An arm the thresholds leave out has metrics and no verdict; its visits still get pfs_visit rows."""
        onlyN = metrics[metrics["arm"] == "n"]
        empty = pd.DataFrame(columns=list(VERDICT_COLUMNS))
        counts = writeTwilight(qadb, empty, onlyN, "c")
        assert counts[TWILIGHT_QA] == 0 and counts[TWILIGHT_QA_METRIC] == len(onlyN)


# --- bin.src/twilightQa.py ------------------------------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location("twilightQa", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def opdbSequence(
    visits=tuple(VISITS), name="Twilight sky", finished="2025-03-22 05:40:51", sequence=SEQUENCE
):
    """Rows as `readSequenceOfVisit` returns them; none for a visit in no sequence."""
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


class FakeOpDB:
    """Records how it was made; answers `ensureVisits` with no known visits."""

    made: ClassVar[list] = []

    def __init__(self, **kwargs):
        FakeOpDB.made.append(kwargs)

    def query_dataframe(self, sql, params=None):
        return pd.DataFrame(columns=["pfs_visit_id", "pfs_visit_description", "pfs_design_id", "issued_at"])


class FakeButler:
    made: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        FakeButler.made.append((args, kwargs))


@pytest.fixture
def cli(monkeypatch, metrics):
    """Return a runner of the script with the stack, the opdb and the readers stubbed; it takes the opdb's answer."""
    lsst = types.ModuleType("lsst")
    daf = types.ModuleType("lsst.daf")
    butler = types.ModuleType("lsst.daf.butler")
    butler.Butler = FakeButler
    monkeypatch.setitem(sys.modules, "lsst", lsst)
    monkeypatch.setitem(sys.modules, "lsst.daf", daf)
    monkeypatch.setitem(sys.modules, "lsst.daf.butler", butler)
    monkeypatch.setattr("pfs.utils.database.opdb.OpDB", FakeOpDB)
    FakeOpDB.made.clear()
    FakeButler.made.clear()

    cameras = {"b1", "r1", "n1", "b2", "r2", "n2", "b3", "r3", "n3", "b4", "r4", "n4"}
    state = {"sequence": opdbSequence(), "stored": dict.fromkeys(VISITS, cameras)}

    monkeypatch.setattr(
        "pfs.drp.qa.comparison.queries.readSequenceOfVisit", lambda opdb, visit: state["sequence"]
    )
    monkeypatch.setattr(
        "pfs.drp.qa.metrics.readers.datasetDetectors",
        lambda butler, datasetType, visits: (
            dict.fromkeys(visits, cameras)
            if datasetType == "pfsArm"
            else {v: set(c) for v, c in state["stored"].items()}
        ),
    )
    monkeypatch.setattr(
        "pfs.drp.qa.metrics.readers.readMetrics",
        lambda butler, visits, datasetType, progress: pd.DataFrame({"visit": visits, "sequence": -1}),
    )
    script = _script()
    # The fibers are stubbed, so the measurement is too: the fixture's metrics of this sequence.
    monkeypatch.setattr(script, "sequenceConsistency", lambda fibers, minSnr: metrics)

    def run(*extra, **changes):
        state.update(changes)
        argv = [
            "-c",
            "qaActor/reductions",
            "--sequence-of-visit",
            "134337",
            "--opdb-host",
            "opdb-host",
            *extra,
        ]
        return script.main(argv)

    run.script = script
    return run


class TestSequenceOfVisit:
    def testNotInASequence(self, cli, capsys):
        assert cli(sequence=opdbSequence(visits=())) == cli.script.EXIT_NOT_TWILIGHT
        assert "in no IIC sequence" in capsys.readouterr().out
        assert not FakeButler.made  # decided from the opdb alone

    def testNotATwilight(self, cli, capsys):
        assert cli(sequence=opdbSequence(name="Arc: HgCd")) == cli.script.EXIT_NOT_TWILIGHT
        assert "is not a twilight" in capsys.readouterr().out

    def testNotFinished(self, cli, capsys):
        assert cli(sequence=opdbSequence(finished=None)) == cli.script.EXIT_PENDING
        assert "has not finished" in capsys.readouterr().out

    def testWaitingForATable(self, cli, capsys):
        stored = {v: {"b1", "r1", "n1", "b2", "r2", "n2", "b3", "r3", "n3", "b4", "r4", "n4"} for v in VISITS}
        stored[134337] = stored[134337] - {"b1"}
        assert cli(stored=stored) == cli.script.EXIT_PENDING
        assert "waiting for 134337 (b1)" in capsys.readouterr().out

    def testJudgedWithoutADatabase(self, cli, capsys, tmp_path):
        assert cli("-o", str(tmp_path)) == cli.script.EXIT_JUDGED
        out = capsys.readouterr().out
        assert "all stored" in out and "PASS" in out
        assert (tmp_path / f"twilightVerdict-{SEQUENCE}.csv").exists()
        assert (tmp_path / f"twilightMetrics-{SEQUENCE}.parquet").exists()
        assert FakeOpDB.made == [{"host": "opdb-host", "port": 5432, "dbname": "opdb", "user": "public_user"}]

    def testNothingOnDiskUnlessAsked(self, cli, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert cli() == cli.script.EXIT_JUDGED
        assert list(tmp_path.iterdir()) == []

    def testJudgedAndWritten(self, cli, qadb, monkeypatch, verdict, metrics):
        made = []

        def fakeQaDB(**kwargs):
            made.append(kwargs)
            return qadb

        monkeypatch.setattr("pfs.utils.database.qadb.QaDB", fakeQaDB)
        assert cli("--qadb-host", "qadb-host") == cli.script.EXIT_JUDGED
        assert made == [{"host": "qadb-host", "port": 5436, "user": "pfs", "dbname": "qadb"}]
        assert qadb.count(TWILIGHT_QA) == len(verdict)
        assert qadb.count(TWILIGHT_QA_METRIC) == len(metrics)
        assert qadb.query_scalars(f"SELECT DISTINCT collection_name FROM {TWILIGHT_QA}") == [
            "qaActor/reductions"
        ]

    def testAFailedWriteIsPrintedInFull(self, cli, monkeypatch, capsys, verdict):
        monkeypatch.setattr("pfs.utils.database.qadb.QaDB", lambda **kwargs: object())

        def broken(*args, **kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr("pfs.drp.qa.qadb.twilight.writeTwilight", broken)
        assert cli("--qadb-host", "qadb-host") == cli.script.EXIT_NOT_WRITTEN
        out = capsys.readouterr().out
        assert "qadb write failed" in out and "connection refused" in out
        assert "Not written, twilightVerdict:" in out and "Not written, twilightMetrics:" in out
        # Every verdict row is in the log, with its reason column, so nothing is lost.
        for visit in VISITS:
            assert str(visit) in out
        assert out.count("Twilight sky") >= len(verdict)  # the metrics' seqName, once per detector row

    def testExitCodesAreDistinct(self, cli):
        script = cli.script
        codes = {script.EXIT_JUDGED, script.EXIT_PENDING, script.EXIT_NOT_TWILIGHT, script.EXIT_NOT_WRITTEN}
        assert len(codes) == 4 and script.EXIT_JUDGED == 0
