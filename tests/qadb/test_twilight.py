"""Tests for the twilight verdict in qadb: ``twilight_qa``, its place in the gate, and the CLI.

The metrics are real ones, ``tests/metrics/data/twilightMetrics.parquet`` (engineering twilights, by visit
id), cut to one sequence; the Butler and the opdb are stubbed, the database is the ``qadb`` fixture.
"""

import importlib.util
import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.twilight import IMAGE_VERDICT_COLUMNS, twilightImageVerdict
from pfs.drp.qa.qadb.loader import loadCollection
from pfs.drp.qa.qadb.tables import KEYS, QA_GATE, TASK_TABLES, TWILIGHT_QA, imageRows
from pfs.drp.qa.qadb.writer import records

METRICS = Path(__file__).parents[1] / "metrics" / "data" / "twilightMetrics.parquet"
SCRIPT = Path(__file__).parents[2] / "bin.src" / "twilightQa.py"
SEQUENCE = 54787
VISITS = [134334, 134335, 134336, 134337]
COLLECTION = "u/test/twilight"
PACKAGES = {"packages": {"drp_qa": "w.2026.44"}}


@pytest.fixture(scope="module")
def metrics():
    table = pd.read_parquet(METRICS)
    return table[table["sequence"] == SEQUENCE].reset_index(drop=True)


@pytest.fixture(scope="module")
def images(metrics):
    return twilightImageVerdict(metrics)


def quiet(_):
    pass


class TestSpec:
    def testTwilightIsAThirdTask(self):
        assert TASK_TABLES["twilightQa"] == (TWILIGHT_QA.source, None)
        assert not any("thresholds" in column for column in QA_GATE)  # the layers table cites them

    def testColumns(self, images):
        rows = imageRows(images, TWILIGHT_QA, 7)
        assert list(rows.columns[:3]) == list(KEYS)
        assert "iic_sequence_id" in rows.columns and "sequence" not in rows.columns
        assert set(rows["iic_sequence_id"]) == {SEQUENCE}
        assert set(rows["processing_id"]) == {7}
        assert rows["camera_name"].iloc[0] == f"{images['arm'].iloc[0]}{images['spectrograph'].iloc[0]}"
        assert list(rows["qa_status"]) == list(images["qaStatus"])
        # Every image column of the dataset reaches qadb, except the thresholds record, which the gate cites.
        unmapped = set(IMAGE_VERDICT_COLUMNS) - set(TWILIGHT_QA.columns) - {"visit", "arm", "spectrograph"}
        assert unmapped == {"thresholdsFiles", "thresholdsSha256"}

    def testNonFiniteBecomesNull(self, images):
        """A detector with no measured fiber has an infinite level offset; qadb gets a null."""
        broken = images.copy()
        broken.loc[0, "twilightLevelOffset"] = np.inf
        bound = records(imageRows(broken, TWILIGHT_QA, 1))
        assert bound[0]["twilight_level_offset"] is None
        # Negative control: the value really was infinite, not already missing.
        assert np.isinf(broken.loc[0, "twilightLevelOffset"])
        assert bound[1]["twilight_level_offset"] is not None


class TestLoad:
    """The loader treats the stored verdict as a third task of the per-image gate."""

    @pytest.fixture
    def butler(self, StubButler, images):
        return StubButler({"twilightQaMetrics": images}, datasets=PACKAGES)

    def load(self, butler, qadb):
        return loadCollection(butler, qadb, COLLECTION, visits=VISITS, pipelineVersions=False, log=quiet)

    def testCountsAndGate(self, butler, qadb, images):
        counts = self.load(butler, qadb)
        assert counts["twilight_qa"] == len(images) == counts["qa_gate"]
        assert counts["pfs_visit"] == 4 and counts["processings"] == 1
        assert qadb.count("qa_thresholds") == 1  # the shipped twilight thresholds, registered from the record
        gate = qadb.query_dataframe("SELECT * FROM qa_gate ORDER BY pfs_visit_id, camera_name")
        assert (gate["decided_by_task"] == "twilightQa").all()
        layers = qadb.query_dataframe("SELECT task, layer, thresholds_id FROM qa_processing_thresholds")
        assert len(layers) == 1 and layers.iloc[0].tolist()[:2] == [
            "twilightQa",
            0,
        ]  # once for the judge's RUN
        assert layers["thresholds_id"].notna().all()
        expected = images.assign(camera=images["arm"] + images["spectrograph"].astype(str))
        expected = expected.set_index(["visit", "camera"])["qaStatus"]
        for row in gate.itertuples():
            assert row.qa_status == expected[(row.pfs_visit_id, row.camera_name)]
        # The n arm is deliberately unjudged, and says so.
        assert set(gate.loc[gate["camera_name"].str.startswith("n"), "qa_status"]) == {"UNKNOWN"}
        assert set(gate.loc[~gate["camera_name"].str.startswith("n"), "qa_status"]) <= {
            "PASS",
            "WARN",
            "FAIL",
        }

    def testLoadingTwiceChangesNothing(self, butler, qadb):
        first = self.load(butler, qadb)
        second = self.load(butler, qadb)
        assert first == second
        assert qadb.count("twilight_qa") == first["twilight_qa"]

    def testBesideTheOtherTasks(self, StubButler, qadb, images, fixtures):
        """A twilight image judged by imageQualityQa too gets the worst of both in one gate row.

        The two verdicts come from different RUNs, as on the server: pipetask's for imageQualityQa and
        the judge's for twilightQaMetrics. The gate must still be one row per image, under the
        reduction's processing, so the sequence's reload replaces the per-visit load's row.
        """
        iq = fixtures["iqQaMetrics"].iloc[:1].copy()
        iq["visit"], iq["arm"], iq["spectrograph"] = VISITS[1], "b", 1
        iq["qaStatus"], iq["qaReason"] = "FAIL", "medFwhm 9 px"
        judgeRun = f"{COLLECTION}/twilightQaMetrics/20261104T194826Z"
        runs = {
            ("twilightQaMetrics", int(v), a, int(s)): judgeRun
            for v, a, s in images[["visit", "arm", "spectrograph"]].itertuples(index=False)
        }
        butler = StubButler({"twilightQaMetrics": images, "iqQaMetrics": iq}, runs=runs, datasets=PACKAGES)
        counts = self.load(butler, qadb)
        assert counts["processings"] == 2
        # One gate row per image, not one per RUN: 47 images, although the b1 image has rows in two RUNs.
        assert counts["qa_gate"] == len(images) == qadb.count("qa_gate")
        rows = qadb.query_dataframe(
            "SELECT g.*, p.run_name FROM qa_gate g JOIN data_processing_pipe2d p USING (processing_id)"
            " WHERE pfs_visit_id = :v AND camera_name = 'b1'",
            params={"v": VISITS[1]},
        )
        assert len(rows) == 1
        row = rows.iloc[0]
        assert row["qa_status"] == "FAIL" and row["decided_by_task"] == "imageQualityQa"
        assert "imageQualityQa: medFwhm 9 px" in row["qa_reason"]
        cited = qadb.query_dataframe(
            "SELECT p.processing_id, p.task FROM qa_processing_thresholds p"
            " JOIN twilight_qa t ON t.processing_id = p.processing_id WHERE t.pfs_visit_id = :v",
            params={"v": VISITS[1]},
        )
        assert set(cited["task"]) == {"twilightQa"}  # cited under the judge's own RUN, not the gate row's
        # Keyed by the reduction's processing (the first task's RUN), not the judge's.
        assert row["run_name"] != judgeRun
        # Negative control: the twilight rows themselves sit under the judge's processing.
        twilightRuns = qadb.query_scalars(
            "SELECT DISTINCT p.run_name FROM twilight_qa t JOIN data_processing_pipe2d p USING (processing_id)"
        )
        assert twilightRuns == [judgeRun]
        # Negative control: the other b1 images, judged by the twilight check alone, are not FAIL.
        others = qadb.query_dataframe(
            "SELECT qa_status FROM qa_gate WHERE camera_name = 'b1' AND pfs_visit_id != :v",
            params={"v": VISITS[1]},
        )
        assert "FAIL" not in set(others["qa_status"])


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


@pytest.fixture
def cli(monkeypatch, metrics, StubButler):
    """Return a runner of the script with the stack, the opdb and the readers stubbed; it takes the opdb's answer.

    Every ``Butler(...)`` the script opens is a stub over one shared table store, so what the (stubbed)
    store writes, the loader then reads.
    """
    shared = {}

    class FakeButler(StubButler):
        made: ClassVar[list] = []

        def __init__(self, *args, **kwargs):
            FakeButler.made.append((args, kwargs))
            super().__init__(shared, datasets=PACKAGES)

    stored = []

    def fakeStore(butler, collection, datasetType, table, **kwargs):
        stored.append((collection, datasetType, len(table)))
        shared[datasetType] = table
        return f"{collection}/{datasetType}/run"

    lsst = types.ModuleType("lsst")
    daf = types.ModuleType("lsst.daf")
    butlerModule = types.ModuleType("lsst.daf.butler")
    butlerModule.Butler = FakeButler
    monkeypatch.setitem(sys.modules, "lsst", lsst)
    monkeypatch.setitem(sys.modules, "lsst.daf", daf)
    monkeypatch.setitem(sys.modules, "lsst.daf.butler", butlerModule)
    monkeypatch.setattr("pfs.utils.database.opdb.OpDB", FakeOpDB)
    monkeypatch.setattr("pfs.drp.qa.qadb.store.storeImageTable", fakeStore)
    FakeOpDB.made.clear()

    cameras = {f"{arm}{sm}" for arm in "brn" for sm in (1, 2, 3, 4)}
    state = {"sequence": opdbSequence(), "stored": dict.fromkeys(VISITS, cameras)}

    import pfs.drp.qa.metrics.readers as readers

    realReadMetrics = readers.readMetrics

    def fakeReadMetrics(butler, visits=None, datasetType="iqQaMetrics", **kwargs):
        if datasetType == "twilightFiberFlux":
            return pd.DataFrame({"visit": list(visits), "sequence": -1})
        return realReadMetrics(butler, visits, datasetType, **kwargs)

    monkeypatch.setattr(
        "pfs.drp.qa.comparison.queries.readSequenceOfVisit", lambda opdb, visit: state["sequence"]
    )
    monkeypatch.setattr(
        "pfs.drp.qa.metrics.readers.datasetDetectors",
        lambda butler, datasetType, visits: dict.fromkeys(visits, cameras)
        if datasetType == "pfsArm"
        else {v: set(c) for v, c in state["stored"].items()},
    )
    monkeypatch.setattr("pfs.drp.qa.metrics.readers.readMetrics", fakeReadMetrics)
    script = _script()
    # The fibers are stubbed, so the measurement is too: the fixture's metrics of this sequence.
    monkeypatch.setattr(script, "sequenceConsistency", lambda fibers, minSnr: metrics)

    def run(*extra, **changes):
        state.update(changes)
        argv = ["-c", COLLECTION, "--sequence-of-visit", "134337", "--opdb-host", "opdb-host", *extra]
        return script.main(argv)

    run.script = script
    run.stored = stored
    run.butlers = FakeButler.made
    run.shared = shared
    return run


class TestSequenceOfVisit:
    def testNotInASequence(self, cli, capsys):
        assert cli(sequence=opdbSequence(visits=())) == cli.script.EXIT_NOT_TWILIGHT
        assert "in no IIC sequence" in capsys.readouterr().out
        assert not cli.butlers  # decided from the opdb alone

    def testNotATwilight(self, cli, capsys):
        assert cli(sequence=opdbSequence(name="Arc: HgCd")) == cli.script.EXIT_NOT_TWILIGHT
        assert "is not a twilight" in capsys.readouterr().out
        assert not cli.stored

    def testNotFinished(self, cli, capsys):
        assert cli(sequence=opdbSequence(finished=None)) == cli.script.EXIT_PENDING
        assert "has not finished" in capsys.readouterr().out
        assert not cli.stored

    def testWaitingForATable(self, cli, capsys):
        stored = {v: {f"{arm}{sm}" for arm in "brn" for sm in (1, 2, 3, 4)} for v in VISITS}
        stored[134337] = stored[134337] - {"b1"}
        assert cli(stored=stored) == cli.script.EXIT_PENDING
        assert "waiting for 134337 (b1)" in capsys.readouterr().out
        assert not cli.stored

    def testJudgedAndStored(self, cli, capsys, metrics):
        assert cli() == cli.script.EXIT_JUDGED
        out = capsys.readouterr().out
        assert "all stored" in out and "PASS" in out
        assert cli.stored == [(COLLECTION, "twilightQaMetrics", len(metrics))]
        # The store got a writeable Butler; the read Butler is read-only.
        assert any(kwargs.get("writeable") is True for _, kwargs in cli.butlers)
        assert FakeOpDB.made == [{"host": "opdb-host", "port": 5432, "dbname": "opdb", "user": "public_user"}]

    def testAWarnOrFailExitsJudgedBad(self, cli, capsys, monkeypatch, metrics):
        table = pd.read_parquet(METRICS)
        failing = table[table["sequence"] == 44625].reset_index(drop=True)
        monkeypatch.setattr(cli.script, "sequenceConsistency", lambda fibers, minSnr: failing)
        assert cli() == cli.script.EXIT_JUDGED_BAD
        assert "FAIL" in capsys.readouterr().out
        assert cli.stored == [(COLLECTION, "twilightQaMetrics", len(failing))]  # stored all the same
        # Negative control: the fixture's own sequence has no WARN or FAIL, and exits judged.
        monkeypatch.setattr(cli.script, "sequenceConsistency", lambda fibers, minSnr: metrics)
        assert cli("--no-store") == cli.script.EXIT_JUDGED

    def testTheExitStatusesAreDistinct(self, cli):
        statuses = {
            cli.script.EXIT_JUDGED,
            cli.script.EXIT_PENDING,
            cli.script.EXIT_NOT_TWILIGHT,
            cli.script.EXIT_NOT_WRITTEN,
            cli.script.EXIT_JUDGED_BAD,
        }
        assert len(statuses) == 5 and cli.script.EXIT_JUDGED == 0

    def testTheOpdbAccountFollowsTheOptions(self, cli):
        cli("--opdb-port", "5433", "--opdb-user", "pfs", "--opdb-dbname", "opdb_copy")
        assert FakeOpDB.made == [{"host": "opdb-host", "port": 5433, "dbname": "opdb_copy", "user": "pfs"}]

    def testStoredRowsAreThePerImageVerdict(self, cli, metrics):
        cli()
        table = cli.shared["twilightQaMetrics"]
        assert list(table.columns) == list(IMAGE_VERDICT_COLUMNS)
        assert len(table) == len(metrics)
        assert table["thresholdsFiles"].iloc[0] == "twilightThresholds.yaml"

    def testNoStore(self, cli):
        assert cli("--no-store") == cli.script.EXIT_JUDGED
        assert not cli.stored

    def testNoStoreWithQadbIsRefused(self, cli):
        """Loading would copy what the collection already holds, not this judgement."""
        with pytest.raises(SystemExit):
            cli("--no-store", "--qadb-host", "qadb-host")
        assert not cli.stored

    def testFiles(self, cli, tmp_path):
        assert cli("-o", str(tmp_path)) == cli.script.EXIT_JUDGED
        assert (tmp_path / f"twilightVerdict-{SEQUENCE}.csv").exists()
        assert (tmp_path / f"twilightMetrics-{SEQUENCE}.parquet").exists()

    def testNothingOnDiskUnlessAsked(self, cli, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert cli() == cli.script.EXIT_JUDGED
        assert list(tmp_path.iterdir()) == []

    def testJudgedStoredAndLoaded(self, cli, qadb, monkeypatch, metrics, scope):
        made = []

        def fakeQaDB(**kwargs):
            made.append(kwargs)
            return qadb

        monkeypatch.setattr("pfs.utils.database.qadb.QaDB", fakeQaDB)
        assert cli("--qadb-host", "qadb-host") == cli.script.EXIT_JUDGED
        assert made == [{"host": "qadb-host", "port": 5436, "user": "pfs", "dbname": "qadb"}]
        assert qadb.count("twilight_qa") == len(metrics)
        assert qadb.count("qa_gate") == len(metrics)
        gate = qadb.query_dataframe("SELECT DISTINCT decided_by_task, mode FROM qa_gate")
        assert gate.to_dict("records") == [{"decided_by_task": "twilightQa", "mode": "realtime"}]
        # Among the test collections: a copy of qadb holds real processings too.
        assert qadb.query_scalars(
            f"SELECT DISTINCT collection_name FROM data_processing_pipe2d WHERE {scope['data_processing_pipe2d']}"
        ) == [COLLECTION]

    def testAFailedStoreIsPrintedInFull(self, cli, monkeypatch, capsys, metrics):
        def broken(*args, **kwargs):
            raise RuntimeError("datastore read-only")

        monkeypatch.setattr("pfs.drp.qa.qadb.store.storeImageTable", broken)
        assert cli() == cli.script.EXIT_NOT_WRITTEN
        out = capsys.readouterr().out
        assert "Storing or loading the judgement failed" in out and "datastore read-only" in out
        assert "Not written, twilightVerdict:" in out and "Not written, twilightMetrics:" in out
        for visit in VISITS:
            assert str(visit) in out
        assert out.count("Twilight sky") >= len(metrics)  # the metrics' seqName, once per detector row

    def testAFailedLoadIsPrintedInFull(self, cli, monkeypatch, capsys):
        monkeypatch.setattr("pfs.utils.database.qadb.QaDB", lambda **kwargs: object())
        assert cli("--qadb-host", "qadb-host") == cli.script.EXIT_NOT_WRITTEN
        out = capsys.readouterr().out
        assert "Not written, twilightVerdict:" in out
        assert cli.stored  # the Butler has it; only qadb is behind

    def testExitCodesAreDistinct(self, cli):
        script = cli.script
        codes = {script.EXIT_JUDGED, script.EXIT_PENDING, script.EXIT_NOT_TWILIGHT, script.EXIT_NOT_WRITTEN}
        assert len(codes) == 4 and script.EXIT_JUDGED == 0
