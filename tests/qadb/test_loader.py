"""Tests for loading a collection from a (stub) Butler into qadb."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import (
    configLayerName,
    sha256Of,
    taskThresholdsRecord,
    thresholdsDocument,
    thresholdsPath,
    thresholdsRecord,
)
from pfs.drp.qa.qadb.loader import (
    CONFIG_LAYERS,
    MODES,
    SPECS,
    collectionExists,
    configDocumentOf,
    drpQaVersionOf,
    loadCollection,
    loadVisit,
    readTables,
    thresholdsFilesOf,
)
from pfs.drp.qa.qadb.tables import GATE_ORDER, worstStatus
from pfs.drp.qa.qadb.writer import UNKNOWN_VERSION, registerThresholds

COLLECTION = "u/test/qa"
RUN = "u/test/qa/20261026T000000Z"
NOW = datetime(2026, 10, 26, 12, 0, 0)
VISITS = [133025, 133040]


def species(iq: pd.DataFrame) -> pd.DataFrame:
    """Return a long ``iqQaSpeciesMetrics`` table for the images of ``iq``: two species, two metrics each."""
    rows = []
    for image in iq[["visit", "arm", "spectrograph"]].itertuples(index=False):
        for description in ("ArI", "Trace"):
            for metric, value in (("fitXRms", 0.05), ("fitYRms", 0.07)):
                rows.append({**image._asdict(), "description": description, "metric": metric, "value": value})
    return pd.DataFrame(rows)


@pytest.fixture
def tables(fixtures):
    return {**fixtures, "iqQaSpeciesMetrics": species(fixtures["iqQaMetrics"])}


@pytest.fixture
def butler(StubButler, tables, configs):
    return StubButler(
        tables, datasets={"packages": {"drp_qa": "w.2026.44", "drp_stella": "w.2026.41"}, **configs}
    )


def recorded(
    tables, iq=("iqQaThresholds-run25.yaml",), dm=("dmLineResidualThresholds-run25.yaml",), configs=None
):
    """Return the tables with the thresholds layers each task records judging with, as the tasks write them.

    The files alone, as the tasks recorded before PIPE2D-1991; with ``configs`` (the ``*_config``
    datasets), each task's config layer after them, as they record now.
    """

    def record(task, names):
        if configs is None:
            return thresholdsRecord(list(names))
        return taskThresholdsRecord(task, list(names), CONFIG_LAYERS[task](configs[f"{task}_config"]))

    return {
        **tables,
        "iqQaMetrics": tables["iqQaMetrics"].assign(**record("imageQualityQa", iq)),
        "dmLineResidualSummary": tables["dmLineResidualSummary"].assign(**record("dmLineResiduals", dm)),
    }


#: The ``qa_gate_thresholds`` task of each former ``qa_gate`` column.
FIRST_LAYER = {
    "iq_thresholds_id": "imageQualityQa",
    "dm_thresholds_id": "dmLineResiduals",
    "twilight_thresholds_id": "twilightQa",
}


def firstLayers(qadb) -> pd.DataFrame:
    """Return each gate row's first layer per task: ``pfs_visit_id``, ``camera_name``, ``iq_thresholds_id``, ...

    The shape ``qa_gate`` had before the layers; null where the task cited nothing (or judged nothing).
    """
    rows = qadb.query_dataframe(
        "SELECT g.pfs_visit_id, g.camera_name, t.task, t.thresholds_id FROM qa_gate g"
        " LEFT JOIN qa_gate_thresholds t ON t.processing_id = g.processing_id"
        " AND t.pfs_visit_id = g.pfs_visit_id AND t.camera_name = g.camera_name AND t.layer = 0"
        " ORDER BY g.pfs_visit_id, g.camera_name"
    )
    frame = rows[["pfs_visit_id", "camera_name"]].drop_duplicates(ignore_index=True)
    for column, task in FIRST_LAYER.items():
        cited = rows[rows["task"] == task].set_index(["pfs_visit_id", "camera_name"])["thresholds_id"]
        frame[column] = [
            cited.get((visit, camera)) for visit, camera in frame[["pfs_visit_id", "camera_name"]].to_numpy()
        ]
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def distinct(qadb, column: str) -> list:
    """Return the distinct first-layer ids of a task's former ``qa_gate`` column, as that query returned them."""
    return sorted(set(firstLayers(qadb)[column].tolist()), key=lambda value: (value is None, value or 0))


def load(butler, qadb, **kwargs):
    kwargs.setdefault("pipelineVersions", False)
    kwargs.setdefault("now", lambda: NOW)
    return loadCollection(butler, qadb, COLLECTION, visits=VISITS, log=lambda _: None, **kwargs)


class TestReadTables:
    def testEveryTypeWithRun(self, butler):
        read = readTables(butler, COLLECTION, visits=VISITS, log=lambda _: None)
        assert set(read) == set(SPECS)
        assert len(read["iqQaMetrics"]) == 4 and (read["iqQaMetrics"]["run"] == RUN).all()
        assert len(read["dmLineResidualMetrics"]) == 128

    def testMissingTypeIsEmpty(self, StubButler, fixtures):
        butler = StubButler({"iqQaMetrics": fixtures["iqQaMetrics"]})
        read = readTables(butler, COLLECTION, visits=VISITS, log=lambda _: None)
        assert read["dmLineResidualSummary"].empty
        assert list(read["dmLineResidualSummary"].columns) == ["visit", "arm", "spectrograph", "run"]


def testHelpers(butler, StubButler):
    assert drpQaVersionOf(butler, RUN) == "w.2026.44"
    assert [p.name for p in thresholdsFilesOf(butler, RUN, "imageQualityQa")] == ["iqQaThresholds-run25.yaml"]
    bare = StubButler({})
    assert thresholdsFilesOf(bare, RUN, "imageQualityQa") == []
    assert drpQaVersionOf(bare, RUN) == UNKNOWN_VERSION  # never the loader's own version


class TestCollectionExists:
    @staticmethod
    def butlerWith(query):
        return SimpleNamespace(collections=SimpleNamespace(query=query))

    def testFound(self):
        assert collectionExists(self.butlerWith(lambda name: [name]), "u/test/qa")

    def testMissingByEmptyAnswer(self):
        assert not collectionExists(self.butlerWith(lambda name: []), "u/test/qa")

    def testMissingByError(self):
        class MissingCollectionError(Exception):
            pass

        def query(name):
            raise MissingCollectionError(name)

        assert not collectionExists(self.butlerWith(query), "u/test/qa")

    def testOtherErrorsPropagate(self):
        """Negative control: a registry failure is not reported as a missing collection."""

        def query(name):
            raise RuntimeError("registry down")

        with pytest.raises(RuntimeError, match="registry down"):
            collectionExists(self.butlerWith(query), "u/test/qa")


def testProgressGoesToTheLog(butler):
    messages = []
    readTables(butler, COLLECTION, visits=VISITS, log=messages.append)
    assert any(m.startswith("iqQaMetrics: reading 4 datasets") for m in messages)


class TestLoadCollection:
    def testCountsAndContent(self, butler, qadb, tables):
        counts = load(butler, qadb)
        assert counts == {
            "processings": 1,
            "pfs_visit": 2,
            "image_quality_qa": 4,
            "image_quality_qa_species": 16,
            "dm_line_residual_qa": 4,
            "dm_line_residual_qa_metric": 128,
            "qa_gate": 4,
            "qa_gate_thresholds": 8,  # one file per task and image; these configs have no threshold fields
        }
        processing = qadb.query_dataframe(
            "SELECT * FROM data_processing_pipe2d WHERE collection_name = :c", params={"c": COLLECTION}
        )
        assert len(processing) == 1
        assert processing.loc[0, "collection_name"] == COLLECTION
        assert processing.loc[0, "run_name"] == RUN
        assert processing.loc[0, "drp_qa_version"] == "w.2026.44"
        assert pd.isna(processing.loc[0, "pipe2d_version"])  # pipelineVersions=False
        assert qadb.count("qa_thresholds") == 2
        gate = qadb.query_dataframe("SELECT * FROM qa_gate ORDER BY pfs_visit_id, camera_name")
        assert gate["camera_name"].tolist() == ["b1", "r1", "b1", "r1"]
        first = firstLayers(qadb)
        assert first["iq_thresholds_id"].notna().all() and first["dm_thresholds_id"].notna().all()
        assert (gate["mode"] == "backfill").all()
        # The gate is the worst of the two tasks' stored verdicts, image by image.
        iq = tables["iqQaMetrics"].set_index(["visit", "arm", "spectrograph"])["qaStatus"]
        dm = tables["dmLineResidualSummary"].set_index(["visit", "arm", "spectrograph"])["qaStatus"]
        for row in gate.itertuples():
            key = (row.pfs_visit_id, row.camera_name[0], int(row.camera_name[1]))
            assert row.qa_status == worstStatus([iq[key], dm[key]])
            assert row.qa_status in GATE_ORDER

    def testLoadingTwiceChangesNothing(self, butler, qadb, tables, scope):
        first = load(butler, qadb)
        before = {table: qadb.count(table, scope.get(table, "")) for table in first if table != "processings"}
        again = load(butler, qadb)
        assert again == first
        assert {table: qadb.count(table, scope.get(table, "")) for table in before} == before
        assert qadb.count("data_processing_pipe2d", scope["data_processing_pipe2d"]) == 1
        # Negative control: a re-judged image is replaced, not duplicated.
        tables["iqQaMetrics"].loc[0, "qaStatus"] = "FAIL"
        tables["iqQaMetrics"].loc[0, "qaReason"] = "test"
        load(butler, qadb)
        assert qadb.count("image_quality_qa") == 4
        status = qadb.query_scalar(
            "SELECT qa_status FROM image_quality_qa WHERE pfs_visit_id = :v AND camera_name = :c",
            params={"v": 133025, "c": "b1"},
        )
        assert status == "FAIL"
        gate = qadb.query_scalar(
            "SELECT qa_status FROM qa_gate WHERE pfs_visit_id = :v AND camera_name = :c",
            params={"v": 133025, "c": "b1"},
        )
        assert gate == "FAIL"

    def testReloadDropsRowsTheNewJudgementLacks(self, butler, qadb, tables, scope):
        """A species or metric missing from a re-judged image is removed, not left with its old value."""
        load(butler, qadb)
        assert qadb.count("image_quality_qa_species") == 16
        assert qadb.count("dm_line_residual_qa_metric") == 128
        species = tables["iqQaSpeciesMetrics"]
        tables["iqQaSpeciesMetrics"] = species[
            ~((species["visit"] == 133025) & (species["description"] == "Trace"))
        ]
        metrics = tables["dmLineResidualMetrics"]
        tables["dmLineResidualMetrics"] = metrics[metrics["metric"] != "linesPerFiber"]
        counts = load(butler, qadb)
        assert counts["image_quality_qa_species"] == 12
        assert qadb.count("image_quality_qa_species") == 12
        assert qadb.count("dm_line_residual_qa_metric") == counts["dm_line_residual_qa_metric"] < 128
        assert qadb.count("dm_line_residual_qa_metric", "metric = 'linesPerFiber'") == 0
        # Another image's rows are untouched by a reload of fewer images.
        loadVisit(
            butler, qadb, COLLECTION, 133040, pipelineVersions=False, log=lambda _: None, now=lambda: NOW
        )
        assert qadb.count("image_quality_qa_species") == 12
        # Negative control: the per-image rows stay (one per image loaded), so the drop is in the long tables only.
        assert qadb.count("image_quality_qa") == 4

    def testReloadWithNoLongRowsLeftRemovesThem(self, butler, qadb, tables):
        """An image whose re-judgement has no species rows at all loses its old ones."""
        load(butler, qadb)
        species = tables["iqQaSpeciesMetrics"]
        tables["iqQaSpeciesMetrics"] = species[species["visit"] != 133025]
        load(butler, qadb)
        assert qadb.count("image_quality_qa_species", "pfs_visit_id = 133025") == 0
        assert qadb.count("image_quality_qa_species") == 8
        # And when the whole long dataset is gone from the collection.
        del tables["iqQaSpeciesMetrics"]
        counts = load(butler, qadb)
        assert counts["image_quality_qa_species"] == 0
        assert qadb.count("image_quality_qa_species") == 0
        assert qadb.count("image_quality_qa") == 4

    def testVersionsSurviveARealtimeReload(self, StubButler, tables, configs, qadb):
        """QaActor's loadVisit reads no pipeline versions; the backfill's stay."""
        tables = dict(tables)
        iq = tables["iqQaMetrics"].assign(calexpRun="u/test/reduce")
        tables["iqQaMetrics"] = iq
        tables["pfsArm"] = iq[["visit", "arm", "spectrograph"]]
        butler = StubButler(tables, datasets={"packages": {"drp_qa": "w.2026.44"}, **configs})
        butler.getURI = lambda ref: ref
        header = {
            "VERSION_DATAMODEL": "w.2026.40",
            "VERSION_OBS_PFS": "w.2026.40",
            "VERSION_DRP_STELLA": "w.2026.41",
        }
        load(butler, qadb, pipelineVersions=True, readHeader=lambda ref: header)
        bare = StubButler(tables, datasets=configs)  # no packages dataset this time either
        loadVisit(bare, qadb, COLLECTION, 133040, pipelineVersions=False, log=lambda _: None, now=lambda: NOW)
        row = qadb.query_dataframe(
            "SELECT * FROM data_processing_pipe2d WHERE collection_name = :c", params={"c": COLLECTION}
        ).iloc[0]
        assert row["pipe2d_version"] == "w.2026.41"
        assert row["drp_qa_version"] == "w.2026.44"

    def testThresholdsPerRun(self, StubButler, tables, configs, qadb):
        """Each RUN's gate rows cite the thresholds its own config named."""
        runs = {
            (133025, "b", 1): "r1",
            (133025, "r", 1): "r1",
            (133040, "b", 1): "r2",
            (133040, "r", 1): "r2",
        }
        other = str(Path(__file__).parents[1] / "metrics" / "data" / "dmLineResidualThresholds-run25.yaml")
        datasets = {
            "packages": {"drp_qa": "x"},
            **configs,
            ("dmLineResiduals_config", "r2"): SimpleNamespace(thresholdsFile=other),
        }
        butler = StubButler(tables, runs=runs, datasets=datasets)
        load(butler, qadb)
        gate = firstLayers(qadb)
        assert gate["iq_thresholds_id"].nunique() == 1
        assert gate.loc[gate["pfs_visit_id"] == 133025, "dm_thresholds_id"].nunique() == 1
        assert gate.loc[gate["pfs_visit_id"] == 133040, "dm_thresholds_id"].nunique() == 1
        assert set(gate["dm_thresholds_id"]).__len__() == 2  # r1 and r2 differ
        assert qadb.count("qa_thresholds") == 3

    def testExtraThresholdsFilesAreRegistered(self, StubButler, tables, configs, qadb):
        """ImageQualityQa's further layers are registered too; qa_gate cites the first."""
        iq = SimpleNamespace(
            thresholdsFile="iqQaThresholds-run25.yaml", extraThresholdsFiles=["iqQaThresholds-iis-run30.yaml"]
        )
        butler = StubButler(tables, datasets={**configs, "imageQualityQa_config": iq})
        load(butler, qadb)
        names = qadb.query_dataframe("SELECT thresholds_id, name FROM qa_thresholds ORDER BY thresholds_id")
        assert set(names["name"]) == {
            "iqQaThresholds-run25.yaml",
            "iqQaThresholds-iis-run30.yaml",
            "dmLineResidualThresholds-run25.yaml",
        }
        first = int(names.loc[names["name"] == "iqQaThresholds-run25.yaml", "thresholds_id"].iloc[0])
        assert distinct(qadb, "iq_thresholds_id") == [first]
        iis = int(names.loc[names["name"] == "iqQaThresholds-iis-run30.yaml", "thresholds_id"].iloc[0])
        assert qadb.count("qa_threshold_entry", f"thresholds_id = {iis} AND illumination = 'iis'") > 0
        assert qadb.count("qa_threshold_entry", f"thresholds_id = {first} AND illumination IS NOT NULL") == 0

    def testMissingThresholdsFileIsLogged(self, StubButler, tables, configs, qadb):
        datasets = {**configs, "dmLineResiduals_config": SimpleNamespace(thresholdsFile="gone.yaml")}
        butler = StubButler(tables, datasets=datasets)
        messages = []
        loadCollection(
            butler,
            qadb,
            COLLECTION,
            visits=VISITS,
            pipelineVersions=False,
            log=messages.append,
            now=lambda: NOW,
        )
        assert any("gone.yaml" in m and "no longer has" in m for m in messages)
        assert firstLayers(qadb)["dm_thresholds_id"].tolist() == [None] * 4
        assert distinct(qadb, "iq_thresholds_id") != [None]

    def testNeedsADatabaseOrADryRun(self, butler):
        with pytest.raises(ValueError, match="dryRun"):
            load(butler, None)

    def testDeletingAProcessingCascades(self, butler, qadb, scope):
        """Removing a processing row takes every QA row of it along (the schema's ON DELETE CASCADE)."""
        load(butler, qadb)
        qadb.commit("DELETE FROM data_processing_pipe2d WHERE collection_name = :c", params={"c": COLLECTION})
        for table in (
            "image_quality_qa",
            "image_quality_qa_species",
            "dm_line_residual_qa",
            "dm_line_residual_qa_metric",
            "qa_gate",
        ):
            assert qadb.count(table) == 0, table
        assert qadb.count("qa_thresholds") == 2  # thresholds outlive the processings that used them

    def testOneProcessingPerRun(self, StubButler, tables, configs, qadb):
        runs = {
            (133025, "b", 1): "r1",
            (133025, "r", 1): "r1",
            (133040, "b", 1): "r2",
            (133040, "r", 1): "r2",
        }
        butler = StubButler(tables, runs=runs, datasets={"packages": {"drp_qa": "x"}, **configs})
        counts = load(butler, qadb)
        assert counts["processings"] == 2
        ids = qadb.query_dataframe(
            "SELECT processing_id, run_name FROM data_processing_pipe2d WHERE collection_name = :c ORDER BY run_name",
            params={"c": COLLECTION},
        )
        assert ids["run_name"].tolist() == ["r1", "r2"]
        byRun = qadb.query_dataframe(
            "SELECT processing_id, pfs_visit_id, count(*) AS n FROM image_quality_qa_species GROUP BY 1, 2 ORDER BY 2"
        )
        # The species rows take their image's processing: visit 133025's are in r1, 133040's in r2.
        assert byRun["processing_id"].tolist() == ids["processing_id"].tolist()
        assert byRun["n"].tolist() == [8, 8]

    def testImageQualityAlone(self, StubButler, fixtures, configs, qadb):
        butler = StubButler({"iqQaMetrics": fixtures["iqQaMetrics"]}, datasets=configs)
        counts = load(butler, qadb)
        assert counts["dm_line_residual_qa"] == 0 if "dm_line_residual_qa" in counts else True
        assert counts["qa_gate"] == 4
        assert set(qadb.query_scalars("SELECT decided_by_task FROM qa_gate")) == {"imageQualityQa"}
        assert firstLayers(qadb)["dm_thresholds_id"].tolist() == [None] * 4

    def testLongRowsWithoutTheirImageAreSkipped(self, StubButler, tables, configs, qadb):
        tables = dict(tables)
        tables["iqQaMetrics"] = tables["iqQaMetrics"].iloc[:3]  # one image's species rows lose their image
        butler = StubButler(tables, datasets=configs)
        messages = []
        counts = loadCollection(
            butler,
            qadb,
            COLLECTION,
            visits=VISITS,
            pipelineVersions=False,
            log=messages.append,
            now=lambda: NOW,
        )
        assert counts["image_quality_qa_species"] == 12
        assert any("4 rows without a iqQaMetrics row" in m for m in messages)

    def testNothingToLoad(self, StubButler, qadb):
        with pytest.raises(LookupError, match=COLLECTION):
            load(StubButler({}), qadb)

    def testBadMode(self, butler, qadb):
        with pytest.raises(ValueError, match="mode"):
            load(butler, qadb, mode="nightly")
        assert "realtime" in MODES

    def testLoadVisit(self, butler, qadb):
        counts = loadVisit(
            butler, qadb, COLLECTION, 133025, pipelineVersions=False, log=lambda _: None, now=lambda: NOW
        )
        assert counts["image_quality_qa"] == 2 and counts["pfs_visit"] == 1
        assert set(qadb.query_scalars("SELECT mode FROM qa_gate")) == {"realtime"}

    def testDryRun(self, butler, tmp_path):
        counts = load(butler, None, dryRun=tmp_path)
        assert counts["qa_gate"] == 4 and "pfs_visit" not in counts
        written = {path.stem for path in tmp_path.glob("*.parquet")}
        assert written == {
            "image_quality_qa",
            "image_quality_qa_species",
            "dm_line_residual_qa",
            "dm_line_residual_qa_metric",
            "qa_gate",
            "qa_gate_thresholds",
        }
        gate = pd.read_parquet(tmp_path / "qa_gate.parquet")
        assert (gate["processing_id"] < 0).all()  # placeholders: no database was asked
        assert not any("thresholds" in column for column in gate.columns)
        assert pd.read_parquet(
            tmp_path / "qa_gate_thresholds.parquet"
        ).empty  # nothing registered, nothing cited

    def testPipelineVersions(self, StubButler, tables, configs, qadb):
        """The reduction's versions come from the pfsArm beside each judged calexp."""
        tables = dict(tables)
        iq = tables["iqQaMetrics"].assign(calexpRun="u/test/reduce")
        tables["iqQaMetrics"] = iq
        tables["pfsArm"] = iq[["visit", "arm", "spectrograph"]]
        butler = StubButler(tables, datasets={"packages": {"drp_qa": "x"}, **configs})
        butler.getURI = lambda ref: ref
        header = {
            "VERSION_DATAMODEL": "w.2026.40",
            "VERSION_OBS_PFS": "w.2026.40",
            "VERSION_DRP_STELLA": "w.2026.41",
        }
        load(butler, qadb, pipelineVersions=True, readHeader=lambda ref: header)
        row = qadb.query_dataframe(
            "SELECT * FROM data_processing_pipe2d WHERE collection_name = :c", params={"c": COLLECTION}
        ).iloc[0]
        assert row["pipe2d_version"] == "w.2026.41"
        assert row["datamodel_version"] == "w.2026.40"
        assert row["obs_pfs_version"] == "w.2026.40"

    def testUnknownPipelineStaysNull(self, butler, qadb):
        """Metrics written before calexpRun existed have no known pipeline; the load goes on."""
        messages = []
        loadCollection(
            butler,
            qadb,
            COLLECTION,
            visits=VISITS,
            pipelineVersions=True,
            log=messages.append,
            now=lambda: NOW,
        )
        row = qadb.query_dataframe(
            "SELECT * FROM data_processing_pipe2d WHERE collection_name = :c", params={"c": COLLECTION}
        ).iloc[0]
        assert pd.isna(row["pipe2d_version"])
        assert any("no single reduction pipeline" in m for m in messages)


class TestRecordedThresholds:
    """The gate cites the files each image records judging with, by content."""

    #: Today's imageQualityQa config: a layer added after the tables below were judged.
    TODAY = SimpleNamespace(
        thresholdsFile="iqQaThresholds-run25.yaml", extraThresholdsFiles=["iqQaThresholds-iis-run30.yaml"]
    )

    def gate(self, qadb):
        return firstLayers(qadb)

    def idOf(self, qadb, name):
        return qadb.query_scalar(
            "SELECT thresholds_id FROM qa_thresholds WHERE name = :n", params={"n": name}
        )

    def testRecordedFilesNotTheConfig(self, StubButler, SqliteQaDB, tables, configs, qadb):
        """A layer the RUN's config gains from today's config class is not registered."""
        datasets = {**configs, "imageQualityQa_config": self.TODAY}
        load(StubButler(recorded(tables), datasets=datasets), qadb)
        names = set(qadb.query_scalars("SELECT name FROM qa_thresholds"))
        assert names == {"iqQaThresholds-run25.yaml", "dmLineResidualThresholds-run25.yaml"}
        gate = self.gate(qadb)
        assert gate["iq_thresholds_id"].tolist() == [self.idOf(qadb, "iqQaThresholds-run25.yaml")] * 4
        assert (
            gate["dm_thresholds_id"].tolist() == [self.idOf(qadb, "dmLineResidualThresholds-run25.yaml")] * 4
        )
        # Negative control: the same tables recording nothing are read through the config, extra layer and all.
        legacy = SqliteQaDB()
        load(StubButler(tables, datasets=datasets), legacy)
        assert "iqQaThresholds-iis-run30.yaml" in set(legacy.query_scalars("SELECT name FROM qa_thresholds"))

    def testEveryLayerIsRegisteredAndTheFirstCited(self, StubButler, tables, configs, qadb):
        iq = ("iqQaThresholds-run25.yaml", "iqQaThresholds-iis-run30.yaml")
        load(StubButler(recorded(tables, iq=iq), datasets=configs), qadb)
        assert qadb.count("qa_thresholds") == 3
        assert set(self.gate(qadb)["iq_thresholds_id"]) == {self.idOf(qadb, iq[0])}
        iis = self.idOf(qadb, iq[1])
        assert qadb.count("qa_threshold_entry", f"thresholds_id = {iis}") > 0

    def testChangedFileIsCitedByTheContentThatJudged(self, StubButler, tables, configs, qadb, tmp_path):
        """A file edited since judgement is cited as qadb registered it then, not as it is now."""
        name = "iqQaThresholds-run25.yaml"
        then = tmp_path / name
        then.write_bytes(thresholdsPath(name).read_bytes() + b"\n# as it was\n")
        judgedWith = registerThresholds(qadb, then, NOW)  # an earlier load, before the file changed
        iq = recorded(tables)["iqQaMetrics"].assign(thresholdsFiles=name, thresholdsSha256=sha256Of(then))
        load(StubButler({**recorded(tables), "iqQaMetrics": iq}, datasets=configs), qadb)
        assert set(self.gate(qadb)["iq_thresholds_id"]) == {judgedWith}
        # Negative control: today's file under that name was not registered, so not cited.
        shipped = sha256Of(thresholdsPath(name))
        assert (
            qadb.query_scalar("SELECT count(*) FROM qa_thresholds WHERE sha256 = :s", params={"s": shipped})
            == 0
        )

    def testRenamedFileIsFoundByContent(self, StubButler, tables, configs, qadb):
        """A file shipped under a new name is registered under the name it judged with."""
        shipped = sha256Of(thresholdsPath("iqQaThresholds-run25.yaml"))
        iq = recorded(tables)["iqQaMetrics"].assign(
            thresholdsFiles="iqQaThresholds-old.yaml", thresholdsSha256=shipped
        )
        load(StubButler({**recorded(tables), "iqQaMetrics": iq}, datasets=configs), qadb)
        row = qadb.query_dataframe(
            "SELECT * FROM qa_thresholds WHERE sha256 = :s", params={"s": shipped}
        ).iloc[0]
        assert row["name"] == "iqQaThresholds-old.yaml"
        assert set(self.gate(qadb)["iq_thresholds_id"]) == {row["thresholds_id"]}
        assert qadb.count("qa_threshold_entry", f"thresholds_id = {row['thresholds_id']}") > 0

    def testUnknownContentIsNullAndLogged(self, StubButler, tables, configs, qadb):
        iq = recorded(tables)["iqQaMetrics"].assign(thresholdsSha256="f" * 64)
        messages = []
        loadCollection(
            StubButler({**recorded(tables), "iqQaMetrics": iq}, datasets=configs),
            qadb,
            COLLECTION,
            visits=VISITS,
            pipelineVersions=False,
            log=messages.append,
            now=lambda: NOW,
        )
        gate = self.gate(qadb)
        assert gate["iq_thresholds_id"].isna().all()
        assert gate["dm_thresholds_id"].notna().all()
        assert sum("iqQaThresholds-run25.yaml" in m and "neither in qadb" in m for m in messages) == 1

    def testNoFileRecordedIsNull(self, StubButler, tables, configs, qadb):
        """Judged by the config's fields alone: no thresholds, though the config names a file."""
        load(StubButler(recorded(tables, iq=()), datasets=configs), qadb)
        assert self.gate(qadb)["iq_thresholds_id"].isna().all()
        assert set(qadb.query_scalars("SELECT name FROM qa_thresholds")) == {
            "dmLineResidualThresholds-run25.yaml"
        }

    def testPerImage(self, StubButler, tables, configs, qadb):
        """Images of one RUN that judged with different files cite their own."""
        iq = recorded(tables)["iqQaMetrics"]
        later = iq["visit"] == 133040
        iq.loc[later, list(thresholdsRecord([]))] = list(
            thresholdsRecord(["iqQaThresholds-iis-run30.yaml"]).values()
        )
        load(StubButler({**recorded(tables), "iqQaMetrics": iq}, datasets=configs), qadb)
        gate = self.gate(qadb).set_index("pfs_visit_id")["iq_thresholds_id"]
        assert set(gate[133025]) == {self.idOf(qadb, "iqQaThresholds-run25.yaml")}
        assert set(gate[133040]) == {self.idOf(qadb, "iqQaThresholds-iis-run30.yaml")}

    def logged(self, butler, qadb):
        messages = []
        loadCollection(
            butler,
            qadb,
            COLLECTION,
            visits=VISITS,
            pipelineVersions=False,
            log=messages.append,
            now=lambda: NOW,
        )
        return messages

    def testOldAndNewTablesTogether(self, StubButler, tables, configs, qadb):
        """A RUN written before PIPE2D-1976 falls back on its config; one written after, in the same load, does not."""
        runs = {
            (133025, "b", 1): "new",
            (133025, "r", 1): "new",
            (133040, "b", 1): "old",
            (133040, "r", 1): "old",
        }
        iq = recorded(tables, iq=["iqQaThresholds-iis-run30.yaml"])["iqQaMetrics"]
        old = (iq["visit"] == 133040).to_numpy()
        iq.loc[old, ["thresholdsFiles", "thresholdsSha256"]] = (
            np.nan
        )  # as concatenation with an old table leaves it
        messages = self.logged(
            StubButler({**recorded(tables), "iqQaMetrics": iq}, runs=runs, datasets=configs), qadb
        )
        gate = self.gate(qadb).set_index("pfs_visit_id")["iq_thresholds_id"]
        assert set(gate[133025]) == {self.idOf(qadb, "iqQaThresholds-iis-run30.yaml")}
        assert set(gate[133040]) == {
            self.idOf(qadb, "iqQaThresholds-run25.yaml")
        }  # what the old RUN's config names
        fallbacks = [m for m in messages if "recorded no thresholds layers" in m]
        assert fallbacks == [
            "old: imageQualityQa recorded no thresholds layers; citing those its stored config gives,"
            " as this drp_qa has them"
        ]

    def testInterruptedRegistrationIsRepaired(self, StubButler, tables, configs, qadb):
        """A row whose entries were never written gets them on the next load of a table recording its file."""
        thresholdsId = registerThresholds(qadb, thresholdsPath("iqQaThresholds-run25.yaml"), NOW)
        complete = qadb.count("qa_threshold_entry", f"thresholds_id = {thresholdsId}")
        qadb.commit("DELETE FROM qa_threshold_entry WHERE thresholds_id = :i", params={"i": thresholdsId})
        assert (
            qadb.count("qa_threshold_entry", f"thresholds_id = {thresholdsId}") == 0
        )  # the interrupted state
        load(StubButler(recorded(tables), datasets=configs), qadb)
        assert qadb.count("qa_threshold_entry", f"thresholds_id = {thresholdsId}") == complete > 0
        assert set(self.gate(qadb)["iq_thresholds_id"]) == {thresholdsId}

    def testIncompleteRowWithoutACopyIsCitedAndLogged(self, StubButler, tables, configs, qadb, tmp_path):
        then = tmp_path / "iqQaThresholds-run25.yaml"
        then.write_bytes(thresholdsPath(then.name).read_bytes() + b"\n# as it was\n")
        thresholdsId = registerThresholds(qadb, then, NOW)
        qadb.commit("DELETE FROM qa_threshold_entry WHERE thresholds_id = :i", params={"i": thresholdsId})
        iq = recorded(tables)["iqQaMetrics"].assign(thresholdsSha256=sha256Of(then))
        messages = self.logged(StubButler({**recorded(tables), "iqQaMetrics": iq}, datasets=configs), qadb)
        assert set(self.gate(qadb)["iq_thresholds_id"]) == {thresholdsId}
        assert sum("entries missing" in m for m in messages) == 1

    def testMalformedRecordIsNullAndLogged(self, StubButler, tables, configs, qadb):
        """A record naming one file with two hashes leaves that image's id null; the load goes on."""
        iq = recorded(tables)["iqQaMetrics"]
        bad = (iq["visit"] == 133025).to_numpy()
        iq.loc[bad, "thresholdsSha256"] = "f" * 64 + "," + "e" * 64
        messages = self.logged(StubButler({**recorded(tables), "iqQaMetrics": iq}, datasets=configs), qadb)
        gate = self.gate(qadb).set_index("pfs_visit_id")
        assert gate.loc[133025, "iq_thresholds_id"].isna().all()
        assert gate.loc[133040, "iq_thresholds_id"].notna().all()
        assert gate["dm_thresholds_id"].notna().all()
        assert (
            sum("1 thresholds layers recorded with 2 SHA-256s" in m for m in messages) == 2
        )  # one per image


class TestConfigLayers:
    """A task's config fields are its last thresholds layer, registered by content and cited with the rest."""

    def layers(self, qadb):
        return qadb.query_dataframe(
            "SELECT g.pfs_visit_id, g.camera_name, g.task, g.layer, t.name FROM qa_gate_thresholds g"
            " JOIN qa_thresholds t ON t.thresholds_id = g.thresholds_id"
            " ORDER BY g.pfs_visit_id, g.camera_name, g.task, g.layer"
        )

    def logged(self, butler, qadb):
        messages = []
        loadCollection(
            butler,
            qadb,
            COLLECTION,
            visits=VISITS,
            pipelineVersions=False,
            log=messages.append,
            now=lambda: NOW,
        )
        return messages

    def testConfigLayerIsRebuiltFromTheStoredConfigAndCited(self, StubButler, tables, fullConfigs, qadb):
        """Each image cites its file, then its config layer, in ``qa_gate_thresholds``; ``qa_gate`` the file."""
        datasets = {"packages": {"drp_qa": "x"}, **fullConfigs}
        counts = load(StubButler(recorded(tables, configs=fullConfigs), datasets=datasets), qadb)
        names = qadb.query_dataframe("SELECT thresholds_id, name, n_entries FROM qa_thresholds")
        assert set(names["name"]) == {
            "iqQaThresholds-run25.yaml",
            "dmLineResidualThresholds-run25.yaml",
            "imageQualityQa.config",
            "dmLineResiduals.config",
        }
        layers = self.layers(qadb)
        assert counts["qa_gate_thresholds"] == len(layers) == 16  # 4 images x 2 tasks x 2 layers
        perImage = layers.groupby(["pfs_visit_id", "camera_name", "task"])["name"].agg(list)
        assert all(
            perImage.xs("imageQualityQa", level="task").map(tuple)
            == ("iqQaThresholds-run25.yaml", "imageQualityQa.config")
        )
        assert all(
            perImage.xs("dmLineResiduals", level="task").map(tuple)
            == ("dmLineResidualThresholds-run25.yaml", "dmLineResiduals.config")
        )
        assert layers["layer"].tolist() == [0, 1] * 8
        # qa_gate itself cites nothing: the layers are the one record.
        assert not {"iq_thresholds_id", "dm_thresholds_id", "twilight_thresholds_id"} & set(
            qadb.query_dataframe("SELECT * FROM qa_gate").columns
        )
        # The config layer's entries are the task's config table: the drift rows a run25 file lacks.
        dm = int(names.loc[names["name"] == "dmLineResiduals.config", "thresholds_id"].iloc[0])
        entries = qadb.query_dataframe(
            "SELECT metric, reference, description, warn, fail FROM qa_threshold_entry WHERE thresholds_id = :i",
            params={"i": dm},
        )
        assert set(entries["metric"]) == {
            "medianDeltaX",
            "medianDeltaY",
            "robustRmsDeltaX",
            "robustRmsDeltaY",
        }
        drift = entries.set_index("metric").loc["medianDeltaX"]
        assert (drift["reference"], drift["description"], drift["warn"], drift["fail"]) == (
            "calib",
            "Trace",
            1.0,
            2.0,
        )
        assert (
            entries.set_index("metric").loc["robustRmsDeltaX", ["warn", "fail"]].isna().all()
        )  # NaN: not judged

    def testStoredConfigThatBuildsOtherThresholdsIsNotCited(self, StubButler, tables, fullConfigs, qadb):
        """A config layer is cited only when the RUN's stored config rebuilds the recorded content."""
        other = SimpleNamespace(**vars(fullConfigs["imageQualityQa_config"]))
        other.fwhmWarnThreshold = 9.0  # the stored config says otherwise than what the task recorded
        datasets = {"packages": {"drp_qa": "x"}, **fullConfigs, "imageQualityQa_config": other}
        messages = self.logged(StubButler(recorded(tables, configs=fullConfigs), datasets=datasets), qadb)
        layers = self.layers(qadb)
        assert set(layers.loc[layers["task"] == "imageQualityQa", "name"]) == {"iqQaThresholds-run25.yaml"}
        assert set(layers.loc[layers["task"] == "dmLineResiduals", "name"]) == {
            "dmLineResidualThresholds-run25.yaml",
            "dmLineResiduals.config",
        }
        assert "imageQualityQa.config" not in set(qadb.query_scalars("SELECT name FROM qa_thresholds"))
        mismatch = [m for m in messages if "imageQualityQa.config" in m]
        assert len(mismatch) == 1 and "what this drp_qa builds from the RUN's stored config" in mismatch[0]
        # Negative control: the same stored config as recorded is cited.
        again = SqliteQaDBOf(qadb)
        load(StubButler(recorded(tables, configs=fullConfigs), datasets={**datasets, **fullConfigs}), again)
        assert "imageQualityQa.config" in set(again.query_scalars("SELECT name FROM qa_thresholds"))

    def testStoredConfigWithoutTheFieldsIsNotCited(self, StubButler, tables, fullConfigs, configs, qadb):
        """A stored config read without threshold fields (an old config class) gives no config layer."""
        datasets = {"packages": {"drp_qa": "x"}, **configs}  # thresholdsFile alone
        messages = self.logged(StubButler(recorded(tables, configs=fullConfigs), datasets=datasets), qadb)
        assert set(self.layers(qadb)["name"]) == {
            "iqQaThresholds-run25.yaml",
            "dmLineResidualThresholds-run25.yaml",
        }
        assert sum("given by the RUN's stored config" in m for m in messages) == 2  # one per task
        assert configDocumentOf(StubButler({}, datasets=datasets), RUN, "imageQualityQa") is None
        assert (
            configDocumentOf(StubButler({}, datasets=datasets), RUN, "twilightQa") is None
        )  # no config layer

    def testRecordedLayerAlreadyInQadbNeedsNoConfig(self, StubButler, tables, fullConfigs, configs, qadb):
        """Once registered, a config layer is cited by its hash alone, whatever the RUN's config now reads as."""
        full = {"packages": {"drp_qa": "x"}, **fullConfigs}
        load(StubButler(recorded(tables, configs=fullConfigs), datasets=full), qadb)
        bare = {"packages": {"drp_qa": "x"}, **configs}
        messages = self.logged(StubButler(recorded(tables, configs=fullConfigs), datasets=bare), qadb)
        assert not any("stored config" in m for m in messages)
        assert set(self.layers(qadb)["name"]) >= {"imageQualityQa.config", "dmLineResiduals.config"}

    def testFallbackRegistersTheConfigLayerToo(self, StubButler, tables, fullConfigs, qadb):
        """A table recording nothing gets its RUN's config: the files it names, then its fields."""
        datasets = {"packages": {"drp_qa": "x"}, **fullConfigs}
        messages = self.logged(StubButler(tables, datasets=datasets), qadb)
        layers = self.layers(qadb)
        perImage = layers.groupby(["pfs_visit_id", "camera_name", "task"])["name"].agg(tuple)
        assert set(perImage.xs("imageQualityQa", level="task")) == {
            ("iqQaThresholds-run25.yaml", "imageQualityQa.config")
        }
        assert set(perImage.xs("dmLineResiduals", level="task")) == {
            ("dmLineResidualThresholds-run25.yaml", "dmLineResiduals.config")
        }
        assert sum("recorded no thresholds layers" in m for m in messages) == 2

    def testReloadReplacesTheLayers(self, StubButler, tables, fullConfigs, qadb):
        """A new judgement's layers replace the old ones: none linger from a judgement with more."""
        datasets = {"packages": {"drp_qa": "x"}, **fullConfigs}
        load(StubButler(recorded(tables, configs=fullConfigs), datasets=datasets), qadb)
        assert len(self.layers(qadb)) == 16
        load(StubButler(recorded(tables), datasets=datasets), qadb)  # judged with the files alone
        layers = self.layers(qadb)
        assert len(layers) == 8 and set(layers["name"]) == {
            "iqQaThresholds-run25.yaml",
            "dmLineResidualThresholds-run25.yaml",
        }
        assert qadb.count("qa_thresholds") == 4  # the config layers outlive the judgements that cited them

    def testMissingLayerKeepsTheOthersPositions(self, StubButler, tables, fullConfigs, qadb):
        """A layer that cannot be cited leaves a gap: the config layer stays at position 1."""
        iq = recorded(tables, configs=fullConfigs)["iqQaMetrics"]
        files, digests = iq["thresholdsFiles"].iloc[0], iq["thresholdsSha256"].iloc[0]
        unknown = "f" * 64 + digests[digests.index(",") :]  # a file this drp_qa never had
        iq = iq.assign(
            thresholdsFiles=files.replace("iqQaThresholds-run25.yaml", "gone.yaml"), thresholdsSha256=unknown
        )
        datasets = {"packages": {"drp_qa": "x"}, **fullConfigs}
        load(
            StubButler({**recorded(tables, configs=fullConfigs), "iqQaMetrics": iq}, datasets=datasets), qadb
        )
        layers = self.layers(qadb)
        iqLayers = layers[layers["task"] == "imageQualityQa"]
        assert set(iqLayers["name"]) == {"imageQualityQa.config"} and set(iqLayers["layer"]) == {1}
        assert distinct(qadb, "iq_thresholds_id") == [None]  # the first layer is unknown

    def testDocumentHashesAsTheTaskRecorded(self, fullConfigs):
        """The loader's rebuild is byte-for-byte what `taskThresholdsRecord` hashed."""
        config = fullConfigs["imageQualityQa_config"]
        record = taskThresholdsRecord(
            "imageQualityQa", ["iqQaThresholds-run25.yaml"], CONFIG_LAYERS["imageQualityQa"](config)
        )
        document = thresholdsDocument(
            CONFIG_LAYERS["imageQualityQa"](config), configLayerName("imageQualityQa")
        )
        assert record["thresholdsSha256"].split(",")[1] == sha256Of(document)


def SqliteQaDBOf(qadb):
    """Return a fresh database of the same kind as ``qadb`` (SQLite only: PostgreSQL tests reuse theirs)."""
    import pytest

    if type(qadb).__name__ != "SqliteQaDB":
        pytest.skip("a second database is only made for SQLite")
    return type(qadb)()
