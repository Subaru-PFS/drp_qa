"""Tests for loading a collection from a (stub) Butler into qadb."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pfs.drp.qa.qadb.loader import (
    MODES,
    SPECS,
    drpQaVersionOf,
    loadCollection,
    loadVisit,
    readTables,
    thresholdsFilesOf,
)
from pfs.drp.qa.qadb.tables import GATE_ORDER, worstStatus
from pfs.drp.qa.qadb.writer import UNKNOWN_VERSION

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
        assert gate["iq_thresholds_id"].notna().all() and gate["dm_thresholds_id"].notna().all()
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
        gate = qadb.query_dataframe(
            "SELECT pfs_visit_id, iq_thresholds_id, dm_thresholds_id FROM qa_gate ORDER BY pfs_visit_id, camera_name"
        )
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
        assert qadb.query_scalars("SELECT DISTINCT iq_thresholds_id FROM qa_gate") == [first]
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
        assert qadb.query_scalars("SELECT dm_thresholds_id FROM qa_gate") == [None] * 4
        assert qadb.query_scalars("SELECT DISTINCT iq_thresholds_id FROM qa_gate") != [None]

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
        assert qadb.query_scalars("SELECT dm_thresholds_id FROM qa_gate") == [None] * 4

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
        }
        gate = pd.read_parquet(tmp_path / "qa_gate.parquet")
        assert (gate["processing_id"] < 0).all()  # placeholders: no database was asked
        assert gate["iq_thresholds_id"].isna().all()

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
