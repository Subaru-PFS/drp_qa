"""Tests for loading a collection from a (stub) Butler into qadb, and the verdict the load gives each image."""

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import STATUS_ORDER, gate
from pfs.drp.qa.metrics.registry import CHECKS, RELATIVE
from pfs.drp.qa.qadb import loader
from pfs.drp.qa.qadb.loader import (
    MODES,
    SPECS,
    collectionExists,
    drpQaVersionOf,
    imageRuns,
    loadCollection,
    loadVisit,
    readTables,
)
from pfs.drp.qa.qadb.reader import PROCESSING_ID, readGate, readNormalValues
from pfs.drp.qa.qadb.writer import UNKNOWN_VERSION

COLLECTION = "u/test/qa"
RUN = "u/test/qa/20261026T000000Z"
NOW = datetime(2026, 10, 26, 12, 0, 0)
LATER = datetime(2026, 10, 26, 12, 5, 0)
VISITS = [133025, 133040]
IMAGE = ["visit", "arm", "spectrograph"]

#: The numbers tables: none of them may hold a verdict.
NUMBERS = (
    "image_quality_qa",
    "image_quality_qa_species",
    "dm_line_residual_qa",
    "dm_line_residual_qa_metric",
    "twilight_qa",
)

#: Normal widths of the fixture's images (measured 2.76, 3.18 and 3.33 px): r1's arc is +10 % (FAIL), b1's
#: quartz +2.4 % (PASS) and r1's quartz +3.9 % (WARN).
WIDTHS = {
    ("medFwhm", "r1", "arc", "Ar"): 2.5,
    ("medFwhm", "b1", "quartz", ""): 3.1,
    ("medFwhm", "r1", "quartz", ""): 3.2,
}


def species(iq: pd.DataFrame) -> pd.DataFrame:
    """Return a long ``iqQaSpeciesMetrics`` table for the images of ``iq``: two species, two metrics each."""
    rows = []
    for image in iq[IMAGE].itertuples(index=False):
        for description in ("ArI", "Trace"):
            for metric, value in (("fitXRms", 0.05), ("fitYRms", 0.07)):
                rows.append({**image._asdict(), "description": description, "metric": metric, "value": value})
    return pd.DataFrame(rows)


@pytest.fixture
def tables(fixtures):
    return {**fixtures, "iqQaSpeciesMetrics": species(fixtures["iqQaMetrics"])}


@pytest.fixture
def butler(StubButler, tables):
    return StubButler(tables, datasets={"packages": {"drp_qa": "w.2026.44", "drp_stella": "w.2026.41"}})


def load(butler, qadb, **kwargs):
    kwargs.setdefault("pipelineVersions", False)
    kwargs.setdefault("now", lambda: NOW)
    kwargs.setdefault("visits", VISITS)
    kwargs.setdefault("log", lambda _: None)
    return loadCollection(butler, qadb, COLLECTION, **kwargs)


def gateOf(qadb) -> pd.DataFrame:
    """Return ``qa_gate``, by image."""
    return qadb.query_dataframe("SELECT * FROM qa_gate ORDER BY pfs_visit_id, camera_name")


def verdictOf(qadb) -> dict:
    """Return each image's verdict, by ``(visit, camera)``."""
    rows = gateOf(qadb)
    return dict(
        zip(zip(rows["pfs_visit_id"], rows["camera_name"], strict=True), rows["qa_status"], strict=True)
    )


def expected(tables, normalValues=None) -> pd.DataFrame:
    """Return what the gate says of the Butler tables: the load must store exactly this."""
    verdicts = gate(
        iq=tables.get("iqQaMetrics"),
        lines=tables.get("dmLineResidualMetrics"),
        normalValues=normalValues,
    )
    return verdicts.assign(camera=verdicts["arm"] + verdicts["spectrograph"].astype(str))


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
        assert list(read["dmLineResidualSummary"].columns) == [*IMAGE, "run"]

    def testNoVerdictDatasetIsRead(self, butler):
        """The tasks' numbers are all the loader reads: no dataset of verdicts, no task config."""
        readTables(butler, COLLECTION, visits=VISITS, log=lambda _: None)
        asked = {datasetType for datasetType, _ in butler.queries}
        assert asked == set(SPECS)
        assert "twilightQaMetrics" not in asked and not any(name.endswith("_config") for name in asked)

    def testALoadReadsNoVerdictDatasetEither(self, butler, tmp_path):
        """A whole load of lamp frames asks for the numbers tables alone: nothing judged, no twilight flux."""
        load(butler, None, dryRun=tmp_path)
        asked = {datasetType for datasetType, _ in butler.queries}
        assert asked == set(SPECS)
        # Negative control: the stub does record what it is asked, by name.
        assert "iqQaMetrics" in asked and len(butler.queries) == len(SPECS)


def testHelpers(butler, StubButler):
    assert drpQaVersionOf(butler, RUN) == "w.2026.44"
    assert drpQaVersionOf(StubButler({}), RUN) == UNKNOWN_VERSION  # never the loader's own version


def testNoThresholdsMachinery():
    """Nothing of the thresholds files is left to call."""
    gone = {"CONFIG_LAYERS", "configDocumentOf", "thresholdsFilesOf", "registerThresholds"}
    assert not gone & set(dir(loader))
    # Negative control: the names the loader does export resolve.
    assert all(hasattr(loader, name) for name in loader.__all__)


def testImageRuns():
    """An image takes the RUN of the first table, in the order given, with a row for it."""
    first = pd.DataFrame({"visit": [1], "arm": ["b"], "spectrograph": [1], "run": ["iq"]})
    second = pd.DataFrame({"visit": [1, 2], "arm": ["b", "b"], "spectrograph": [1, 1], "run": ["dm", "dm"]})
    runs = imageRuns(first, second).set_index("visit")["run"]
    assert runs.to_dict() == {1: "iq", 2: "dm"}
    # Negative control: the order given decides, not the tables' content.
    assert imageRuns(second, first).set_index("visit")["run"].to_dict() == {1: "dm", 2: "dm"}
    assert imageRuns(first.iloc[:0], second.iloc[:0]).empty


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
    def testCountsAndContent(self, butler, qadb):
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
        assert qadb.count("twilight_qa") == 0  # no twilight among them

    def testLoadingTwiceChangesNothing(self, butler, qadb, tables, scope):
        first = load(butler, qadb)
        before = {table: qadb.count(table, scope.get(table, "")) for table in first if table != "processings"}
        gateBefore = gateOf(qadb)
        again = load(butler, qadb)
        assert again == first
        assert {table: qadb.count(table, scope.get(table, "")) for table in before} == before
        assert qadb.count("data_processing_pipe2d", scope["data_processing_pipe2d"]) == 1
        pd.testing.assert_frame_equal(gateOf(qadb), gateBefore)
        # Negative control: an image measured again is replaced, not duplicated, and judged on the new number.
        tables["iqQaMetrics"].loc[0, "medFwhm"] = 9.0
        metrics = tables["dmLineResidualMetrics"]
        offset = (
            (metrics["visit"] == 133025)
            & (metrics["arm"] == "b")
            & (metrics["description"] == "Trace")
            & (metrics["reference"] == "calib")
            & (metrics["metric"] == "medianDeltaX")
        )
        assert offset.sum() == 1
        metrics.loc[offset, "value"] = 2.5
        load(butler, qadb, now=lambda: LATER)
        assert qadb.count("image_quality_qa") == 4 and qadb.count("qa_gate") == 4
        width = qadb.query_scalar(
            "SELECT med_fwhm FROM image_quality_qa WHERE pfs_visit_id = :v AND camera_name = :c",
            params={"v": 133025, "c": "b1"},
        )
        assert width == 9.0
        row = gateOf(qadb).iloc[0]
        assert (row["pfs_visit_id"], row["camera_name"]) == (133025, "b1")
        assert (row["qa_status"], row["qa_decided_by"]) == ("FAIL", "calibOffset")
        assert gateBefore.loc[0, "qa_status"] == "PASS"
        assert pd.Timestamp(row["decided_at"]) == pd.Timestamp(LATER)

    def testReloadDropsRowsTheNewMeasurementLacks(self, butler, qadb, tables):
        """A species or metric missing from an image measured again is removed, not left with its old value."""
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
        """An image whose new measurement has no species rows at all loses its old ones."""
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

    def testVersionsSurviveARealtimeReload(self, StubButler, tables, qadb):
        """QaActor's loadVisit reads no pipeline versions; the backfill's stay."""
        tables = dict(tables)
        iq = tables["iqQaMetrics"].assign(calexpRun="u/test/reduce")
        tables["iqQaMetrics"] = iq
        tables["pfsArm"] = iq[IMAGE]
        butler = StubButler(tables, datasets={"packages": {"drp_qa": "w.2026.44"}})
        butler.getURI = lambda ref: ref
        header = {
            "VERSION_DATAMODEL": "w.2026.40",
            "VERSION_OBS_PFS": "w.2026.40",
            "VERSION_DRP_STELLA": "w.2026.41",
        }
        load(butler, qadb, pipelineVersions=True, readHeader=lambda ref: header)
        bare = StubButler(tables)  # no packages dataset this time either
        loadVisit(bare, qadb, COLLECTION, 133040, pipelineVersions=False, log=lambda _: None, now=lambda: NOW)
        row = qadb.query_dataframe(
            "SELECT * FROM data_processing_pipe2d WHERE collection_name = :c", params={"c": COLLECTION}
        ).iloc[0]
        assert row["pipe2d_version"] == "w.2026.41"
        assert row["drp_qa_version"] == "w.2026.44"

    def testNeedsADatabaseOrADryRun(self, butler):
        with pytest.raises(ValueError, match="dryRun"):
            load(butler, None)

    def testDeletingAProcessingCascades(self, butler, qadb, storeValues, scope):
        """Removing a processing row takes every QA row of it along (the schema's ON DELETE CASCADE)."""
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=1)
        load(butler, qadb)
        qadb.commit("DELETE FROM data_processing_pipe2d WHERE collection_name = :c", params={"c": COLLECTION})
        for table in (*NUMBERS, "qa_gate"):
            assert qadb.count(table) == 0, table
        # The normal values are the collection's, not a processing's: they outlive it.
        assert qadb.count("qa_normal_value", scope["qa_normal_value"]) == len(WIDTHS)

    def testOneProcessingPerRun(self, StubButler, tables, qadb):
        runs = {
            (133025, "b", 1): "r1",
            (133025, "r", 1): "r1",
            (133040, "b", 1): "r2",
            (133040, "r", 1): "r2",
        }
        butler = StubButler(tables, runs=runs, datasets={"packages": {"drp_qa": "x"}})
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
        # And so does each image's gate row.
        gateRuns = gateOf(qadb).groupby("pfs_visit_id")["processing_id"].agg(set)
        assert gateRuns[133025] == {ids.loc[0, "processing_id"]}
        assert gateRuns[133040] == {ids.loc[1, "processing_id"]}

    def testLongRowsWithoutTheirImageAreSkipped(self, StubButler, tables, qadb):
        tables = dict(tables)
        tables["iqQaMetrics"] = tables["iqQaMetrics"].iloc[:3]  # one image's species rows lose their image
        messages = []
        counts = load(StubButler(tables), qadb, log=messages.append)
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

    def testDryRun(self, butler, tmp_path, tables):
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
        stored = pd.read_parquet(tmp_path / "qa_gate.parquet")
        assert (stored["processing_id"] < 0).all()  # placeholders: no database was asked
        # Without a database there are no normal values, unless they are given.
        assert stored["qa_status"].tolist() == expected(tables)["qaStatus"].tolist() == ["PASS"] * 4

    def testDryRunJudgesWithTheNormalValuesGiven(self, butler, tmp_path, tables, SqliteQaDB, storeValues):
        other = SqliteQaDB()
        storeValues(other, COLLECTION, WIDTHS, validFrom=1)
        values = readNormalValues(other, COLLECTION)
        load(butler, None, dryRun=tmp_path, normalValues=values)
        stored = pd.read_parquet(tmp_path / "qa_gate.parquet")
        assert stored["qa_status"].tolist() == ["PASS", "FAIL", "PASS", "WARN"]
        assert stored["qa_status"].tolist() == expected(tables, values)["qaStatus"].tolist()

    def testPipelineVersions(self, StubButler, tables, qadb):
        """The reduction's versions come from the pfsArm beside each judged calexp."""
        tables = dict(tables)
        iq = tables["iqQaMetrics"].assign(calexpRun="u/test/reduce")
        tables["iqQaMetrics"] = iq
        tables["pfsArm"] = iq[IMAGE]
        butler = StubButler(tables, datasets={"packages": {"drp_qa": "x"}})
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
        load(butler, qadb, pipelineVersions=True, log=messages.append)
        row = qadb.query_dataframe(
            "SELECT * FROM data_processing_pipe2d WHERE collection_name = :c", params={"c": COLLECTION}
        ).iloc[0]
        assert pd.isna(row["pipe2d_version"])
        assert any("no single reduction pipeline" in m for m in messages)

    def testVisitDescriptionsComeFromTheOpdb(self, butler, qadb):
        """With an opdb, each visit's description, design and time are copied; an arc asks it nothing else."""

        class OpDB:
            asked: list

            def __init__(self):
                self.asked = []

            def query_dataframe(self, sql, /, *, params=None, conn=None):
                self.asked.append(sql)
                assert "FROM pfs_visit" in sql
                return pd.DataFrame(
                    {
                        "pfs_visit_id": [133025],
                        "pfs_visit_description": ["Arc: Argon"],
                        "pfs_design_id": [0x1234],
                        "issued_at": pd.to_datetime(["2025-11-12 20:00"]),
                    }
                )

        opdb = OpDB()
        counts = load(butler, qadb, opdb=opdb)
        assert counts["pfs_visit"] == 2 and len(opdb.asked) == 1
        visits = qadb.query_dataframe(
            "SELECT * FROM pfs_visit WHERE pfs_visit_id IN (133025, 133040) ORDER BY pfs_visit_id"
        )
        assert visits["pfs_visit_description"].tolist()[0] == "Arc: Argon"
        assert pd.isna(visits.loc[1, "pfs_visit_description"])  # unknown to the opdb: its id alone


class TestGate:
    """The load gives each image one verdict: the gate's, in ``qa_gate`` and nowhere else."""

    def testOneRowPerImageWithTheGatesVerdict(self, butler, qadb, tables):
        load(butler, qadb, mode="comparison")
        rows = gateOf(qadb)
        want = expected(tables)
        assert len(rows) == 4 and not rows.duplicated(["pfs_visit_id", "camera_name"]).any()
        assert rows["pfs_visit_id"].tolist() == want["visit"].tolist()
        assert rows["camera_name"].tolist() == want["camera"].tolist() == ["b1", "r1", "b1", "r1"]
        assert rows["frame_type"].tolist() == want["frameType"].tolist() == ["arc", "arc", "quartz", "quartz"]
        assert rows["qa_status"].tolist() == want["qaStatus"].tolist()
        assert rows["qa_decided_by"].tolist() == want["qaDecidedBy"].tolist()
        assert rows["qa_reason"].tolist() == want["qaReason"].tolist()
        assert set(rows["qa_status"]) <= set(STATUS_ORDER)
        assert (rows["mode"] == "comparison").all()
        assert (pd.to_datetime(rows["decided_at"]) == pd.Timestamp(NOW)).all()
        assert not {"decided_by_task", "provisional"} & set(rows.columns)

    def testNoNumbersTableGetsAVerdict(self, StubButler, tables, qadb):
        """Even from Butler tables that still carry the tasks' old verdict columns."""
        old = dict(tables)
        old["iqQaMetrics"] = tables["iqQaMetrics"].assign(qaStatus="FAIL", qaReason="then")
        old["dmLineResidualSummary"] = tables["dmLineResidualSummary"].assign(
            qaStatus="FAIL", recommendedAction="RECALIBRATE"
        )
        old["dmLineResidualMetrics"] = tables["dmLineResidualMetrics"].assign(status="FAIL")
        load(StubButler(old), qadb)
        verdictColumns = {"qa_status", "qa_decided_by", "qa_reason", "recommended_action", "status"}
        for table in NUMBERS:
            columns = set(qadb.query_dataframe(f"SELECT * FROM {table}").columns)
            assert not verdictColumns & columns, table
        # The old verdicts decide nothing: the gate judges the numbers.
        assert set(gateOf(qadb)["qa_status"]) == {"PASS"}
        # Negative control: the gate table is where the verdict columns are.
        assert verdictColumns & set(gateOf(qadb).columns) == {"qa_status", "qa_decided_by", "qa_reason"}

    def testAFixedCheckThatFailsIsNamedWithItsReason(self, butler, qadb, tables):
        metrics = tables["dmLineResidualMetrics"]
        offset = (
            (metrics["visit"] == 133040)
            & (metrics["arm"] == "r")
            & (metrics["description"] == "Trace")
            & (metrics["reference"] == "calib")
            & (metrics["statusType"] == "BOTH")
            & (metrics["metric"] == "medianDeltaX")
        )
        assert offset.sum() == 1
        metrics.loc[offset, "value"] = -1.5  # beyond WARN (1 px), short of FAIL (2 px), in either direction
        load(butler, qadb)
        row = gateOf(qadb).set_index(["pfs_visit_id", "camera_name"]).loc[(133040, "r1")]
        assert (row["qa_status"], row["qa_decided_by"]) == ("WARN", "calibOffset")
        assert row["qa_reason"] == "x offset -1.50 px, limit 1 px"
        # Negative control: the other images, whose offsets are a fraction of a pixel, pass.
        assert [status for key, status in verdictOf(qadb).items() if key != (133040, "r1")] == ["PASS"] * 3

    def testNormalValuesInQadbJudgeTheRelativeChecks(self, butler, qadb, tables, storeValues):
        """A relative check needs the camera's normal value; the load takes the collection's from qadb."""
        load(butler, qadb)
        without = gateOf(qadb)
        # Without normal values only the fixed checks judge: nothing here is decided by a relative one.
        assert without["qa_status"].tolist() == ["PASS"] * 4 and set(without["qa_decided_by"]) == {""}

        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
        load(butler, qadb, now=lambda: LATER)
        rows = gateOf(qadb)
        assert rows["qa_status"].tolist() == ["PASS", "FAIL", "PASS", "WARN"]
        assert rows["qa_decided_by"].tolist() == ["", "width", "", "width"]
        assert CHECKS["width"].kind == RELATIVE
        assert rows.loc[1, "qa_reason"] == "width 2.76 px is +10.3 % from normal 2.50 px, limit +5 %"
        assert rows.loc[3, "qa_reason"] == "width 3.33 px is +3.9 % from normal 3.20 px, limit +3 %"
        want = expected(tables, readNormalValues(qadb, COLLECTION))
        assert rows["qa_status"].tolist() == want["qaStatus"].tolist()
        assert rows["qa_reason"].tolist() == want["qaReason"].tolist()

        # Negative control: wider is worse, narrower is not. The same widths against normal values as far
        # above them pass.
        narrower = {key: value * 1.2 for key, value in WIDTHS.items()}
        storeValues(qadb, COLLECTION, narrower, validFrom=133000)
        load(butler, qadb)
        assert gateOf(qadb)["qa_status"].tolist() == ["PASS"] * 4

    def testWithoutNormalValuesTheRelativeChecksAreUnknown(self, StubButler, fixtures, qadb, storeValues):
        """Image quality alone has only relative checks on a lamp frame: no normal value, no verdict."""
        butler = StubButler({"iqQaMetrics": fixtures["iqQaMetrics"]})
        counts = load(butler, qadb)
        assert "dm_line_residual_qa" not in counts and counts["qa_gate"] == 4
        rows = gateOf(qadb)
        assert rows["qa_status"].tolist() == ["UNKNOWN"] * 4
        assert set(rows["qa_reason"]) == {"no normal value"} and set(rows["qa_decided_by"]) == {""}
        assert rows["frame_type"].tolist() == ["arc", "arc", "quartz", "quartz"]
        # Negative control: with the normal values in qadb the same load judges them.
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
        load(butler, qadb)
        assert gateOf(qadb)["qa_status"].tolist() == ["UNKNOWN", "FAIL", "PASS", "WARN"]
        # 133025 b1 stays unjudged: an arc under the floor of lines per lit fiber has no width to judge.
        assert fixtures["iqQaMetrics"].loc[0, "linesPerFiber"] < 5

    def testANormalValueJudgesFromItsFirstVisitOn(self, butler, qadb, storeValues):
        """``valid_from`` is the first visit a value judges: an earlier frame is not judged by it."""
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133030)
        load(butler, qadb)
        rows = gateOf(qadb)
        # 133040 is judged by them; 133025, before them, only by its fixed checks.
        assert rows["qa_status"].tolist() == ["PASS", "PASS", "PASS", "WARN"]
        # Negative control: valid from the frame itself, the value judges it (at or before, not strictly before).
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133025)
        load(butler, qadb)
        assert gateOf(qadb)["qa_status"].tolist() == ["PASS", "FAIL", "PASS", "WARN"]

    def testTheLatestValueInForceJudges(self, butler, qadb, storeValues):
        """Of two accepted values, the later one at or before the frame is the one in force."""
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
        wide = {("medFwhm", "r1", "quartz", ""): 3.3}  # 3.33 px is +0.8 % from it
        storeValues(qadb, COLLECTION, wide, validFrom=133035)
        load(butler, qadb)
        assert verdictOf(qadb)[(133040, "r1")] == "PASS"
        # Negative control: the earlier value alone warned on that image.
        qadb.commit("DELETE FROM qa_normal_value WHERE valid_from = 133035")
        load(butler, qadb)
        assert verdictOf(qadb)[(133040, "r1")] == "WARN"

    def testAnotherCollectionsNormalValuesJudgeNothing(self, butler, qadb, storeValues):
        storeValues(qadb, "u/test/other", WIDTHS, validFrom=133000)
        load(butler, qadb)
        assert gateOf(qadb)["qa_status"].tolist() == ["PASS"] * 4
        # Negative control: the same values under this collection do judge.
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
        load(butler, qadb)
        assert gateOf(qadb)["qa_status"].tolist() == ["PASS", "FAIL", "PASS", "WARN"]

    def testNormalValuesGivenAreUsedInsteadOfQadbs(self, butler, qadb, storeValues, SqliteQaDB):
        other = SqliteQaDB()
        storeValues(other, COLLECTION, WIDTHS, validFrom=133000)
        load(butler, qadb, normalValues=readNormalValues(other, COLLECTION))
        assert gateOf(qadb)["qa_status"].tolist() == ["PASS", "FAIL", "PASS", "WARN"]
        assert qadb.count("qa_normal_value", "collection_name = 'u/test/qa'") == 0  # judged with, not stored

    def testLineResidualsAloneLeaveALampFrameUnknown(self, StubButler, tables, qadb):
        """A lamp frame's type needs its illumination, which only iqQaMetrics records: no row, no verdict."""
        dm = {name: table for name, table in tables.items() if name.startswith("dmLineResidual")}
        load(StubButler(dm), qadb)
        rows = gateOf(qadb)
        assert len(rows) == 4 and set(rows["qa_status"]) == {"UNKNOWN"}
        assert set(rows["qa_reason"]) == {"nothing to judge was measured"}
        # Negative control: the same residuals beside their iqQaMetrics rows are judged.
        load(StubButler(tables), qadb)
        assert set(gateOf(qadb)["qa_status"]) == {"PASS"}

    def testTheGateRowTakesTheFirstTablesProcessing(self, StubButler, tables, qadb):
        """ImageQualityQa and dmLineResiduals in two RUNs: one gate row per image, under imageQualityQa's."""
        runs = {
            (name, *image): "u/test/qa/dm"
            for name in ("dmLineResidualSummary", "dmLineResidualMetrics")
            for image in tables["iqQaMetrics"][IMAGE].itertuples(index=False, name=None)
        }
        counts = load(StubButler(tables, runs=runs), qadb)
        assert counts["processings"] == 2 and counts["qa_gate"] == 4
        rows = qadb.query_dataframe(
            "SELECT g.pfs_visit_id, g.camera_name, p.run_name FROM qa_gate g"
            " JOIN data_processing_pipe2d p ON p.processing_id = g.processing_id"
        )
        assert len(rows) == 4 and set(rows["run_name"]) == {RUN}
        # Negative control: the line residuals themselves sit under their own RUN's processing.
        dmRuns = qadb.query_scalars(
            "SELECT DISTINCT p.run_name FROM dm_line_residual_qa t"
            " JOIN data_processing_pipe2d p ON p.processing_id = t.processing_id"
        )
        assert dmRuns == ["u/test/qa/dm"]

    def testAVisitMeasuredAgainInANewerRunIsGatedUnderIt(self, StubButler, tables, qadb):
        """Numbers measured again into a new RUN: the gate row goes under the new processing, not the old row's."""
        load(StubButler(tables), qadb)
        before = readGate(qadb, COLLECTION)
        assert before["qaStatus"].tolist() == ["PASS"] * 4
        # The newer RUN's measurement of 133040 finds its traces 2.5 px from the calibration.
        metrics = tables["dmLineResidualMetrics"].copy()
        offset = (
            (metrics["visit"] == 133040)
            & (metrics["description"] == "Trace")
            & (metrics["reference"] == "calib")
            & (metrics["metric"] == "medianDeltaX")
        )
        metrics.loc[offset, "value"] = 2.5
        newer = "u/test/qa/20261027T000000Z"
        runs = {(133040, "b", 1): newer, (133040, "r", 1): newer}
        again = StubButler({**tables, "dmLineResidualMetrics": metrics}, runs=runs)
        loadVisit(
            again, qadb, COLLECTION, 133040, pipelineVersions=False, log=lambda _: None, now=lambda: LATER
        )

        ids = qadb.query_dataframe(
            "SELECT processing_id, run_name FROM data_processing_pipe2d WHERE collection_name = :c",
            params={"c": COLLECTION},
        ).set_index("run_name")["processing_id"]
        stored = gateOf(qadb)
        # The first RUN's rows stay; the visit measured again has a second row, under the newer processing.
        assert len(stored) == 6
        assert stored.groupby("pfs_visit_id")["processing_id"].agg(set).to_dict() == {
            133025: {ids[RUN]},
            133040: {ids[RUN], ids[newer]},
        }
        new = stored[stored["processing_id"] == ids[newer]]
        assert new["qa_status"].tolist() == ["FAIL", "FAIL"] and set(new["qa_decided_by"]) == {"calibOffset"}
        # Negative control: the row under the first RUN's processing was not overwritten by the new verdict.
        old = stored[(stored["processing_id"] == ids[RUN]) & (stored["pfs_visit_id"] == 133040)]
        assert old["qa_status"].tolist() == ["PASS", "PASS"]
        assert (pd.to_datetime(old["decided_at"]) == pd.Timestamp(NOW)).all()
        # Readers take the newer RUN's row.
        after = readGate(qadb, COLLECTION).set_index("visit")
        assert after.loc[133040, "qaStatus"].tolist() == ["FAIL", "FAIL"]
        assert set(after.loc[133040, PROCESSING_ID]) == {ids[newer]} and set(after.loc[133040, "run"]) == {
            newer
        }
        assert set(after.loc[133025, PROCESSING_ID]) == {ids[RUN]}

    def testLineResidualsWithoutAnyImageRowAreNotGated(self, StubButler, tables, qadb):
        """Residuals with neither a summary nor an iqQaMetrics row have no processing to take: skipped, and said."""
        orphaned = dict(tables)
        orphaned["dmLineResidualSummary"] = tables["dmLineResidualSummary"].iloc[:3]
        orphaned["iqQaMetrics"] = tables["iqQaMetrics"].iloc[:3]
        del orphaned["iqQaSpeciesMetrics"]
        messages = []
        counts = load(StubButler(orphaned), qadb, log=messages.append)
        assert counts["qa_gate"] == 3 and (133040, "r1") not in verdictOf(qadb)
        assert any("qa_gate: 1 images with line residuals and no summary row, skipped" in m for m in messages)
        assert any(
            "dmLineResidualMetrics: 24 rows without a dmLineResidualSummary row" in m for m in messages
        )

    def testTheTallyIsLogged(self, butler, qadb, storeValues):
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
        messages = []
        load(butler, qadb, log=messages.append)
        assert "verdicts: 2 PASS, 1 WARN, 1 FAIL" in messages


def reductionRows(iq: pd.DataFrame) -> pd.DataFrame:
    """Return a long ``reductionStats`` table for the images of ``iq``.

    Two whole-detector metrics (empty amplifier) and two per-amplifier metrics on ``c1``.
    """
    rows = []
    for image in iq[IMAGE].itertuples(index=False):
        base = image._asdict()
        rows.append({**base, "amplifier": "", "metric": "fitXRms", "value": 0.02})
        rows.append({**base, "amplifier": "", "metric": "nDefects", "value": 201178.0})
        rows.append({**base, "amplifier": "c1", "metric": "readNoise", "value": 3.9})
        rows.append({**base, "amplifier": "c1", "metric": "gain", "value": 1.2})
    return pd.DataFrame(rows)


PACKAGES = {"packages": {"drp_qa": "w.2026.44", "drp_stella": "w.2026.41"}}


class TestDetectorHealth:
    """``reductionStats`` loads into ``detector_health_qa``: a standalone long table, not judged."""

    def butlerWithHealth(self, StubButler, tables):
        """Return a stub Butler whose tables also carry a ``reductionStats`` dataset."""
        tables["reductionStats"] = reductionRows(tables["iqQaMetrics"])
        return StubButler(tables, datasets=PACKAGES)

    def testLoadsRows(self, StubButler, tables, qadb):
        butler = self.butlerWithHealth(StubButler, tables)
        counts = load(butler, qadb)
        assert counts["detector_health_qa"] == 16  # 4 images x 4 metrics
        health = qadb.query_dataframe(
            "SELECT * FROM detector_health_qa ORDER BY pfs_visit_id, camera_name, amplifier, metric"
        )
        assert len(health) == 16
        # A whole-detector statistic has an empty amplifier; a per-amp one names the amplifier.
        assert (health.loc[health["metric"] == "fitXRms", "amplifier"] == "").all()
        assert (health.loc[health["metric"] == "readNoise", "amplifier"] == "c1").all()
        # The value is stored as double precision, keyed to the image.
        value = qadb.query_scalar(
            "SELECT value FROM detector_health_qa"
            " WHERE pfs_visit_id = :v AND camera_name = :c AND amplifier = '' AND metric = 'nDefects'",
            params={"v": 133025, "c": "b1"},
        )
        assert value == 201178.0
        # It is not judged: no verdict mentions a detector-health metric.
        assert qadb.count("qa_gate", "qa_decided_by IN ('fitXRms', 'nDefects', 'readNoise')") == 0

    def testCoverageMetricsLoad(self, StubButler, tables, qadb):
        """The reduction-coverage rows (isrRan, reduceExposureRan, reductionStage) load as detector health.

        They are whole-detector metrics like any other reductionStats row, so the loader needs no
        change: a detector whose reduction never ran still lands a row in detector_health_qa.
        """
        coverage = []
        for image in tables["iqQaMetrics"][IMAGE].itertuples(index=False):
            base = image._asdict()
            coverage.append({**base, "amplifier": "", "metric": "isrRan", "value": 1.0})
            coverage.append({**base, "amplifier": "", "metric": "reduceExposureRan", "value": 0.0})
            coverage.append({**base, "amplifier": "", "metric": "reductionStage", "value": 1.0})
        tables["reductionStats"] = pd.DataFrame(coverage)
        butler = StubButler(tables, datasets=PACKAGES)
        counts = load(butler, qadb)
        assert counts["detector_health_qa"] == 12  # 4 images x 3 coverage metrics
        stage = qadb.query_scalar(
            "SELECT value FROM detector_health_qa"
            " WHERE pfs_visit_id = :v AND camera_name = :c AND amplifier = '' AND metric = 'reductionStage'",
            params={"v": 133025, "c": "b1"},
        )
        assert stage == 1.0
        # Negative control: coverage is stored, not judged — no verdict is decided by a coverage metric.
        assert (
            qadb.count("qa_gate", "qa_decided_by IN ('isrRan', 'reduceExposureRan', 'reductionStage')") == 0
        )

    def testOnlyReductionStatsCollection(self, StubButler, fixtures, qadb):
        """A collection with reductionStats but no QA numbers loads detector health and no verdicts.

        reductionStats keeps the load alive where the QA-number tables are all empty (it would
        otherwise raise ``LookupError``), and gate() gives no verdict without any numbers.
        """
        butler = StubButler({"reductionStats": reductionRows(fixtures["iqQaMetrics"])}, datasets=PACKAGES)
        counts = load(butler, qadb)
        assert counts["detector_health_qa"] == 16
        assert qadb.count("detector_health_qa") == 16
        assert qadb.count("qa_gate") == 0

    def testMissingDatasetLoadsNothing(self, butler, qadb):
        """A collection without ``reductionStats`` loads as before, with no detector-health rows."""
        counts = load(butler, qadb)
        assert "detector_health_qa" not in counts
        assert qadb.count("detector_health_qa") == 0

    def testReloadReplacesAndDrops(self, StubButler, tables, qadb):
        """A reload replaces an image's rows and removes a metric the new measurement lacks."""
        butler = self.butlerWithHealth(StubButler, tables)
        load(butler, qadb)
        assert qadb.count("detector_health_qa") == 16
        # Change one value and drop the 'gain' rows, then reload.
        rs = tables["reductionStats"]
        one = (
            (rs["visit"] == 133025)
            & (rs["arm"] == "b")
            & (rs["spectrograph"] == 1)
            & (rs["metric"] == "fitXRms")
        )
        assert one.sum() == 1
        rs.loc[one, "value"] = 0.09
        tables["reductionStats"] = rs[rs["metric"] != "gain"]
        counts = load(butler, qadb, now=lambda: LATER)
        assert counts["detector_health_qa"] == 12  # 4 images x 3 metrics
        assert qadb.count("detector_health_qa") == 12
        # Negative control: the gain metric is gone, not left at its old value.
        assert qadb.count("detector_health_qa", "metric = 'gain'") == 0
        updated = qadb.query_scalar(
            "SELECT value FROM detector_health_qa"
            " WHERE pfs_visit_id = :v AND camera_name = :c AND amplifier = '' AND metric = 'fitXRms'",
            params={"v": 133025, "c": "b1"},
        )
        assert updated == 0.09
