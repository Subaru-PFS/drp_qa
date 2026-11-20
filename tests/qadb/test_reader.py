"""Tests for reading qadb back in the shapes the gate reads (`pfs.drp.qa.qadb.reader`).

Each test loads tables with the loader, as qaActor does, and reads them back.
"""

from datetime import datetime

import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import NORMAL_KEYS, gate
from pfs.drp.qa.metrics.normalValues import NORMAL_COLUMNS
from pfs.drp.qa.qadb.loader import loadCollection
from pfs.drp.qa.qadb.reader import (
    GATE_COLUMNS,
    PROCESSING_ID,
    collectionLoaded,
    readGate,
    readNormalValues,
    readQadb,
)

COLLECTION = "u/test/qa"
VISITS = [133025, 133040]
NOW = datetime(2026, 10, 26, 12, 0, 0)
LATER = datetime(2026, 10, 26, 12, 5, 0)
IMAGE = ["visit", "arm", "spectrograph"]
NUMBERS = ("iqQaMetrics", "dmLineResidualSummary", "dmLineResidualMetrics", "twilightMetrics")

#: Normal widths of the fixture's images: r1's arc is +10 % (FAIL), b1's quartz +2.4 % (PASS) and r1's
#: quartz +3.9 % (WARN).
WIDTHS = {
    ("medFwhm", "r1", "arc", "Argon"): 2.5,
    ("medFwhm", "b1", "quartz", ""): 3.1,
    ("medFwhm", "r1", "quartz", ""): 3.2,
}


def load(butler, qadb, visits=VISITS, now=NOW, **kwargs):
    return loadCollection(
        butler,
        qadb,
        COLLECTION,
        visits=visits,
        pipelineVersions=False,
        log=lambda _: None,
        now=lambda: now,
        **kwargs,
    )


@pytest.fixture
def butler(StubButler, fixtures):
    return StubButler(fixtures, datasets={"packages": {"drp_qa": "w.2026.44"}})


def byImage(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(IMAGE).reset_index(drop=True)


def testRoundTrip(butler, fixtures, qadb):
    """What the loader wrote reads back as the Butler tables, camera split into arm and spectrograph."""
    load(butler, qadb)
    tables = readQadb(qadb, COLLECTION)
    assert set(tables) == {*NUMBERS, "visitTimes"}
    iq, expected = byImage(tables["iqQaMetrics"]), byImage(fixtures["iqQaMetrics"])
    assert iq[IMAGE].equals(expected[IMAGE])
    assert iq["medFwhm"].to_numpy() == pytest.approx(expected["medFwhm"].to_numpy(), rel=1e-6)
    assert iq["obsType"].tolist() == expected["obsType"].tolist()
    assert iq["traceOnly"].tolist() == expected["traceOnly"].tolist() == [False] * 4
    # Negative control: arm and spectrograph come from the camera name the right way round.
    assert not (iq["arm"].astype(str) == iq["spectrograph"].astype(str)).any()
    assert len(tables["dmLineResidualSummary"]) == 4
    metrics = tables["dmLineResidualMetrics"]
    assert len(metrics) == len(fixtures["dmLineResidualMetrics"]) == 128
    assert set(metrics["arm"]) == {"b", "r"} and set(metrics["spectrograph"]) == {1}
    assert sorted(tables["visitTimes"]["pfs_visit_id"]) == VISITS
    assert tables["twilightMetrics"].empty and "twilightLevelOffset" in tables["twilightMetrics"]


def testNoTableReadBackHoldsAVerdict(butler, qadb):
    """The numbers come back as numbers: the verdict is `readGate`'s."""
    load(butler, qadb)
    tables = readQadb(qadb, COLLECTION)
    verdicts = {"qaStatus", "qaDecidedBy", "qaReason", "recommendedAction", "status", "provisional"}
    for name in NUMBERS:
        assert not verdicts & set(tables[name].columns), name
        assert not any("thresholds" in column.lower() for column in tables[name].columns), name
    assert "thresholds" not in tables and "thresholdsFiles" not in tables
    # Negative control: the gate's table is where they are.
    assert {"qaStatus", "qaDecidedBy", "qaReason"} <= set(readGate(qadb, COLLECTION).columns)


def testEveryRowNamesItsProcessing(StubButler, fixtures, qadb):
    """`PROCESSING_ID` is the ``data_processing_pipe2d`` row each table's rows were loaded under."""
    runs = {(133040, "b", 1): "u/test/qa/second", (133040, "r", 1): "u/test/qa/second"}
    load(StubButler(fixtures, runs=runs), qadb)
    ids = qadb.query_dataframe(
        "SELECT processing_id, run_name FROM data_processing_pipe2d WHERE collection_name = :c",
        params={"c": COLLECTION},
    ).set_index("run_name")["processing_id"]
    tables = readQadb(qadb, COLLECTION)
    for name in ("iqQaMetrics", "dmLineResidualSummary", "dmLineResidualMetrics"):
        byVisit = tables[name].groupby("visit")[PROCESSING_ID].agg(set)
        assert byVisit[133025] == {ids["u/test/qa/20261026T000000Z"]}, name
        assert byVisit[133040] == {ids["u/test/qa/second"]}, name
    # Negative control: the two RUNs are two processings, so the column tells them apart.
    assert ids.nunique() == 2
    assert PROCESSING_ID in tables["twilightMetrics"]


def testLineRowsCarryTheirImagesCameraSetAndObsType(butler, fixtures, qadb):
    """The long table has neither column in qadb; they come from the image's summary row, as the gate needs."""
    load(butler, qadb)
    metrics = readQadb(qadb, COLLECTION)["dmLineResidualMetrics"]
    stored = set(qadb.query_dataframe("SELECT * FROM dm_line_residual_qa_metric").columns)
    assert not {"camera_set", "obs_type"} & stored
    assert set(metrics["cameraSet"]) == {"brn"}
    byVisit = metrics.groupby("visit")["obsType"].agg(set)
    assert byVisit[133025] == {"arc"} and byVisit[133040] == {"trace"}
    keys = [*IMAGE, "reference", "description", "statusType", "metric"]
    expected = fixtures["dmLineResidualMetrics"].astype({"arm": str}).sort_values(keys, ignore_index=True)
    got = metrics.sort_values(keys, ignore_index=True)
    assert got["value"].to_numpy() == pytest.approx(expected["value"].to_numpy(), rel=1e-6)
    assert got["obsType"].tolist() == expected["obsType"].tolist()


def testOtherCollectionsAndVisitsAreLeftOut(butler, qadb):
    load(butler, qadb)
    assert all(readQadb(qadb, "u/test/other")[name].empty for name in NUMBERS)
    iq = readQadb(qadb, COLLECTION, first=133030)["iqQaMetrics"]
    assert set(iq["visit"]) == {133040}
    assert set(readQadb(qadb, COLLECTION, last=133030)["dmLineResidualMetrics"]["visit"]) == {133025}


def testLatestRunWins(StubButler, fixtures, qadb):
    """An image measured in two RUNs of the collection reads back once, as the later RUN measured it."""
    load(StubButler(fixtures), qadb)
    again = fixtures["iqQaMetrics"].assign(medFwhm=9.0)
    runs = {tuple(image): "u/test/qa/20261026T010000Z" for image in again[IMAGE].itertuples(index=False)}
    load(StubButler({**fixtures, "iqQaMetrics": again}, runs=runs), qadb, now=LATER)
    assert qadb.count("image_quality_qa") == 8  # both RUNs' rows are kept
    iq = readQadb(qadb, COLLECTION)["iqQaMetrics"]
    assert len(iq) == 4
    assert (iq["medFwhm"] == 9.0).all()


def testSinceReadsOnlyNewerLoads(butler, qadb):
    """``since`` is inclusive, and leaves out images not loaded again since."""
    load(butler, qadb)
    first = readQadb(qadb, COLLECTION)
    stamp = first["iqQaMetrics"]["decidedAt"].max()
    assert len(readQadb(qadb, COLLECTION, since=stamp)["iqQaMetrics"]) == 4
    load(butler, qadb, visits=[133040], now=LATER)
    stamp = readQadb(qadb, COLLECTION)["iqQaMetrics"]["decidedAt"].max()
    newer = readQadb(qadb, COLLECTION, since=stamp)
    assert set(newer["iqQaMetrics"]["visit"]) == {133040}
    assert set(newer["dmLineResidualMetrics"]["visit"]) == {133040}


def testATaskRerunAloneReadsBackItsNewRows(StubButler, fixtures, qadb):
    """A dmLineResiduals rerun alone into a newer RUN: its new rows come back, beside the older IQ rows."""
    load(StubButler(fixtures), qadb)
    before = readQadb(qadb, COLLECTION)
    summary = fixtures["dmLineResidualSummary"].assign(nSpecies=99)
    metrics = fixtures["dmLineResidualMetrics"].assign(value=-0.25)
    rerun = "u/test/qa/20261026T020000Z"
    runs = {
        (name, *image): rerun
        for name in ("dmLineResidualSummary", "dmLineResidualMetrics")
        for image in summary[IMAGE].itertuples(index=False, name=None)
    }
    # The collection is a chain: the load finds the IQ tables in the older RUN and the residuals in the newer.
    chained = {**fixtures, "dmLineResidualSummary": summary, "dmLineResidualMetrics": metrics}
    load(StubButler(chained, runs=runs), qadb, now=LATER)
    after = readQadb(qadb, COLLECTION)
    assert (after["dmLineResidualSummary"]["nSpecies"] == 99).all()
    assert (after["dmLineResidualMetrics"]["value"] == -0.25).all()
    assert len(after["dmLineResidualMetrics"]) == len(before["dmLineResidualMetrics"])
    assert qadb.count("dm_line_residual_qa") == 8  # the older RUN's rows are still stored
    # The IQ rows are the older RUN's, unchanged, under their own processing.
    columns = [column for column in before["iqQaMetrics"].columns if column != "decidedAt"]
    pd.testing.assert_frame_equal(
        byImage(after["iqQaMetrics"])[columns], byImage(before["iqQaMetrics"])[columns]
    )
    assert set(after["iqQaMetrics"][PROCESSING_ID]) != set(after["dmLineResidualSummary"][PROCESSING_ID])
    # The gate decided again, on the same row, so a since read at the rerun returns these images.
    assert qadb.count("qa_gate") == 4
    stamp = after["iqQaMetrics"]["decidedAt"].max()
    assert stamp > before["iqQaMetrics"]["decidedAt"].max()
    assert len(readQadb(qadb, COLLECTION, since=stamp)["dmLineResidualSummary"]) == 4
    # Negative control: before the rerun, the summary read back as first measured.
    assert (before["dmLineResidualSummary"]["nSpecies"] != 99).all()


class TestReadNormalValues:
    def testAsTheGateTakesThem(self, qadb, storeValues):
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000, count=4, when=NOW)
        values = readNormalValues(qadb, COLLECTION)
        assert set(values.columns) == set(NORMAL_COLUMNS) and len(values) == 3
        assert set(NORMAL_KEYS) | {"value", "count", "validFrom"} <= set(values.columns)
        got = {tuple(row[list(NORMAL_KEYS)]): row["value"] for _, row in values.iterrows()}
        assert got == WIDTHS
        assert values["value"].dtype == "float64"
        assert values["count"].dtype == "int64" and set(values["count"]) == {4}
        assert values["validFrom"].dtype == "int64" and set(values["validFrom"]) == {133000}
        assert set(values["collection"]) == {COLLECTION} and set(values["acceptedBy"]) == {"tester"}
        assert (pd.to_datetime(values["acceptedAt"]) == pd.Timestamp(NOW)).all()
        # A value with no lamp or species has an empty selector, a string like the others.
        assert "" in set(values["selector"]) and values["selector"].map(type).eq(str).all()

    def testEveryAcceptedRowInOrder(self, qadb, storeValues):
        """Each ``valid_from`` of a key is a row: the gate, not the reader, takes the one in force."""
        key = ("medFwhm", "b1", "quartz", "")
        storeValues(qadb, COLLECTION, {key: 3.3}, validFrom=134000)
        storeValues(qadb, COLLECTION, {key: 3.1}, validFrom=133000)
        values = readNormalValues(qadb, COLLECTION)
        assert values["validFrom"].tolist() == [133000, 134000]
        assert values["value"].tolist() == [3.1, 3.3]

    def testOnlyTheCollections(self, qadb, storeValues):
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
        other = readNormalValues(qadb, "u/test/other")
        assert other.empty and set(other.columns) == set(NORMAL_COLUMNS)
        # An empty table is one the gate takes: nothing relative is judged, nothing breaks.
        assert gate(normalValues=other).empty


class TestJudgedAsAtLoad:
    """What the loader wrote, read back, gives the gate the same verdicts as at load."""

    def judge(self, qadb):
        tables = readQadb(qadb, COLLECTION)
        verdicts = gate(
            iq=tables["iqQaMetrics"],
            lines=tables["dmLineResidualMetrics"],
            twilight=tables["twilightMetrics"],
            normalValues=readNormalValues(qadb, COLLECTION),
        )
        return verdicts.sort_values(IMAGE, ignore_index=True)

    def compare(self, qadb):
        atLoad = readGate(qadb, COLLECTION).sort_values(IMAGE, ignore_index=True)
        again = self.judge(qadb)
        assert again[IMAGE].to_numpy().tolist() == atLoad[IMAGE].to_numpy().tolist()
        for column in ("frameType", "qaStatus", "qaDecidedBy", "qaReason"):
            assert again[column].tolist() == atLoad[column].tolist(), column
        return atLoad

    def testLampFrames(self, butler, qadb, storeValues):
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
        load(butler, qadb)
        atLoad = self.compare(qadb)
        assert atLoad["qaStatus"].tolist() == ["PASS", "FAIL", "PASS", "WARN"]
        assert atLoad["qaDecidedBy"].tolist() == ["", "width", "", "width"]
        assert atLoad["frameType"].tolist() == ["arc", "arc", "quartz", "quartz"]
        # Negative control: without the normal values read back, the relative checks would not be judged.
        tables = readQadb(qadb, COLLECTION)
        bare = gate(iq=tables["iqQaMetrics"], lines=tables["dmLineResidualMetrics"])
        assert bare["qaStatus"].tolist() == ["PASS"] * 4

    def testWithoutNormalValues(self, butler, qadb):
        load(butler, qadb)
        assert self.compare(qadb)["qaStatus"].tolist() == ["PASS"] * 4

    def testTwilights(self, twilight, fixtures, qadb, storeValues):
        """Lamp frames and a complete twilight sequence in one load: every kind of table read back."""
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
        butler = twilight.butler()
        butler.tables.update(
            iqQaMetrics=pd.concat([twilight.iq(), fixtures["iqQaMetrics"]], ignore_index=True),
            dmLineResidualSummary=fixtures["dmLineResidualSummary"],
            dmLineResidualMetrics=fixtures["dmLineResidualMetrics"],
        )
        counts = load(
            butler,
            qadb,
            visits=[*twilight.VISITS, *VISITS],
            opdb=twilight.OpDB(),
            readSequence=twilight.Sequences(),
        )
        assert counts["twilight_qa"] == 48 and counts["qa_gate"] == 52
        tables = readQadb(qadb, COLLECTION)
        metrics = tables["twilightMetrics"]
        assert len(metrics) == 48 and set(metrics["sequence"]) == {twilight.SEQUENCE}
        assert "iic_sequence_id" not in metrics
        assert metrics["tooFaint"].map(type).eq(bool).all() and not metrics["tooFaint"].any()
        atLoad = self.compare(qadb)
        assert set(atLoad["qaStatus"]) == {"PASS", "WARN", "FAIL", "UNKNOWN"}
        twilights = atLoad[atLoad["frameType"] == "twilight"]
        failed = twilights[twilights["qaStatus"] == "FAIL"]
        assert failed[IMAGE].to_numpy().tolist() == [[503, "r", 2]]
        assert failed["qaDecidedBy"].tolist() == ["twilightLevel"]
        # Negative control: without the twilight metrics read back those images have no verdict.
        bare = gate(iq=tables["iqQaMetrics"], lines=tables["dmLineResidualMetrics"])
        assert set(bare.loc[bare["frameType"] == "twilight", "qaStatus"]) == {"UNKNOWN"}


class TestReadGate:
    def testColumnsAndContent(self, butler, qadb, storeValues):
        storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
        load(butler, qadb, mode="comparison")
        rows = readGate(qadb, COLLECTION)
        assert list(rows.columns) == list(GATE_COLUMNS)
        assert list(rows.columns) == [
            *IMAGE,
            "frameType",
            "qaStatus",
            "qaDecidedBy",
            "qaReason",
            "mode",
            "decidedAt",
            "run",
            PROCESSING_ID,
        ]
        assert not {"decidedByTask", "provisional"} & set(rows.columns)
        # The processing is that of the row returned: the one a later verdict on the image replaces.
        stored = qadb.query_dataframe(
            "SELECT g.processing_id, g.pfs_visit_id, g.camera_name FROM qa_gate g"
            " JOIN data_processing_pipe2d p ON p.processing_id = g.processing_id"
            " WHERE p.collection_name = :c ORDER BY g.pfs_visit_id, g.camera_name",
            params={"c": COLLECTION},
        )
        assert rows[PROCESSING_ID].tolist() == stored["processing_id"].tolist()
        assert rows[PROCESSING_ID].dtype == "int64"
        assert set(rows[PROCESSING_ID]) == set(readQadb(qadb, COLLECTION)["iqQaMetrics"][PROCESSING_ID])
        assert rows[IMAGE].to_numpy().tolist() == [[v, a, 1] for v in VISITS for a in "br"]
        assert rows["frameType"].tolist() == ["arc", "arc", "quartz", "quartz"]
        assert rows["qaStatus"].tolist() == ["PASS", "FAIL", "PASS", "WARN"]
        assert set(rows["mode"]) == {"comparison"} and set(rows["run"]) == {"u/test/qa/20261026T000000Z"}
        assert (rows["decidedAt"] == pd.Timestamp(NOW)).all()

    def testTheProcessingIsTheLatestRowsNotAnEarlierOnes(self, StubButler, fixtures, qadb):
        """An image with gate rows under two processings reads back with the later one's id and RUN."""
        load(StubButler(fixtures), qadb)
        first = readGate(qadb, COLLECTION)
        newer = "u/test/qa/20261026T010000Z"
        runs = {tuple(image): newer for image in fixtures["iqQaMetrics"][IMAGE].itertuples(index=False)}
        load(StubButler(fixtures, runs=runs), qadb, now=LATER)
        rows = readGate(qadb, COLLECTION)
        assert qadb.count("qa_gate") == 8 and len(rows) == 4
        newId = qadb.query_scalar(
            "SELECT processing_id FROM data_processing_pipe2d WHERE run_name = :r", params={"r": newer}
        )
        assert set(rows[PROCESSING_ID]) == {newId} and set(rows["run"]) == {newer}
        # Negative control: the earlier rows' processing is another, and lower.
        assert set(first[PROCESSING_ID]) != {newId} and first[PROCESSING_ID].max() < newId

    def testTimesWithAndWithoutAFractionOfASecond(self, butler, qadb):
        """Rows decided on the whole second and rows decided between seconds read back together.

        SQLite returns the times as text, with the fraction only when there is one; a visit judged again
        by the clock beside one loaded at a whole second is such a mix.
        """
        between = datetime(2026, 10, 26, 12, 5, 0, 250000)
        load(butler, qadb)
        load(butler, qadb, visits=[133040], now=between)
        rows = readGate(qadb, COLLECTION)
        assert rows["decidedAt"].tolist() == [pd.Timestamp(NOW)] * 2 + [pd.Timestamp(between)] * 2
        assert rows["decidedAt"].dtype.kind == "M"
        # Negative control: the fraction is kept, not rounded away to make the two alike.
        assert rows["decidedAt"].iloc[-1].microsecond == 250000

    def testCollectionLoaded(self, butler, qadb):
        assert not collectionLoaded(qadb, COLLECTION)
        load(butler, qadb)
        assert collectionLoaded(qadb, COLLECTION) and not collectionLoaded(qadb, "u/test/typo")
