"""Tests for reading qadb back in the Butler tables' shapes (`pfs.drp.qa.qadb.reader`).

Each test loads the fixtures with the loader, as qaActor does, and reads them back.
"""

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from pfs.drp.qa.metrics.calibration import readThresholds
from pfs.drp.qa.metrics.gate import THRESHOLD_FIELDS, judge
from pfs.drp.qa.qadb.loader import loadCollection
from pfs.drp.qa.qadb.reader import THRESHOLDS_COLUMNS, THRESHOLDS_FILES_COLUMNS, readQadb, thresholdsTables

COLLECTION = "u/test/qa"
VISITS = [133025, 133040]
NOW = datetime(2026, 10, 26, 12, 0, 0)
LATER = datetime(2026, 10, 26, 12, 5, 0)
IMAGE = ["visit", "arm", "spectrograph"]


def load(butler, qadb, visits=VISITS, now=NOW):
    loadCollection(
        butler, qadb, COLLECTION, visits=visits, pipelineVersions=False, log=lambda _: None, now=lambda: now
    )


@pytest.fixture
def butler(StubButler, fixtures, configs):
    return StubButler(fixtures, datasets={"packages": {"drp_qa": "w.2026.44"}, **configs})


def byImage(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(IMAGE).reset_index(drop=True)


def testRoundTrip(butler, fixtures, qadb):
    """What the loader wrote reads back as the Butler tables, camera split into arm and spectrograph."""
    load(butler, qadb)
    tables = readQadb(qadb, COLLECTION)
    iq, expected = byImage(tables["iqQaMetrics"]), byImage(fixtures["iqQaMetrics"])
    assert iq[IMAGE].equals(expected[IMAGE])
    assert iq["medFwhm"].to_numpy() == pytest.approx(expected["medFwhm"].to_numpy(), rel=1e-6)
    assert iq["qaStatus"].tolist() == expected["qaStatus"].tolist()
    # Negative control: arm and spectrograph come from the camera name the right way round.
    assert not (iq["arm"].astype(str) == iq["spectrograph"].astype(str)).any()
    assert len(tables["dmLineResidualSummary"]) == 4
    metrics = tables["dmLineResidualMetrics"]
    assert len(metrics) == len(fixtures["dmLineResidualMetrics"]) == 128
    assert set(metrics["arm"]) == {"b", "r"} and set(metrics["spectrograph"]) == {1}
    assert sorted(tables["visitTimes"]["pfs_visit_id"]) == VISITS


def testOtherCollectionsAndVisitsAreLeftOut(butler, qadb):
    load(butler, qadb)
    assert all(
        readQadb(qadb, "u/test/other")[name].empty for name in ("iqQaMetrics", "dmLineResidualMetrics")
    )
    iq = readQadb(qadb, COLLECTION, first=133030)["iqQaMetrics"]
    assert set(iq["visit"]) == {133040}


def testLatestRunWins(StubButler, fixtures, configs, qadb):
    """An image judged in two RUNs of the collection reads back once, as the later RUN judged it."""
    datasets = {"packages": {"drp_qa": "x"}, **configs}
    load(StubButler(fixtures, datasets=datasets), qadb)
    again = fixtures["iqQaMetrics"].assign(medFwhm=9.0)
    runs = {tuple(image): "u/test/qa/20261026T010000Z" for image in again[IMAGE].itertuples(index=False)}
    load(StubButler({**fixtures, "iqQaMetrics": again}, runs=runs, datasets=datasets), qadb, now=LATER)
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


def _ordered(table: pd.DataFrame, columns) -> pd.DataFrame:
    """Return a thresholds table's rows in one order, by metric and population, with ``columns``."""
    populations = sorted(column for column in columns if column not in THRESHOLD_FIELDS)
    return table.sort_values(["metric", *populations], key=lambda s: s.map(str), ignore_index=True)[columns]


def _plain(column: pd.Series) -> list:
    """Return a column as plain values, every null `None`."""
    return column.astype(object).where(column.notna(), None).tolist()


def _comparable(judged: pd.DataFrame) -> pd.DataFrame:
    """Return a `judge` table sorted by image and metric, its population labels order-free."""
    frame = judged.sort_values(["row", "metric"], ignore_index=True)
    return frame.assign(population=frame["population"].map(lambda label: tuple(sorted(label.split(" ")))))


def testThresholdsReadBackAsTheFilesJudgedWith(butler, qadb, thresholdsFiles):
    """Each image cites its task's thresholds, whose stored entries rebuild the file's table exactly."""
    load(butler, qadb)
    tables = readQadb(qadb, COLLECTION)
    files = tables["thresholdsFiles"].set_index("thresholdsId")
    assert list(tables["thresholdsFiles"].columns) == list(THRESHOLDS_FILES_COLUMNS)
    assert set(files["name"]) == {path.name for path in thresholdsFiles.values()}
    assert files["nEntries"].sum() == len(tables["thresholds"])
    assert set(THRESHOLDS_COLUMNS) <= set(tables["thresholds"].columns)
    cited = {
        task: int(tables[source]["thresholdsId"].iloc[0])
        for task, source in (("imageQualityQa", "iqQaMetrics"), ("dmLineResiduals", "dmLineResidualSummary"))
    }
    for source in ("iqQaMetrics", "dmLineResidualSummary", "dmLineResidualMetrics"):
        assert tables[source]["thresholdsId"].nunique() == 1 and tables[source]["thresholdsId"].notna().all()
    assert tables["dmLineResidualMetrics"]["thresholdsId"].iloc[0] == cited["dmLineResiduals"]
    assert cited["imageQualityQa"] != cited["dmLineResiduals"]

    stored = thresholdsTables(tables["thresholds"])
    for task, thresholdsId in cited.items():
        assert files.loc[thresholdsId, "name"] == thresholdsFiles[task].name
        _, filed = readThresholds(thresholdsFiles[task])
        table = stored[thresholdsId]
        assert sorted(table.columns) == sorted(filed.columns)
        expected, got = _ordered(filed, filed.columns), _ordered(table, filed.columns)
        for column in filed.columns:
            if column in ("warn", "fail"):
                assert got[column].to_numpy() == pytest.approx(expected[column].to_numpy(), nan_ok=True), (
                    column
                )
            else:
                assert _plain(got[column]) == _plain(expected[column]), column

    # Judging the images with the stored entries is judging them with the file.
    iq = tables["iqQaMetrics"]
    fromQadb = _comparable(judge(iq, stored[cited["imageQualityQa"]]))
    fromFile = _comparable(judge(iq, thresholdsFiles["imageQualityQa"]))
    columns = ["row", "metric", "status", "warn", "fail", "population"]
    pd.testing.assert_frame_equal(fromQadb[columns], fromFile[columns])
    assert (fromQadb["status"] != "").any()
    # Negative control: the stored columns alone lose the keys only the population names (species, seqName),
    # and the gate then finds equally specific entries.
    naive = stored[cited["imageQualityQa"]].drop(columns=["species", "seqName"])
    with pytest.raises(ValueError, match="equally specific"):
        judge(iq, naive)


def testThresholdsTablesTakeOnlyWhatTheGateTakes(butler, qadb):
    """A judge table has no id or key column (the gate would take them for populations) and no unnamed population."""
    load(butler, qadb)
    tables = readQadb(qadb, COLLECTION)
    for table in thresholdsTables(tables["thresholds"]).values():
        assert not {"thresholdsId", "population"} & set(table.columns)
        populations = [column for column in table.columns if column not in THRESHOLD_FIELDS]
        assert all(table[column].notna().any() for column in populations)
    # Negative control: the entries table itself keeps them, for provenance.
    assert {"thresholdsId", "population"} <= set(tables["thresholds"].columns)


def testAnImageCitingNoThresholdsHasNone(StubButler, fixtures, configs, qadb):
    """A task whose file the loader could not cite leaves the id null, and no entries are read for it."""
    datasets = {
        "packages": {"drp_qa": "x"},
        **configs,
        "dmLineResiduals_config": SimpleNamespace(thresholdsFile="gone.yaml"),
    }
    load(StubButler(fixtures, datasets=datasets), qadb)
    tables = readQadb(qadb, COLLECTION)
    assert tables["dmLineResidualSummary"]["thresholdsId"].isna().all()
    assert tables["dmLineResidualMetrics"]["thresholdsId"].isna().all()
    assert tables["thresholdsFiles"]["name"].tolist() == ["iqQaThresholds-run25.yaml"]
    assert set(tables["thresholds"]["metric"]) <= {"medFwhm", "linesPerFiber", "pctFlagged"}
    # Negative control: the other task's id is there.
    assert tables["iqQaMetrics"]["thresholdsId"].notna().all()


def testEmptyReadHasTheThresholdsColumns(butler, qadb):
    load(butler, qadb)
    tables = readQadb(qadb, "u/test/other")
    assert tables["thresholds"].empty and list(tables["thresholds"].columns) == list(THRESHOLDS_COLUMNS)
    assert tables["thresholdsFiles"].empty
    assert "thresholdsId" in tables["iqQaMetrics"].columns
    assert thresholdsTables(tables["thresholds"]) == {}


def testSinceReadsTheThresholdsOfTheImagesRead(butler, qadb):
    """A read of the images loaded since holds the thresholds those images cite."""
    load(butler, qadb)
    load(butler, qadb, visits=[133040], now=LATER)
    stamp = readQadb(qadb, COLLECTION)["iqQaMetrics"]["decidedAt"].max()
    newer = readQadb(qadb, COLLECTION, since=stamp)
    assert set(newer["iqQaMetrics"]["visit"]) == {133040}
    cited = set(newer["iqQaMetrics"]["thresholdsId"]) | set(newer["dmLineResidualSummary"]["thresholdsId"])
    assert set(newer["thresholds"]["thresholdsId"]) == cited and len(cited) == 2
