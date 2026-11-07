"""Tests for reading qadb back in the Butler tables' shapes (`pfs.drp.qa.qadb.reader`).

Each test loads the fixtures with the loader, as qaActor does, and reads them back.
"""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.calibration import readThresholds
from pfs.drp.qa.metrics.gate import THRESHOLD_FIELDS, judge
from pfs.drp.qa.qadb.loader import loadCollection
from pfs.drp.qa.qadb.reader import (
    THRESHOLDS_COLUMNS,
    THRESHOLDS_FILES_COLUMNS,
    THRESHOLDS_IDS,
    THRESHOLDS_LAYERS_COLUMNS,
    layeredThresholds,
    readQadb,
    thresholdsLayers,
    thresholdsTables,
)

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
        task: thresholdsLayers(tables[source][THRESHOLDS_IDS].iloc[0])[0]
        for task, source in (("imageQualityQa", "iqQaMetrics"), ("dmLineResiduals", "dmLineResidualSummary"))
    }
    for source in ("iqQaMetrics", "dmLineResidualSummary", "dmLineResidualMetrics"):
        assert tables[source][THRESHOLDS_IDS].nunique() == 1 and (tables[source][THRESHOLDS_IDS] != "").all()
    assert thresholdsLayers(tables["dmLineResidualMetrics"][THRESHOLDS_IDS].iloc[0]) == [
        cited["dmLineResiduals"]
    ]
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
    assert (tables["dmLineResidualSummary"][THRESHOLDS_IDS] == "").all()
    assert (tables["dmLineResidualMetrics"][THRESHOLDS_IDS] == "").all()
    assert tables["thresholdsFiles"]["name"].tolist() == ["iqQaThresholds-run25.yaml"]
    assert set(tables["thresholds"]["metric"]) <= {"medFwhm", "linesPerFiber", "pctFlagged"}
    # Negative control: the other task's id is there.
    assert (tables["iqQaMetrics"][THRESHOLDS_IDS] != "").all()


def testEmptyReadHasTheThresholdsColumns(butler, qadb):
    load(butler, qadb)
    tables = readQadb(qadb, "u/test/other")
    assert tables["thresholds"].empty and list(tables["thresholds"].columns) == list(THRESHOLDS_COLUMNS)
    assert tables["thresholdsFiles"].empty
    assert THRESHOLDS_IDS in tables["iqQaMetrics"].columns and tables["thresholdsLayers"].empty
    assert thresholdsTables(tables["thresholds"]) == {}


def testSinceReadsTheThresholdsOfTheImagesRead(butler, qadb):
    """A read of the images loaded since holds the thresholds those images cite."""
    load(butler, qadb)
    load(butler, qadb, visits=[133040], now=LATER)
    stamp = readQadb(qadb, COLLECTION)["iqQaMetrics"]["decidedAt"].max()
    newer = readQadb(qadb, COLLECTION, since=stamp)
    assert set(newer["iqQaMetrics"]["visit"]) == {133040}
    cited = {
        i
        for frame in (newer["iqQaMetrics"], newer["dmLineResidualSummary"])
        for k in frame[THRESHOLDS_IDS]
        for i in thresholdsLayers(k)
    }
    assert set(newer["thresholds"]["thresholdsId"]) == cited and len(cited) == 2


def recordedWithConfigs(fixtures, fullConfigs):
    """Return the fixtures recording each task's file and then its config layer, as the tasks do now."""
    from pfs.drp.qa.metrics.gate import taskThresholdsRecord
    from pfs.drp.qa.qadb.loader import CONFIG_LAYERS

    def record(task, name):
        return taskThresholdsRecord(task, [name], CONFIG_LAYERS[task](fullConfigs[f"{task}_config"]))

    return {
        **fixtures,
        "iqQaMetrics": fixtures["iqQaMetrics"].assign(
            **record("imageQualityQa", "iqQaThresholds-run25.yaml")
        ),
        "dmLineResidualSummary": fixtures["dmLineResidualSummary"].assign(
            **record("dmLineResiduals", "dmLineResidualThresholds-run25.yaml")
        ),
    }


def testEveryLayerReadsBack(StubButler, fixtures, fullConfigs, qadb):
    """Each image names every layer its task judged with, in order, and the config layer's limits are among them."""
    datasets = {"packages": {"drp_qa": "x"}, **fullConfigs}
    load(StubButler(recordedWithConfigs(fixtures, fullConfigs), datasets=datasets), qadb)
    tables = readQadb(qadb, COLLECTION)
    files = tables["thresholdsFiles"].set_index("name")["thresholdsId"]
    for source, task in (
        ("iqQaMetrics", "imageQualityQa"),
        ("dmLineResidualSummary", "dmLineResiduals"),
        ("dmLineResidualMetrics", "dmLineResiduals"),
    ):
        frame = tables[source]
        fileName = (
            "iqQaThresholds-run25.yaml" if task == "imageQualityQa" else "dmLineResidualThresholds-run25.yaml"
        )
        expected = f"{files[fileName]},{files[f'{task}.config']}"
        assert set(frame[THRESHOLDS_IDS]) == {expected}, source
    layers = tables["thresholdsLayers"]
    assert list(layers.columns) == list(THRESHOLDS_LAYERS_COLUMNS)
    assert len(layers) == 16 and set(layers["layer"]) == {0, 1}
    # The layers give judge what the task judged with: the drift rows, which the run25 file lacks, have limits.
    stored = thresholdsTables(tables["thresholds"])
    key = tables["dmLineResidualSummary"][THRESHOLDS_IDS].iloc[0]
    assert thresholdsLayers(key) == [
        files["dmLineResidualThresholds-run25.yaml"],
        files["dmLineResiduals.config"],
    ]
    layered = layeredThresholds(key, stored)
    assert (
        len(layered) == 2
        and "medianDeltaX" not in set(layered[0]["metric"])
        and "medianDeltaX" in set(layered[1]["metric"])
    )
    drift = pd.DataFrame(
        {
            "medianDeltaX": [0.5, 1.5],
            "medianDeltaY": [np.nan, np.nan],
            "robustRmsDeltaX": [np.nan, np.nan],
            "robustRmsDeltaY": [np.nan, np.nan],
            "reference": ["calib", "calib"],
            "description": ["Trace", "Trace"],
            "statusType": ["BOTH", "BOTH"],
        }
    )
    judged = judge(drift, layered)
    judged = judged[judged["metric"] == "medianDeltaX"]
    assert judged["status"].tolist() == ["PASS", "WARN"] and judged["layer"].tolist() == [1, 1]
    # Negative control: the first layer alone (the file) gives the drift no verdict at all.
    assert "medianDeltaX" not in set(judge(drift, layered[:1])["metric"])


def testLayersSinceAndNone(StubButler, fixtures, fullConfigs, configs, qadb):
    """``since`` scopes the layers like the images; an image citing nothing has an empty key."""
    datasets = {"packages": {"drp_qa": "x"}, **fullConfigs}
    load(StubButler(recordedWithConfigs(fixtures, fullConfigs), datasets=datasets), qadb)
    load(
        StubButler(recordedWithConfigs(fixtures, fullConfigs), datasets=datasets),
        qadb,
        visits=[133040],
        now=LATER,
    )
    stamp = readQadb(qadb, COLLECTION)["iqQaMetrics"]["decidedAt"].max()
    newer = readQadb(qadb, COLLECTION, since=stamp)
    assert set(newer["thresholdsLayers"]["visit"]) == {133040}
    assert thresholdsLayers("") == thresholdsLayers(None) == [] and layeredThresholds(None, {}) == []
    none = StubButler(
        {
            **fixtures,
            "dmLineResidualSummary": fixtures["dmLineResidualSummary"].assign(
                thresholdsFiles="", thresholdsSha256=""
            ),
        },
        datasets={"packages": {"drp_qa": "x"}, **configs},
    )
    other = type(qadb)() if type(qadb).__name__ == "SqliteQaDB" else pytest.skip("SQLite only")
    load(none, other)
    tables = readQadb(other, COLLECTION)
    assert (tables["dmLineResidualSummary"][THRESHOLDS_IDS] == "").all()
    empty = readQadb(other, "u/test/none")
    assert THRESHOLDS_IDS in empty["iqQaMetrics"].columns and list(empty["thresholdsLayers"].columns) == list(
        THRESHOLDS_LAYERS_COLUMNS
    )
