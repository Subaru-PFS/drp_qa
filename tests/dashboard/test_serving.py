"""Tests of the dashboard's metrics table and its DuckDB store."""

from pathlib import Path

import pandas as pd
import pytest

from pfs.drp.qa.dashboard import schema
from pfs.drp.qa.dashboard.schema import NOTE
from pfs.drp.qa.dashboard.serving import (
    VERDICT,
    buildMetricsTable,
    configDefaults,
    firstThresholdsFile,
    inferSequences,
    noteRows,
)
from pfs.drp.qa.dashboard.store import MetricsStore, visitTimeline
from pfs.drp.qa.metrics.gate import STATUS_ORDER


def _thresholds() -> pd.DataFrame:
    """Return thresholds judging only medFwhm: WARN at 3, FAIL at 4, for arm b."""
    return pd.DataFrame(
        {
            "metric": ["medFwhm"],
            "arm": ["b"],
            "warn": [3.0],
            "fail": [4.0],
            "higherIsWorse": [True],
            "absolute": [False],
            "provenance": ["test"],
        }
    )


def _iq() -> pd.DataFrame:
    """Three b-arm arcs of one sequence, then a trace far away: FWHM 2.5, 3.5, 4.5, 2.0."""
    return pd.DataFrame(
        {
            "visit": [100, 101, 102, 200],
            "arm": ["b"] * 4,
            "spectrograph": [1] * 4,
            "obsType": ["arc", "arc", "arc", "trace"],
            "seqName": ["Arc: Neon"] * 3 + ["Trace"],
            "medFwhm": [2.5, 3.5, 4.5, 2.0],
            "traceOnly": [False] * 4,
            # 101's stored verdict is stale: gated with older thresholds.
            "qaStatus": ["PASS", "PASS", "FAIL", "PASS"],
            "qaReason": ["", "", "medFWHM=4.50px >= fail threshold 4px", ""],
            "calexpRun": ["run/1"] * 4,
        }
    )


def _lines() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        {
            "visit": [100, 101, 102, 200],
            "arm": ["b"] * 4,
            "spectrograph": [1] * 4,
            "qaStatus": ["WARN", "PASS", "PASS", "UNKNOWN"],
            "qaReason": ["calib NeI: |medianDeltaY|=1.2px >= warn threshold 1px", "", "", ""],
            "recommendedAction": ["APPLY_SHIFT", "NOMINAL", "NOMINAL", "INSUFFICIENT_DATA"],
        }
    )
    metrics = pd.DataFrame(
        {
            "visit": [100, 100],
            "arm": ["b", "b"],
            "spectrograph": [1, 1],
            "cameraSet": ["brn", "brn"],
            "reference": ["calib", "visit"],
            "description": ["NeI", "NeI"],
            "statusType": ["BOTH", "RESERVED"],
            "metric": ["medianDeltaY", "robustRmsDeltaY"],
            "value": [1.2, 0.05],
            "status": ["WARN", ""],
        }
    )
    return summary, metrics


def testStatusOrderMatchesGate():
    """The light schema module copies the gate's order rather than importing scipy with it."""
    assert schema.STATUS_ORDER == STATUS_ORDER


def testInferSequences():
    """A new seqName or a gap of more than three visits starts a sequence; opdb ids win."""
    images = pd.DataFrame({"visit": [1, 2, 3, 10, 11, 12], "seqName": ["A", "A", "B", "B", "B", "B"]})
    inferred = inferSequences(images).set_index("visit")
    keys = inferred["sequenceKey"]
    assert keys[1] == keys[2] != keys[3]
    assert keys[3] != keys[10] == keys[11] == keys[12]
    assert (keys < 0).all()
    assert inferred["timeExpStart"].isna().all()
    # Negative control: grouping by seqName alone would merge 3 and 10.
    bySeqName = images.join(inferred, on="visit").groupby("seqName")["sequenceKey"].nunique()
    assert bySeqName["B"] == 2

    listing = pd.DataFrame(
        {
            "pfs_visit_id": [1, 2],
            "time_exp_start": pd.to_datetime(["2026-03-01 18:00", "2026-03-01 18:05"]),
            "iic_sequence_id": [555, 555],
        }
    )
    withOpdb = inferSequences(images, listing).set_index("visit")
    assert withOpdb.loc[1, "sequenceKey"] == 555
    assert withOpdb.loc[1, "sequenceSource"] == "opdb"
    assert withOpdb.loc[3, "sequenceSource"] == "inferred"
    assert withOpdb.loc[2, "timeExpStart"] == pd.Timestamp("2026-03-01 18:05")


def testBuildMetricsTable():
    summary, metrics = _lines()
    table = buildMetricsTable(_iq(), _thresholds(), summary, metrics)
    assert list(table.columns) == list(schema.METRICS_COLUMNS)
    fwhm = table[(table["source"] == "iqQaMetrics") & (table["metric"] == "medFwhm")].set_index("visit")
    assert fwhm["status"].to_dict() == {100: "PASS", 101: "WARN", 102: "FAIL", 200: "PASS"}
    assert (fwhm["warn"] == 3.0).all() and (fwhm["fail"] == 4.0).all()
    verdicts = table[table["metric"] == VERDICT]
    assert set(verdicts["source"]) == {"iqQaMetrics", "dmLineResidualSummary"}
    assert len(verdicts) == 8
    # The stored verdict is kept, not replaced by the re-judged one.
    stored = verdicts[verdicts["source"] == "iqQaMetrics"].set_index("visit")["status"]
    assert stored[101] == "PASS"
    assert table["obsType"].ne("").all()


def testStore():
    summary, metrics = _lines()
    store = MetricsStore(buildMetricsTable(_iq(), _thresholds(), summary, metrics), {"collection": "c"})
    assert store.provenance == {"collection": "c"}
    images = store.images().set_index("visit")
    # Worst of the sources: 100 is PASS in iqQaMetrics but WARN in dmLineResidualSummary.
    assert images["status"].to_dict() == {100: "WARN", 101: "PASS", 102: "FAIL", 200: "PASS"}
    # Negative control: iqQaMetrics alone would call 100 PASS.
    assert images.loc[100, "iqStatus"] == "PASS"
    assert store.images(101, 102)["visit"].tolist() == [101, 102]
    assert store.visits()["visit"].tolist() == [100, 101, 102, 200]

    calib = store.trend("dmLineResidualMetrics", "medianDeltaY", reference="calib", statusType="BOTH")
    assert calib["value"].tolist() == [1.2]
    assert store.trend("dmLineResidualMetrics", "medianDeltaY", reference="visit").empty
    assert store.trend("iqQaMetrics", "medFwhm", 101, 200)["visit"].tolist() == [101, 102, 200]


def testLineThresholdsAreRecorded():
    """The task stores line-residual verdicts without thresholds; the table records the ones judged with."""
    from pfs.drp.qa.metrics.lineResiduals import residualThresholds

    summary, metrics = _lines()
    layers = [residualThresholds(1.0, 2.0, 1.0, 2.0, 0.1, 0.2, 0.1, 0.2)]
    table = buildMetricsTable(_iq(), _thresholds(), summary, metrics, lineThresholds=layers)
    lines = table[(table["source"] == "dmLineResidualMetrics")].set_index("reference")
    assert lines.loc["calib", ["warn", "fail"]].tolist() == [1.0, 2.0]
    assert lines.loc["visit", ["warn", "fail"]].tolist() == [0.1, 0.2]
    # The stored verdict is kept as it is.
    assert lines.loc["calib", "status"] == "WARN" and lines.loc["visit", "status"] == ""
    # Negative control: without the layers, no thresholds are invented.
    bare = buildMetricsTable(_iq(), _thresholds(), summary, metrics)
    assert bare.loc[bare["source"] == "dmLineResidualMetrics", ["warn", "fail"]].isna().all().all()


def testConfigDefaultsReadFromSource():
    defaults = configDefaults("pfs.drp.qa.imageQualityQa", "ImageQualityQaConfig")
    assert defaults.thresholdsFile == "iqQaThresholds-run25.yaml"
    assert defaults.dxCenterWarnThreshold < defaults.dxCenterFailThreshold
    assert isinstance(defaults.flagRateWarnThreshold, dict) and "b" in defaults.flagRateWarnThreshold
    lineDefaults = configDefaults("pfs.drp.qa.dmLineResiduals", "DmLineResidualsConfig")
    assert lineDefaults.yWarnThreshold < lineDefaults.yFailThreshold
    # Negative control: a class the module does not define is an error, not empty defaults.
    with pytest.raises(LookupError):
        configDefaults("pfs.drp.qa.imageQualityQa", "NoSuchConfig")


def testNoteRows():
    """A visit's note, a camera's annotation, and a sequence's note on each of its visits."""
    notes = pd.DataFrame(
        {
            "pfs_visit_id": pd.array([100, 101, pd.NA], dtype="Int64"),
            "iic_sequence_id": pd.array([pd.NA, pd.NA, 7], dtype="Int64"),
            "camera": [None, "r2", None],
            "data_flag": pd.array([pd.NA, 1, pd.NA], dtype="Int64"),
            "note": ["Cloudy ", "bad amp", "SM1 lamp off"],
            "source": ["obslog", "sps_annotation", "obslog_sequence"],
        }
    )
    listing = pd.DataFrame(
        {"pfs_visit_id": [100, 101, 102], "iic_sequence_id": pd.array([7, 7, 8], dtype="Int64")}
    )
    rows = noteRows(notes, listing).sort_values(["visit", "reason"]).reset_index(drop=True)
    assert rows[["visit", "arm", "spectrograph", "reason"]].values.tolist() == [
        [100, "", 0, "Cloudy"],
        [100, "", 0, "SM1 lamp off"],
        [101, "", 0, "SM1 lamp off"],
        [101, "r", 2, "bad amp"],
    ]
    assert (rows["metric"] == NOTE).all()
    # Negative control: without the listing, a sequence's note reaches no visit.
    assert "SM1 lamp off" not in set(noteRows(notes)["reason"])


def testStoreNotesKeepImagesClean():
    notes = pd.DataFrame(
        {
            "pfs_visit_id": pd.array([100], dtype="Int64"),
            "iic_sequence_id": pd.array([pd.NA], dtype="Int64"),
            "camera": [None],
            "data_flag": pd.array([pd.NA], dtype="Int64"),
            "note": ["Cloudy"],
            "source": ["obslog"],
        }
    )
    summary, metrics = _lines()
    store = MetricsStore(buildMetricsTable(_iq(), _thresholds(), summary, metrics, notes=notes))
    assert store.notes()[["visit", "note"]].values.tolist() == [[100, "Cloudy"]]
    # The note is not an image: the rollup still has one row per image.
    assert len(store.images()) == 4


def testVisitTimelineNights():
    """A visit after midnight belongs to the night that began the evening before."""
    visits = pd.DataFrame(
        {
            "visit": [10, 11, 12, 13],
            "timeExpStart": pd.to_datetime(
                ["2026-03-09 19:00", "2026-03-10 06:13", "2026-03-10 19:30", None]
            ),
        }
    )
    timeline = visitTimeline(visits).set_index("visit")
    assert timeline["night"].to_dict() == {10: "2026-03-09", 11: "2026-03-09", 12: "2026-03-10", 13: ""}
    assert timeline["position"].tolist() == [0, 1, 2, 3]
    assert timeline.loc[11, "clock"] == "06:13"
    # Negative control: the calendar date would split the first night at midnight.
    assert timeline.loc[11, "time"].strftime("%Y-%m-%d") != timeline.loc[11, "night"]


def testNotesOnVisitsNotJudgedAreLeftOut():
    """A note on a visit with no iqQaMetrics row has no sequence: it is dropped, not a crash."""
    notes = pd.DataFrame(
        {
            "pfs_visit_id": pd.array([100, 999], dtype="Int64"),
            "iic_sequence_id": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "camera": [None, None],
            "data_flag": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "note": ["Cloudy", "Not reduced"],
            "source": ["obslog", "obslog"],
        }
    )
    table = buildMetricsTable(_iq(), _thresholds(), notes=notes)
    notesKept = table[table["metric"] == NOTE]
    assert notesKept[["visit", "reason"]].values.tolist() == [[100, "Cloudy"]]
    assert set(table["visit"]) == {100, 101, 102, 200}


def testFirstThresholdsFile(tmp_path):
    table = _thresholds()
    assert firstThresholdsFile([tmp_path / "a.yaml", table]) == tmp_path / "a.yaml"
    assert firstThresholdsFile([table, "b.yaml", "c.yaml"]) == Path("b.yaml")
    with pytest.raises(ValueError):
        firstThresholdsFile([table])
