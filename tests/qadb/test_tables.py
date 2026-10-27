"""Tests for shaping the QA tasks' Butler tables as qadb rows."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.calibration import readThresholds
from pfs.drp.qa.metrics.gate import THRESHOLD_FIELDS
from pfs.drp.qa.qadb.tables import (
    GATE_ORDER,
    IMAGE_QUALITY,
    IMAGE_QUALITY_SPECIES,
    KEYS,
    LINE_RESIDUAL,
    LINE_RESIDUAL_METRIC,
    QA_GATE,
    QA_THRESHOLD_ENTRY,
    QA_THRESHOLDS,
    TASK_TABLES,
    cameraName,
    gateRows,
    imageRows,
    populationKey,
    snakeCase,
    thresholdRows,
    worstStatus,
)

NOW = datetime(2026, 10, 26, 12, 0, 0)


@pytest.mark.parametrize(
    ("camel", "snake"),
    [
        ("medFwhm", "med_fwhm"),
        ("pctLowSN", "pct_low_sn"),
        ("fitReservedXRms", "fit_reserved_x_rms"),
        ("fitXSoften", "fit_x_soften"),
        ("nLines", "n_lines"),
        ("calexpRun", "calexp_run"),
        ("medDxCenter", "med_dx_center"),
        ("already_snake", "already_snake"),
        ("value", "value"),
    ],
)
def testSnakeCase(camel, snake):
    assert snakeCase(camel) == snake
    # Negative control: lower-casing alone is not the convention.
    if camel != snake:
        assert camel.lower() != snake


def testCameraName():
    assert cameraName("b", 1) == "b1"
    assert cameraName("n", np.int64(4)) == "n4"


def testSpecsMatchTheSchema(ddlColumns):
    """Every spec's columns are exactly its qadb table's, in order (the DDL mirrors spt_qa_database)."""
    for spec in (IMAGE_QUALITY, IMAGE_QUALITY_SPECIES, LINE_RESIDUAL, LINE_RESIDUAL_METRIC):
        assert list(spec.qadbColumns) == ddlColumns[spec.name], spec.name
        assert spec.keys[:3] == KEYS
    assert list(QA_THRESHOLDS) == ddlColumns["qa_thresholds"]
    assert list(QA_THRESHOLD_ENTRY) == ddlColumns["qa_threshold_entry"]
    assert list(QA_GATE) == ddlColumns["qa_gate"]
    # Negative control: a column the schema lacks is caught.
    assert "fit_trace_x_rms" not in ddlColumns["image_quality_qa"]
    assert "fitTraceXRms" not in IMAGE_QUALITY.columns


def testTaskTablesNameTheSpecs():
    sources = {
        IMAGE_QUALITY.source,
        IMAGE_QUALITY_SPECIES.source,
        LINE_RESIDUAL.source,
        LINE_RESIDUAL_METRIC.source,
    }
    assert {table for pair in TASK_TABLES.values() for table in pair} == sources


class TestImageRows:
    def testImageQuality(self, fixtures):
        rows = imageRows(fixtures["iqQaMetrics"], IMAGE_QUALITY, 7)
        assert list(rows.columns) == list(IMAGE_QUALITY.qadbColumns)
        assert len(rows) == 4
        assert set(rows["processing_id"]) == {7}
        assert rows["pfs_visit_id"].tolist() == [133025, 133025, 133040, 133040]
        assert rows["camera_name"].tolist() == ["b1", "r1", "b1", "r1"]
        # Columns the fixture lacks are null; the ones it has are Python scalars of the declared type.
        assert rows["fit_chi2"].isna().all()
        assert all(isinstance(v, int) for v in rows["n_lit_fibers"])  # stored as float 585.0 in the fixture
        assert all(isinstance(v, float) for v in rows["med_fwhm"])
        assert all(isinstance(v, bool) for v in rows["trace_only"])
        assert set(rows["qa_status"]) <= set(GATE_ORDER)

    def testDatetime(self):
        frame = pd.DataFrame(
            {
                "visit": [1, 2, 3],
                "arm": ["b"] * 3,
                "spectrograph": [1] * 3,
                "calibCreated": ["2025-11-12", None, "n/a"],
            }
        )
        rows = imageRows(frame, LINE_RESIDUAL, 1)
        assert rows["calib_created"].tolist() == [datetime(2025, 11, 12), None, None]
        # Negative control: the header's date is not kept as text.
        assert not isinstance(rows.loc[0, "calib_created"], str)

    def testListsAreJoined(self):
        frame = pd.DataFrame({"visit": [1], "arm": ["b"], "spectrograph": [1], "qaReason": [["a", "b"]]})
        assert imageRows(frame, IMAGE_QUALITY, 1).loc[0, "qa_reason"] == "a,b"

    def testNanBecomesNull(self):
        frame = pd.DataFrame(
            {
                "visit": [1],
                "arm": ["b"],
                "spectrograph": [1],
                "medFwhm": [np.nan],
                "nLines": [np.nan],
                "qaReason": [""],
            }
        )
        rows = imageRows(frame, IMAGE_QUALITY, 1)
        assert rows.loc[0, "med_fwhm"] is None
        assert rows.loc[0, "n_lines"] is None
        assert rows.loc[0, "qa_reason"] == ""  # an empty reason is a value, not a null

    def testProcessingByRun(self, fixtures):
        frame = fixtures["iqQaMetrics"].assign(run=["r1", "r1", "r2", "r2"])
        rows = imageRows(frame, IMAGE_QUALITY, {"r1": 10, "r2": 20})
        assert rows["processing_id"].tolist() == [10, 10, 20, 20]
        with pytest.raises(ValueError, match="run"):
            imageRows(fixtures["iqQaMetrics"], IMAGE_QUALITY, {"r1": 10})

    def testLongTable(self, fixtures):
        rows = imageRows(fixtures["dmLineResidualMetrics"], LINE_RESIDUAL_METRIC, 3)
        assert list(rows.columns) == list(LINE_RESIDUAL_METRIC.qadbColumns)
        assert LINE_RESIDUAL_METRIC.keys == (*KEYS, "reference", "description", "status_type", "metric")
        assert len(rows) == 128
        assert not rows.duplicated(list(LINE_RESIDUAL_METRIC.keys)).any()
        assert set(rows["reference"]) == {"calib", "visit"}
        assert "" in set(rows["status"])  # unjudged rows keep an empty status

    def testSummary(self, fixtures):
        rows = imageRows(fixtures["dmLineResidualSummary"], LINE_RESIDUAL, 3)
        assert rows["calib_first_visit"].tolist() == [133025] * 4
        assert all(isinstance(v, datetime) for v in rows["calib_created"])
        assert all(isinstance(v, bool) for v in rows["fitted"])
        assert (
            rows["recommended_action"]
            .isin(["NOMINAL", "APPLY_SHIFT", "RECALIBRATE", "INSUFFICIENT_DATA", "UNJUDGED"])
            .all()
        )


class TestThresholdRows:
    def testIqThresholds(self, thresholdsFiles):
        metadata, table = readThresholds(thresholdsFiles["imageQualityQa"])
        header, entries = thresholdRows(metadata, table, 5, "iqQaThresholds-run25.yaml", "ab" * 32, NOW)
        assert list(header.columns) == list(QA_THRESHOLDS)
        assert header.loc[0, "thresholds_id"] == 5
        assert header.loc[0, "reference_runs"] == "25,29"
        assert header.loc[0, "derived_on"] == datetime(2026, 10, 19)
        assert header.loc[0, "drp_stella_version"] == metadata["pipelineVersions"]["drp_stella"]
        assert header.loc[0, "n_entries"] == len(table) == len(entries)
        assert list(entries.columns) == list(QA_THRESHOLD_ENTRY)
        assert not entries.duplicated(["thresholds_id", "metric", "population"]).any()
        first = entries.iloc[0]
        assert first["population"] == "arm=b obsType=arc"
        assert first["arm"] == "b" and first["obs_type"] == "arc"
        assert isinstance(first["warn"], float) and first["warn"] < first["fail"]
        # Negative control: the population has no spectrograph, so the column is null, not 0.
        assert first["spectrograph"] is None

    def testDmThresholdsKeepUnjudgedEntries(self, thresholdsFiles):
        metadata, table = readThresholds(thresholdsFiles["dmLineResiduals"])
        _, entries = thresholdRows(metadata, table, 6, "dm.yaml", "cd" * 32, NOW)
        assert len(entries) == len(table)
        assert entries["warn"].isna().any()  # "Not judged" entries have null thresholds
        assert set(entries["reference"].dropna()) <= {"calib", "visit"}
        assert "statusType" not in entries.columns and "status_type" in entries.columns

    def testPopulationKey(self):
        assert populationKey({"obsType": "arc", "arm": "b", "spectrograph": None}) == "arm=b obsType=arc"
        assert populationKey({"spectrograph": 2}) == "spectrograph=2"
        assert set(THRESHOLD_FIELDS) >= {"warn", "fail", "method", "provenance"}


class TestGateRows:
    @staticmethod
    def task(rows):
        return pd.DataFrame(
            rows, columns=["visit", "arm", "spectrograph", "qaStatus", "qaDecidedBy", "qaReason"]
        )

    def testWorstVerdictWins(self):
        iq = self.task(
            [
                (1, "b", 1, "PASS", "", ""),
                (1, "r", 1, "FAIL", "medFwhm", "medFwhm 3.5 > 3.0"),
                (2, "b", 1, "UNKNOWN", "", "no metric measured"),
                (3, "b", 1, "UNKNOWN", "", ""),
                (4, "b", 1, "WARN", "pctFlagged", "flags"),
            ]
        )
        dm = self.task(
            [
                (1, "b", 1, "WARN", "robustRmsDeltaX", "ArI scatter"),
                (1, "r", 1, "WARN", "robustRmsDeltaX", "scatter"),
                (2, "b", 1, "PASS", "", ""),
                (4, "b", 1, "FAIL", "medianDeltaX", "offset"),
            ]
        )
        gate = gateRows(
            {"imageQualityQa": iq, "dmLineResiduals": dm}, 9, "comparison", NOW, {"imageQualityQa": 1}
        )
        assert list(gate.columns) == list(QA_GATE)
        assert gate["camera_name"].tolist() == ["b1", "r1", "b1", "b1", "b1"]
        assert gate["qa_status"].tolist() == ["WARN", "FAIL", "PASS", "UNKNOWN", "FAIL"]
        assert gate["decided_by_task"].tolist() == [
            "dmLineResiduals",
            "imageQualityQa",
            "dmLineResiduals",
            "imageQualityQa",
            "dmLineResiduals",
        ]
        assert gate["qa_decided_by"].tolist() == ["robustRmsDeltaX", "medFwhm", "", "", "medianDeltaX"]
        assert gate.loc[1, "qa_reason"] == "imageQualityQa: medFwhm 3.5 > 3.0; dmLineResiduals: scatter"
        assert gate.loc[2, "qa_reason"] == "imageQualityQa: no metric measured"
        assert (gate["mode"] == "comparison").all()
        assert gate["iq_thresholds_id"].tolist() == [1] * 5
        assert gate["dm_thresholds_id"].isna().all()
        assert gate["provisional"].isna().all()  # neither task said
        assert (gate["decided_at"] == NOW).all()
        # Negative control: alphabetical order would call image r1 "WARN", not FAIL.
        assert max(["FAIL", "WARN"]) == "WARN"
        assert gate.loc[1, "qa_status"] != max(["FAIL", "WARN"])

    def testProvisionalAndRuns(self):
        iq = self.task([(1, "b", 1, "PASS", "", "")]).assign(qaProvisional=[True], run=["r1"])
        dm = self.task([(1, "b", 1, "PASS", "", ""), (1, "r", 1, "PASS", "", "")]).assign(run=["r1", "r2"])
        gate = gateRows({"imageQualityQa": iq, "dmLineResiduals": dm}, {"r1": 11, "r2": 22}, "realtime", NOW)
        assert gate["processing_id"].tolist() == [11, 22]
        assert gate["provisional"].tolist() == [True, None]

    def testThresholdsOnlyWhereTheTaskJudged(self):
        iq = self.task([(1, "b", 1, "PASS", "", "")])
        dm = self.task([(1, "b", 1, "PASS", "", ""), (1, "r", 1, "PASS", "", "")])
        ids = {"imageQualityQa": 7, "dmLineResiduals": 8}
        gate = gateRows({"imageQualityQa": iq, "dmLineResiduals": dm}, 1, "backfill", NOW, ids)
        assert gate["iq_thresholds_id"].tolist() == [7, None]  # r1 was not judged by imageQualityQa
        assert gate["dm_thresholds_id"].tolist() == [8, 8]

    def testRowWithoutVerdictIsUnknownAndDecides(self):
        iq = self.task([(1, "b", 1, None, "", ""), (2, "b", 1, np.nan, "", "")])
        gate = gateRows({"imageQualityQa": iq}, 1, "backfill", NOW)
        assert gate["qa_status"].tolist() == ["UNKNOWN", "UNKNOWN"]
        assert gate["decided_by_task"].tolist() == ["imageQualityQa"] * 2

    def testEmpty(self):
        gate = gateRows({}, 1, "backfill", NOW)
        assert gate.empty and list(gate.columns) == list(QA_GATE)

    def testWorstStatus(self):
        assert worstStatus([]) == "UNKNOWN"
        assert worstStatus(["PASS", None, np.nan]) == "PASS"
        assert worstStatus(["UNKNOWN", "PASS"]) == "PASS"
        assert worstStatus(["WARN", "FAIL", "PASS"]) == "FAIL"
