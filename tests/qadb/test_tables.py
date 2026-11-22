"""Tests for shaping the QA numbers, the gate's verdicts and the normal values as qadb rows."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import NORMAL_KEYS, STATUS_ORDER, VERDICT_COLUMNS, gate
from pfs.drp.qa.metrics.normalValues import NORMAL_COLUMNS
from pfs.drp.qa.qadb.tables import (
    IMAGE_QUALITY,
    IMAGE_QUALITY_SPECIES,
    KEYS,
    LINE_RESIDUAL,
    LINE_RESIDUAL_METRIC,
    NORMAL_VALUE,
    NORMAL_VALUE_KEYS,
    QA_GATE,
    TWILIGHT_QA,
    cameraName,
    gateRows,
    imageRows,
    normalValueFrame,
    normalValueRows,
    snakeCase,
)

NOW = datetime(2026, 10, 26, 12, 0, 0)
SPECS = (IMAGE_QUALITY, IMAGE_QUALITY_SPECIES, LINE_RESIDUAL, LINE_RESIDUAL_METRIC, TWILIGHT_QA)

#: Columns a numbers table had when each task gave its own verdict.
VERDICTS = {"qa_status", "qa_decided_by", "qa_reason", "recommended_action", "status", "provisional"}


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
        ("nSaturatedLines", "n_saturated_lines"),
        ("pctSaturatedPixels", "pct_saturated_pixels"),
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
    """Every spec's columns are exactly its qadb table's, in order."""
    for spec in SPECS:
        assert list(spec.qadbColumns) == ddlColumns[spec.name], spec.name
        assert spec.keys[:3] == KEYS
    assert list(QA_GATE) == ddlColumns["qa_gate"]
    # The normal values have a surrogate key the code never writes: the database numbers the rows.
    assert ddlColumns["qa_normal_value"] == ["normal_value_id", *NORMAL_VALUE]
    assert set(NORMAL_VALUE_KEYS) <= set(NORMAL_VALUE)
    # Negative control: a column the schema lacks is caught.
    assert "fit_trace_x_rms" not in ddlColumns["image_quality_qa"]
    assert "fitTraceXRms" not in IMAGE_QUALITY.columns


def testNoNumbersTableHoldsAVerdict(ddlColumns):
    """The tasks store numbers only: the verdict is ``qa_gate``'s, and nowhere else."""
    for spec in SPECS:
        assert not VERDICTS & set(spec.qadbColumns), spec.name
        assert not VERDICTS & set(ddlColumns[spec.name]), spec.name
    assert {"qa_status", "qa_decided_by", "qa_reason", "frame_type"} <= set(QA_GATE)
    # What went with the thresholds files is gone from the gate row and from the schema.
    assert not {"decided_by_task", "provisional"} & set(QA_GATE)
    assert not {"qa_thresholds", "qa_threshold_entry", "qa_processing_thresholds"} & set(ddlColumns)
    # Negative control: the check finds a verdict column where there is one.
    assert VERDICTS & set(QA_GATE)


def testSaturationIsStored():
    """The saturation numbers follow ``nTraceRows``, as the schema has them."""
    columns = list(IMAGE_QUALITY.columns)
    at = columns.index("nTraceRows")
    assert columns[at + 1 : at + 3] == ["nSaturatedLines", "pctSaturatedPixels"]
    assert IMAGE_QUALITY.columns["nSaturatedLines"] == "int"
    assert IMAGE_QUALITY.columns["pctSaturatedPixels"] == "float"


class TestImageRows:
    def testImageQuality(self, fixtures):
        rows = imageRows(fixtures["iqQaMetrics"], IMAGE_QUALITY, 7)
        assert list(rows.columns) == list(IMAGE_QUALITY.qadbColumns)
        assert len(rows) == 4
        assert set(rows["processing_id"]) == {7}
        assert rows["pfs_visit_id"].tolist() == [133025, 133025, 133040, 133040]
        assert rows["camera_name"].tolist() == ["b1", "r1", "b1", "r1"]
        # Columns the fixture lacks are null; the ones it has are Python scalars of the declared type.
        assert rows["pct_meas_fail"].isna().all()
        assert all(isinstance(v, int) for v in rows["n_lit_fibers"])
        assert all(isinstance(v, float) for v in rows["med_fwhm"])
        assert all(isinstance(v, bool) for v in rows["trace_only"])

    def testAVerdictColumnInTheButlerTableIsNotCopied(self, fixtures):
        """A table written when the tasks still judged carries ``qaStatus``; none of it reaches qadb."""
        old = fixtures["iqQaMetrics"].assign(qaStatus="FAIL", qaDecidedBy="medFwhm", qaReason="then")
        rows = imageRows(old, IMAGE_QUALITY, 7)
        assert list(rows.columns) == list(IMAGE_QUALITY.qadbColumns)
        pd.testing.assert_frame_equal(rows, imageRows(fixtures["iqQaMetrics"], IMAGE_QUALITY, 7))

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
        frame = pd.DataFrame({"visit": [1], "arm": ["b"], "spectrograph": [1], "seqName": [["a", "b"]]})
        assert imageRows(frame, IMAGE_QUALITY, 1).loc[0, "seq_name"] == "a,b"

    def testNanBecomesNull(self):
        frame = pd.DataFrame(
            {
                "visit": [1],
                "arm": ["b"],
                "spectrograph": [1],
                "medFwhm": [np.nan],
                "nLines": [np.nan],
                "seqName": [""],
            }
        )
        rows = imageRows(frame, IMAGE_QUALITY, 1)
        assert rows.loc[0, "med_fwhm"] is None
        assert rows.loc[0, "n_lines"] is None
        assert rows.loc[0, "seq_name"] == ""  # an empty name is a value, not a null

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
        assert all(isinstance(v, float) for v in rows["value"])

    def testSummary(self, fixtures):
        rows = imageRows(fixtures["dmLineResidualSummary"], LINE_RESIDUAL, 3)
        assert rows["calib_first_visit"].tolist() == [133025] * 4
        assert all(isinstance(v, datetime) for v in rows["calib_created"])
        assert all(isinstance(v, bool) for v in rows["fitted"])
        assert set(rows["camera_set"]) == {"brn"} and set(rows["obs_type"]) == {"arc", "trace"}

    def testTwilightSequenceIsRenamed(self):
        """The metrics' ``sequence`` is qadb's ``iic_sequence_id``; an infinite level offset is a null."""
        frame = pd.DataFrame(
            {
                "visit": [5, 5],
                "arm": ["b", "r"],
                "spectrograph": [1, 1],
                "sequence": [54787, 54787],
                "tooFaint": [False, True],
                "twilightLevelOffset": [np.inf, 0.4],
            }
        )
        rows = imageRows(frame, TWILIGHT_QA, 7)
        assert list(rows.columns) == list(TWILIGHT_QA.qadbColumns)
        assert "iic_sequence_id" in rows and "sequence" not in rows
        assert rows["iic_sequence_id"].tolist() == [54787, 54787]
        assert rows["twilight_level_offset"].tolist() == [None, 0.4]
        assert rows["too_faint"].tolist() == [False, True]
        # Negative control: the value really was infinite, not already missing.
        assert np.isinf(frame.loc[0, "twilightLevelOffset"])


class TestGateRows:
    @staticmethod
    def verdicts(rows):
        return pd.DataFrame(rows, columns=list(VERDICT_COLUMNS))

    def testOneRowPerImageAsTheGateGaveIt(self):
        verdicts = self.verdicts(
            [
                (1, "b", 1, "arc", "PASS", "", ""),
                (1, "r", 1, "arc", "FAIL", "width", "width 3.50 px is +16.7 % from normal 3.00 px"),
                (2, "b", 1, "quartz", "WARN", "calibOffset", "x offset +1.20 px, limit 1 px"),
                (3, "n", 2, None, "UNKNOWN", "", "no frame type"),
            ]
        )
        rows = gateRows(verdicts, 9, "comparison", NOW)
        assert list(rows.columns) == list(QA_GATE)
        assert rows["processing_id"].tolist() == [9] * 4
        assert rows["pfs_visit_id"].tolist() == [1, 1, 2, 3]
        assert rows["camera_name"].tolist() == ["b1", "r1", "b1", "n2"]
        assert rows["frame_type"].tolist() == ["arc", "arc", "quartz", None]
        assert rows["qa_status"].tolist() == ["PASS", "FAIL", "WARN", "UNKNOWN"]
        assert rows["qa_decided_by"].tolist() == ["", "width", "calibOffset", ""]
        assert rows.loc[1, "qa_reason"] == "width 3.50 px is +16.7 % from normal 3.00 px"
        assert rows.loc[3, "qa_reason"] == "no frame type"
        assert (rows["mode"] == "comparison").all() and (rows["decided_at"] == NOW).all()
        assert all(type(v) is int for v in rows["pfs_visit_id"])
        # Negative control: the camera is arm then spectrograph, not the reverse.
        assert rows.loc[3, "camera_name"] != "2n"

    def testProcessingByRun(self):
        verdicts = self.verdicts(
            [(1, "b", 1, "arc", "PASS", "", ""), (1, "r", 1, "arc", "PASS", "", "")]
        ).assign(run=["r1", "r2"])
        rows = gateRows(verdicts, {"r1": 11, "r2": 22}, "realtime", NOW)
        assert rows["processing_id"].tolist() == [11, 22]
        with pytest.raises(ValueError, match="run"):
            gateRows(verdicts.drop(columns="run"), {"r1": 11}, "realtime", NOW)

    def testTheGatesOwnTable(self, fixtures):
        """What `pfs.drp.qa.metrics.gate.gate` returns is what `gateRows` takes, column for column."""
        verdicts = gate(iq=fixtures["iqQaMetrics"], lines=fixtures["dmLineResidualMetrics"])
        assert list(verdicts.columns) == list(VERDICT_COLUMNS)
        rows = gateRows(verdicts, 1, "backfill", NOW)
        assert rows["frame_type"].tolist() == ["arc", "arc", "quartz", "quartz"]
        assert rows["qa_status"].tolist() == verdicts["qaStatus"].tolist()
        assert set(rows["qa_status"]) <= set(STATUS_ORDER)

    def testEmpty(self):
        rows = gateRows(gate(), 1, "backfill", NOW)
        assert rows.empty and list(rows.columns) == list(QA_GATE)


class TestNormalValueRows:
    @staticmethod
    def values(**changes):
        row = {
            "metric": "medFwhm",
            "camera": "b1",
            "frameType": "quartz",
            "selector": "",
            "value": 3.2,
            "count": 5,
            "collection": "u/test/qa",
            "validFrom": 133042,
            "sourceVisits": "133025-133041",
            "acceptedBy": "tester",
            "acceptedAt": pd.Timestamp(NOW),
            "reason": "",
        }
        return pd.DataFrame([row | changes], columns=list(NORMAL_COLUMNS))

    def testColumnsAndTypes(self):
        rows = normalValueRows(self.values())
        assert list(rows.columns) == list(NORMAL_VALUE)
        row = rows.iloc[0]
        assert (row["collection_name"], row["metric"], row["camera"]) == ("u/test/qa", "medFwhm", "b1")
        assert (row["frame_type"], row["selector"]) == ("quartz", "")
        assert type(row["valid_from"]) is int and row["valid_from"] == 133042
        assert type(row["n_frames"]) is int and row["n_frames"] == 5
        assert type(row["value"]) is float and row["value"] == 3.2
        assert row["accepted_at"] == NOW and type(row["accepted_at"]) is datetime
        assert (row["source_visits"], row["accepted_by"], row["reason"]) == ("133025-133041", "tester", "")

    def testAMissingFrameTypeOrSelectorIsEmptyNotNull(self):
        """Both are part of the key: an m-arm offset has neither, and a null would never conflict."""
        rows = normalValueRows(self.values(metric="mArmOffsetX", frameType=None, selector=np.nan))
        assert rows.loc[0, "frame_type"] == "" and rows.loc[0, "selector"] == ""
        # Negative control: a value that may be missing stays null.
        assert normalValueRows(self.values(reason=None)).loc[0, "reason"] is None

    def testRoundTrip(self):
        """`normalValueFrame` is the inverse: qadb's rows come back under the gate's names."""
        values = pd.concat(
            [self.values(), self.values(metric="linesPerFiber", camera="b", selector="Neon", value=40.0)],
            ignore_index=True,
        )
        frame = normalValueFrame(normalValueRows(values))
        assert list(frame.columns) == list(NORMAL_VALUE.values())
        assert set(frame.columns) == set(NORMAL_COLUMNS)
        for column in ("metric", "camera", "frameType", "selector", "value", "count", "validFrom"):
            assert frame[column].tolist() == values[column].tolist(), column
        # What the gate looks a value up by is there under its own names.
        assert set(NORMAL_KEYS) | {"value", "count", "validFrom"} <= set(frame.columns)
        # Negative control: qadb's own names are not the gate's.
        assert "n_frames" not in frame and "count" not in normalValueRows(values)

    def testKeysNameTheUniqueConstraint(self):
        assert NORMAL_VALUE_KEYS == (
            "collection_name",
            "metric",
            "camera",
            "frame_type",
            "selector",
            "valid_from",
        )
        # The gate's key, plus whose values they are and from when.
        assert [NORMAL_VALUE[column] for column in NORMAL_VALUE_KEYS] == [
            "collection",
            *NORMAL_KEYS,
            "validFrom",
        ]
