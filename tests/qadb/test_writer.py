"""Tests for the idempotent qadb writes, on SQLite (and PostgreSQL when PFS_QADB_TEST_DSN is set)."""

import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import sqlalchemy.exc

from pfs.drp.qa.metrics.calibration import readThresholds
from pfs.drp.qa.qadb.tables import IMAGE_QUALITY, imageRows
from pfs.drp.qa.qadb.writer import (
    ensureVisits,
    processingId,
    records,
    registerThresholds,
    sha256Of,
    upsert,
    upsertSql,
)

NOW = datetime(2026, 10, 26, 12, 0, 0)


class TestUpsertSql:
    def testUpdatesNonKeyColumns(self):
        sql = upsertSql(
            "t",
            ["processing_id", "pfs_visit_id", "camera_name", "med_fwhm", "qa_status"],
            ["processing_id", "pfs_visit_id", "camera_name"],
        )
        assert sql.startswith(
            "INSERT INTO t (processing_id, pfs_visit_id, camera_name, med_fwhm, qa_status) VALUES (:processing_id,"
        )
        assert "ON CONFLICT (processing_id, pfs_visit_id, camera_name) DO UPDATE SET" in sql
        assert "med_fwhm = EXCLUDED.med_fwhm, qa_status = EXCLUDED.qa_status" in sql
        # Negative control: a key is never updated.
        assert "processing_id = EXCLUDED" not in sql

    def testKeysOnlyDoesNothing(self):
        assert upsertSql("pfs_visit", ["pfs_visit_id"], ["pfs_visit_id"]).endswith(
            "ON CONFLICT (pfs_visit_id) DO NOTHING"
        )

    def testKeyMustBeAColumn(self):
        with pytest.raises(ValueError, match="camera_name"):
            upsertSql("t", ["processing_id"], ["processing_id", "camera_name"])


def testRecordsAreNative():
    frame = pd.DataFrame(
        {
            "a": np.array([1, 2], dtype=np.int64),
            "b": [np.nan, 2.5],
            "c": [pd.Timestamp("2026-10-26"), pd.NaT],
            "d": ["x", None],
        }
    )
    rows = records(frame)
    assert rows == [
        {"a": 1, "b": None, "c": datetime(2026, 10, 26), "d": "x"},
        {"a": 2, "b": 2.5, "c": None, "d": None},
    ]
    assert type(rows[0]["a"]) is int


class TestProcessingId:
    def testSameRunSameId(self, qadb, scope):
        first = processingId(
            qadb, "u/test/qa", "run1", "w.2026.44", {"drp_stella": "w.2026.41", "datamodel": "w.2026.40"}
        )
        again = processingId(qadb, "u/test/qa", "run1", "w.2026.45", {"drp_stella": "w.2026.42"})
        other = processingId(qadb, "u/test/qa", "run2", "w.2026.44", None)
        assert first == again != other
        assert qadb.count("data_processing_pipe2d", scope["data_processing_pipe2d"]) == 2
        row = qadb.query_dataframe(
            "SELECT * FROM data_processing_pipe2d WHERE processing_id = :id", params={"id": first}
        ).iloc[0]
        # The versions follow the latest load; a product not given is null.
        assert row["drp_qa_version"] == "w.2026.45"
        assert row["pipe2d_version"] == "w.2026.42"
        assert pd.isna(row["datamodel_version"])
        assert pd.isna(row["obs_pfs_version"])


class TestUpsert:
    def testLoadingTwiceReplaces(self, qadb, fixtures):
        pid = processingId(qadb, "u/test/writer", "r", "v")
        ensureVisits(qadb, [133025, 133040])
        rows = imageRows(fixtures["iqQaMetrics"], IMAGE_QUALITY, pid)
        assert upsert(qadb, "image_quality_qa", rows, list(IMAGE_QUALITY.keys)) == 4
        assert upsert(qadb, "image_quality_qa", rows, list(IMAGE_QUALITY.keys)) == 4
        assert qadb.count("image_quality_qa") == 4
        changed = rows.copy()
        changed["med_fwhm"] = [9.0] * 4
        upsert(qadb, "image_quality_qa", changed, list(IMAGE_QUALITY.keys))
        assert qadb.count("image_quality_qa") == 4
        assert qadb.query_scalars("SELECT med_fwhm FROM image_quality_qa") == [9.0] * 4
        # Negative control: without ON CONFLICT the same rows are a key violation.
        plain = upsertSql("image_quality_qa", list(rows.columns), list(IMAGE_QUALITY.keys)).split(
            " ON CONFLICT"
        )[0]
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            qadb.commit(plain, params=records(rows))

    def testEmptyWritesNothing(self, qadb, scope):
        assert upsert(qadb, "pfs_visit", pd.DataFrame(columns=["pfs_visit_id"]), ["pfs_visit_id"]) == 0
        assert qadb.count("pfs_visit", scope["pfs_visit"]) == 0

    def testForeignKeyNeedsTheImage(self, qadb):
        """A species row without its image's row is refused (the schema's foreign key)."""
        pid = processingId(qadb, "u/test/writer", "r", "v")
        ensureVisits(qadb, [1])
        orphan = pd.DataFrame(
            [
                {
                    "processing_id": pid,
                    "pfs_visit_id": 1,
                    "camera_name": "b1",
                    "description": "ArI",
                    "metric": "fitXRms",
                    "value": 0.1,
                }
            ]
        )
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            upsert(
                qadb,
                "image_quality_qa_species",
                orphan,
                ["processing_id", "pfs_visit_id", "camera_name", "description", "metric"],
            )


class FakeOpDB:
    """Answer the pfs_visit query from a table."""

    def __init__(self, visits):
        self.visits = visits

    def query_dataframe(self, sql, /, *, params=None, conn=None):
        assert "FROM pfs_visit" in sql
        return self.visits[self.visits["pfs_visit_id"].isin(params["visits"])].reset_index(drop=True)


class TestEnsureVisits:
    def testIdsOnly(self, qadb):
        assert ensureVisits(qadb, [2, 1, 1]) == 2
        assert ensureVisits(qadb, [1, 3]) == 2
        assert qadb.query_scalars(
            "SELECT pfs_visit_id FROM pfs_visit WHERE pfs_visit_id < 1000 ORDER BY 1"
        ) == [1, 2, 3]
        assert ensureVisits(qadb, []) == 0

    def testFromOpdb(self, qadb):
        ensureVisits(qadb, [1])
        opdb = FakeOpDB(
            pd.DataFrame(
                {
                    "pfs_visit_id": [1, 2],
                    "pfs_visit_description": ["Arc: Argon", "Trace"],
                    "pfs_design_id": [0x1234, 0x5678],
                    "issued_at": pd.to_datetime(["2026-10-01 20:00", "2026-10-01 20:05"]),
                }
            )
        )
        assert ensureVisits(qadb, [1, 2, 3], opdb) == 3
        table = qadb.query_dataframe(
            "SELECT * FROM pfs_visit WHERE pfs_visit_id < 1000 ORDER BY pfs_visit_id"
        )
        assert table["pfs_visit_description"].tolist()[:2] == ["Arc: Argon", "Trace"]
        assert pd.isna(table.loc[2, "pfs_visit_description"])  # unknown to the opdb: id alone
        assert int(table.loc[0, "pfs_design_id"]) == 0x1234


class TestRegisterThresholds:
    def testSameContentSameId(self, qadb, thresholdsFiles, tmp_path):
        path = thresholdsFiles["imageQualityQa"]
        first = registerThresholds(qadb, path, NOW)
        copy = tmp_path / "renamed.yaml"
        shutil.copy(path, copy)
        again = registerThresholds(qadb, copy, datetime(2027, 1, 1))
        assert first == again
        assert qadb.count("qa_thresholds") == 1
        row = qadb.query_dataframe("SELECT * FROM qa_thresholds").iloc[0]
        assert row["name"] == "renamed.yaml"  # the latest name
        assert pd.Timestamp(row["registered_at"]) == pd.Timestamp(NOW)  # the first registration
        assert row["sha256"] == sha256Of(path)
        _, table = readThresholds(path)
        assert qadb.count("qa_threshold_entry") == len(table) == row["n_entries"]

    def testDifferentContentDifferentId(self, qadb, thresholdsFiles):
        iq = registerThresholds(qadb, thresholdsFiles["imageQualityQa"], NOW)
        dm = registerThresholds(qadb, thresholdsFiles["dmLineResiduals"], NOW)
        assert iq != dm
        assert qadb.count("qa_thresholds") == 2
        entries = qadb.query_dataframe("SELECT thresholds_id, metric, reference FROM qa_threshold_entry")
        assert set(entries.loc[entries["thresholds_id"] == dm, "reference"].dropna()) <= {"calib", "visit"}
        assert entries.loc[entries["thresholds_id"] == iq, "reference"].isna().all()
