"""Tests for the idempotent qadb writes, on SQLite (and PostgreSQL when PFS_QADB_TEST_DSN is set)."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import sqlalchemy.exc

from pfs.drp.qa.metrics.normalValues import NORMAL_COLUMNS
from pfs.drp.qa.qadb import writer
from pfs.drp.qa.qadb.tables import (
    IMAGE_QUALITY,
    NORMAL_VALUE,
    NORMAL_VALUE_KEYS,
    QA_GATE,
    gateRows,
    imageRows,
    normalValueRows,
)
from pfs.drp.qa.qadb.writer import (
    UNKNOWN_VERSION,
    deleteRows,
    ensureVisits,
    processingId,
    records,
    upsert,
    upsertSql,
)

NOW = datetime(2026, 10, 26, 12, 0, 0)


class TestUpsertSql:
    def testUpdatesNonKeyColumns(self):
        sql = upsertSql(
            "t",
            ["processing_id", "pfs_visit_id", "camera_name", "med_fwhm", "n_lines"],
            ["processing_id", "pfs_visit_id", "camera_name"],
        )
        assert sql.startswith(
            "INSERT INTO t (processing_id, pfs_visit_id, camera_name, med_fwhm, n_lines) VALUES (:processing_id,"
        )
        assert "ON CONFLICT (processing_id, pfs_visit_id, camera_name) DO UPDATE SET" in sql
        assert "med_fwhm = EXCLUDED.med_fwhm, n_lines = EXCLUDED.n_lines" in sql
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
        # A version given replaces; one not given keeps what was recorded.
        assert row["drp_qa_version"] == "w.2026.45"
        assert row["pipe2d_version"] == "w.2026.42"
        assert row["datamodel_version"] == "w.2026.40"
        assert pd.isna(row["obs_pfs_version"])

    def testUnknownVersionsKeepTheRecorded(self, qadb):
        """A later load that read no versions (qaActor, pipelineVersions=False) must not null them."""
        first = processingId(qadb, "u/test/qa", "run1", "w.2026.44", {"drp_stella": "w.2026.41"})
        assert processingId(qadb, "u/test/qa", "run1", UNKNOWN_VERSION, None) == first
        row = qadb.query_dataframe(
            "SELECT * FROM data_processing_pipe2d WHERE processing_id = :id", params={"id": first}
        ).iloc[0]
        assert row["drp_qa_version"] == "w.2026.44"
        assert row["pipe2d_version"] == "w.2026.41"
        # Negative control: a plain overwrite would have nulled them.
        assert row["pipe2d_version"] is not None and not pd.isna(row["pipe2d_version"])


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

    def testDeleteRows(self, qadb, fixtures):
        pid = processingId(qadb, "u/test/writer", "r", "v")
        ensureVisits(qadb, [133025, 133040])
        rows = imageRows(fixtures["iqQaMetrics"], IMAGE_QUALITY, pid)
        upsert(qadb, "image_quality_qa", rows, list(IMAGE_QUALITY.keys))
        keys = rows[["processing_id", "pfs_visit_id", "camera_name"]].iloc[:2]
        assert deleteRows(qadb, "image_quality_qa", pd.concat([keys, keys])) == 2  # duplicates deleted once
        assert qadb.count("image_quality_qa") == 2
        assert deleteRows(qadb, "image_quality_qa", keys.iloc[:0]) == 0

    def testOneTransaction(self, qadb, fixtures):
        """A delete and an upsert on one connection commit together."""
        pid = processingId(qadb, "u/test/writer", "r", "v")
        ensureVisits(qadb, [133025, 133040])
        rows = imageRows(fixtures["iqQaMetrics"], IMAGE_QUALITY, pid)
        upsert(qadb, "image_quality_qa", rows, list(IMAGE_QUALITY.keys))
        with qadb.connection() as conn:
            deleteRows(
                qadb, "image_quality_qa", rows[["processing_id", "pfs_visit_id", "camera_name"]], conn=conn
            )
            upsert(qadb, "image_quality_qa", rows.iloc[:1], list(IMAGE_QUALITY.keys), conn=conn)
        assert qadb.count("image_quality_qa") == 1

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


def testNoThresholdsAreRegistered():
    """The limits are the registry's and the normal values a table: the writer registers no thresholds file."""
    assert not {"registerThresholds", "sha256Of"} & set(dir(writer))
    assert not any("threshold" in name.lower() for name in writer.__all__)
    # Negative control: the writes the loader does use are still there.
    assert {"upsert", "deleteRows", "ensureVisits", "processingId"} <= set(writer.__all__)


class TestGateRow:
    """``qa_gate`` holds one row per image and processing: judging again replaces it."""

    @staticmethod
    def verdict(status, decidedBy="", reason=""):
        return pd.DataFrame(
            {
                "visit": [7],
                "arm": ["b"],
                "spectrograph": [1],
                "frameType": ["arc"],
                "qaStatus": [status],
                "qaDecidedBy": [decidedBy],
                "qaReason": [reason],
            }
        )

    def testJudgingAgainReplacesTheRow(self, qadb):
        pid = processingId(qadb, "u/test/writer", "r", "v")
        ensureVisits(qadb, [7])
        keys = list(QA_GATE[:3])
        upsert(qadb, "qa_gate", gateRows(self.verdict("PASS"), pid, "realtime", NOW), keys)
        later = datetime(2026, 10, 27)
        upsert(
            qadb, "qa_gate", gateRows(self.verdict("FAIL", "width", "too wide"), pid, "backfill", later), keys
        )
        gate = qadb.query_dataframe("SELECT * FROM qa_gate WHERE pfs_visit_id = 7")
        assert len(gate) == 1
        row = gate.iloc[0]
        assert (row["qa_status"], row["qa_decided_by"], row["qa_reason"]) == ("FAIL", "width", "too wide")
        assert (row["frame_type"], row["mode"]) == ("arc", "backfill")
        assert pd.Timestamp(row["decided_at"]) == pd.Timestamp(later)
        # Negative control: another processing of the same image is another row, not a replacement.
        other = processingId(qadb, "u/test/writer", "r2", "v")
        upsert(qadb, "qa_gate", gateRows(self.verdict("PASS"), other, "backfill", NOW), keys)
        assert qadb.count("qa_gate", "pfs_visit_id = 7") == 2


class TestNormalValues:
    """``qa_normal_value``: storing the same key again replaces the row; another ``valid_from`` adds one."""

    @staticmethod
    def values(value, validFrom=100, selector="Neon", reason=""):
        row = {
            "metric": "linesPerFiber",
            "camera": "b",
            "frameType": "arc",
            "selector": selector,
            "value": value,
            "count": 4,
            "collection": "u/test/writer",
            "validFrom": validFrom,
            "sourceVisits": "90-99",
            "acceptedBy": "tester",
            "acceptedAt": NOW,
            "reason": reason,
        }
        return normalValueRows(pd.DataFrame([row], columns=list(NORMAL_COLUMNS)))

    def stored(self, qadb):
        return qadb.query_dataframe(
            "SELECT * FROM qa_normal_value WHERE collection_name = 'u/test/writer' ORDER BY valid_from, selector"
        )

    def testSameKeyReplaces(self, qadb):
        keys = list(NORMAL_VALUE_KEYS)
        assert upsert(qadb, "qa_normal_value", self.values(40.0), keys) == 1
        assert upsert(qadb, "qa_normal_value", self.values(44.0, reason="lamp changed"), keys) == 1
        stored = self.stored(qadb)
        assert len(stored) == 1
        assert stored.loc[0, "value"] == 44.0 and stored.loc[0, "reason"] == "lamp changed"
        assert set(NORMAL_VALUE) < set(stored.columns)  # and the id the database gave
        assert stored["normal_value_id"].notna().all()
        # Negative control: a later valid_from, or another lamp, is another value, and the first stays.
        upsert(qadb, "qa_normal_value", self.values(50.0, validFrom=200), keys)
        upsert(qadb, "qa_normal_value", self.values(20.0, selector="Krypton"), keys)
        stored = self.stored(qadb)
        assert stored["value"].tolist() == [20.0, 44.0, 50.0]
        assert stored["normal_value_id"].is_unique

    def testAnEmptySelectorIsPartOfTheKey(self, qadb):
        """A value with no lamp or species is stored with an empty selector, so storing it again conflicts."""
        keys = list(NORMAL_VALUE_KEYS)
        upsert(qadb, "qa_normal_value", self.values(3.1, selector=None), keys)
        upsert(qadb, "qa_normal_value", self.values(3.2, selector=None), keys)
        stored = self.stored(qadb)
        assert stored["value"].tolist() == [3.2] and stored["selector"].tolist() == [""]
        # Negative control: a null selector never conflicts, in SQLite or PostgreSQL, and would add a row;
        # the schema refuses one.
        null = self.values(3.3, selector=None).assign(selector=None)
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            upsert(qadb, "qa_normal_value", null, keys)
