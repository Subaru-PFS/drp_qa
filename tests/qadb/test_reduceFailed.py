"""``markReduceFailed`` writes a REDUCE FAILED ``qa_gate`` verdict per exposed detector, no schema change."""

import pandas as pd

from pfs.drp.qa.qadb.reduceFailed import REDUCE_FAILED_STATUS, markReduceFailed


class FakeOpDB:
    """Answer ``readVisitCameras`` from a cameras table and ``ensureVisits`` from a visits table."""

    def __init__(self, cameras: pd.DataFrame, visits: pd.DataFrame | None = None):
        self.cameras = cameras
        self.visits = visits

    def query_dataframe(self, sql, /, *, params=None, conn=None):
        if "sps_exposure" in sql:  # readVisitCameras
            inRange = (self.cameras["visit"] >= params["first"]) & (self.cameras["visit"] <= params["last"])
            return self.cameras[inRange].reset_index(drop=True)
        assert "FROM pfs_visit" in sql, sql  # ensureVisits
        if self.visits is None:
            return pd.DataFrame(
                columns=["pfs_visit_id", "pfs_visit_description", "pfs_design_id", "issued_at"]
            )
        return self.visits[self.visits["pfs_visit_id"].isin(params["visits"])].reset_index(drop=True)


def _cameras(visit: int, names) -> pd.DataFrame:
    return pd.DataFrame({"visit": [visit] * len(names), "camera": list(names)})


def testMarksEachExposedDetector(qadb):
    opdb = FakeOpDB(_cameras(1000, ["b1", "n1", "r1"]))
    written = markReduceFailed(qadb, opdb, "u/test/qa", "run/1000", 1000, returnCode=1, status="FAILED")
    assert written == 3
    rows = qadb.query_dataframe(
        "SELECT camera_name, qa_status, qa_decided_by, qa_reason, mode FROM qa_gate ORDER BY camera_name"
    )
    assert rows["camera_name"].tolist() == ["b1", "n1", "r1"]
    assert set(rows["qa_status"]) == {REDUCE_FAILED_STATUS}
    assert set(rows["qa_decided_by"]) == {"reduceExposure"}
    assert rows["qa_reason"].iloc[0] == "reduceExposure failed (returnCode 1: FAILED)"
    assert set(rows["mode"]) == {"realtime"}
    # The data_processing_pipe2d row (for the FK) and the pfs_visit row were made.
    assert qadb.query_scalar("SELECT count(*) FROM data_processing_pipe2d WHERE run_name = 'run/1000'") == 1
    assert qadb.query_scalar("SELECT count(*) FROM pfs_visit WHERE pfs_visit_id = 1000") == 1
    # Negative control: the marker is its own status, NOT a QA FAIL, so the dashboard tells them apart.
    assert REDUCE_FAILED_STATUS != "FAIL"
    assert qadb.query_scalar("SELECT count(*) FROM qa_gate WHERE qa_status = 'FAIL'") == 0


def testIdempotentReplacesNotDuplicates(qadb):
    opdb = FakeOpDB(_cameras(1000, ["b1", "r1"]))
    assert markReduceFailed(qadb, opdb, "u/test/qa", "run/1000", 1000, returnCode=1, status="FAILED") == 2
    assert markReduceFailed(qadb, opdb, "u/test/qa", "run/1000", 1000, returnCode=1, status="FAILED") == 2
    assert qadb.query_scalar("SELECT count(*) FROM qa_gate") == 2  # replaced, not doubled


def testNoExposedCameraMarksNothing(qadb):
    opdb = FakeOpDB(_cameras(1000, []))
    assert markReduceFailed(qadb, opdb, "u/test/qa", "run/1000", 1000, returnCode=1, status="FAILED") == 0
    assert qadb.query_scalar("SELECT count(*) FROM qa_gate") == 0


def testReasonWithoutAReturnCode(qadb):
    markReduceFailed(qadb, FakeOpDB(_cameras(5, ["b1"])), "u/test/qa", "run/5", 5)
    assert qadb.query_scalar("SELECT qa_reason FROM qa_gate") == "reduceExposure failed"


def testVisitDescriptionCopiedFromOpdbWhenKnown(qadb):
    visits = pd.DataFrame(
        {
            "pfs_visit_id": [7],
            "pfs_visit_description": ["Science"],
            "pfs_design_id": [0x1234],
            "issued_at": pd.to_datetime(["2026-11-24 04:26"]),
        }
    )
    markReduceFailed(qadb, FakeOpDB(_cameras(7, ["b1"]), visits), "u/test/qa", "run/7", 7, returnCode=1)
    assert (
        qadb.query_scalar("SELECT pfs_visit_description FROM pfs_visit WHERE pfs_visit_id = 7") == "Science"
    )
