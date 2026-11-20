"""Tests for judging stored numbers again and making normal values, from qadb alone (`pfs.drp.qa.qadb.judge`).

The frames are synthetic quartzes (``quartzFrames`` of ``conftest.py``), loaded
with the loader as qaActor does: calibration frames 101 to 105, then frames 106
to 108, whose b1 widths sit 0.3 %, 5.3 % and 3.3 % above the calibration frames'
median: a PASS, a FAIL and a WARN of the width check once that median is the
normal value.
"""

from datetime import datetime

import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import gate
from pfs.drp.qa.metrics.normalValues import NORMAL_COLUMNS
from pfs.drp.qa.metrics.registry import CHECKS, MIN_NORMAL_FRAMES
from pfs.drp.qa.qadb.judge import judgeStored, makeValues, storedNumbers
from pfs.drp.qa.qadb.loader import loadCollection
from pfs.drp.qa.qadb.reader import PROCESSING_ID, readGate, readNormalValues

COLLECTION = "u/test/judge"
NOW = datetime(2026, 10, 26, 12, 0, 0)
LATER = datetime(2026, 10, 26, 13, 0, 0)
LATEST = datetime(2026, 10, 26, 14, 0, 0)
CALIBRATION = [101, 102, 103, 104, 105]
FRAMES = [106, 107, 108]

#: The b1 trace width of each frame (px); r1's is 3.30 throughout.
B1_WIDTH = {101: 3.00, 102: 3.02, 103: 3.04, 104: 3.06, 105: 3.08, 106: 3.05, 107: 3.20, 108: 3.14}
NORMAL_B1 = 3.04
WIDTH = {**{(visit, "b1"): width for visit, width in B1_WIDTH.items()}, None: 3.30}

#: The key of b1's normal width on a quartz.
B1 = ("medFwhm", "b1", "quartz", "")


def load(butler, qadb, visits, now=NOW):
    loadCollection(
        butler, qadb, COLLECTION, visits=visits, pipelineVersions=False, log=lambda _: None, now=lambda: now
    )


@pytest.fixture
def loaded(StubButler, quartzFrames, qadb):
    """Return a database with the calibration frames and the frames after them loaded, and no normal value."""
    visits = [*CALIBRATION, *FRAMES]
    load(StubButler(quartzFrames(visits, width=WIDTH)), qadb, visits)
    return qadb


def gateOf(qadb) -> pd.DataFrame:
    rows = qadb.query_dataframe(
        "SELECT g.* FROM qa_gate g JOIN data_processing_pipe2d p ON p.processing_id = g.processing_id"
        " WHERE p.collection_name = :c ORDER BY g.pfs_visit_id, g.camera_name",
        params={"c": COLLECTION},
    )
    return rows.assign(decided_at=pd.to_datetime(rows["decided_at"]))


def statusOf(qadb) -> dict:
    rows = gateOf(qadb)
    return dict(
        zip(zip(rows["pfs_visit_id"], rows["camera_name"], strict=True), rows["qa_status"], strict=True)
    )


def normalValues(qadb) -> pd.DataFrame:
    return qadb.query_dataframe(
        "SELECT * FROM qa_normal_value WHERE collection_name = :c ORDER BY valid_from, metric, camera",
        params={"c": COLLECTION},
    )


def quiet(_):
    pass


def testTheFramesAsLoaded(loaded):
    """What the tests below rest on: without normal values every frame passes, on its fixed check alone."""
    rows = gateOf(loaded)
    assert len(rows) == 16 and set(rows["qa_status"]) == {"PASS"} and set(rows["qa_decided_by"]) == {""}
    assert (rows["decided_at"] == pd.Timestamp(NOW)).all()
    assert pd.Series([B1_WIDTH[v] for v in CALIBRATION]).median() == NORMAL_B1


class TestStoredNumbers:
    def testTheGatesThreeTables(self, loaded):
        numbers = storedNumbers(loaded, COLLECTION, [(101, 108)])
        assert set(numbers) == {"iq", "lines", "twilight"}
        assert len(numbers["iq"]) == 16 and len(numbers["lines"]) == 32 and numbers["twilight"].empty
        assert all(PROCESSING_ID in table for table in numbers.values())
        assert set(gate(**numbers)["qaStatus"]) == {"PASS"}

    def testRangesAndArms(self, loaded):
        numbers = storedNumbers(loaded, COLLECTION, [(101, 102), (107, 107)])
        assert sorted(set(numbers["iq"]["visit"])) == [101, 102, 107]
        assert sorted(set(numbers["lines"]["visit"])) == [101, 102, 107]
        # Negative control: the span of the ranges is not what is read, only the ranges.
        assert 104 not in set(numbers["iq"]["visit"])
        blue = storedNumbers(loaded, COLLECTION, [(101, 108)], arms=["b"])
        assert set(blue["iq"]["arm"]) == {"b"} and set(blue["lines"]["arm"]) == {"b"} and len(blue["iq"]) == 8


class TestMakeValues:
    def testNothingIsWrittenWithoutStore(self, loaded):
        before = gateOf(loaded)
        compared, stored, refused = makeValues(loaded, COLLECTION, [(101, 105)], log=quiet, now=lambda: LATER)
        assert len(compared) == 6 and set(compared["status"]) == {"NEW"}
        assert refused.empty
        # What would be stored comes back, and none of it is in qadb.
        assert list(stored.columns) == list(NORMAL_COLUMNS) and len(stored) == 6
        assert loaded.count("qa_normal_value", "collection_name = 'u/test/judge'") == 0
        pd.testing.assert_frame_equal(gateOf(loaded), before)
        made = compared.set_index(["metric", "camera", "frameType", "selector"])
        assert made.loc[B1, "value"] == pytest.approx(NORMAL_B1)
        assert made.loc[("medFwhm", "r1", "quartz", ""), "value"] == pytest.approx(3.30)
        assert made.loc[("linesPerFiber", "b", "quartz", ""), "value"] == 4000.0  # per arm, not per camera
        assert made.loc[("robustRmsDeltaX", "b1", "quartz", "Trace"), "value"] == pytest.approx(0.04)
        assert set(compared["count"]) == {5} and set(compared["sourceVisits"]) == {"101-105"}

    def testStoringNeedsWhoAccepts(self, loaded):
        with pytest.raises(ValueError, match="who accepts"):
            makeValues(loaded, COLLECTION, [(101, 105)], store=True, log=quiet)
        assert loaded.count("qa_normal_value", "collection_name = 'u/test/judge'") == 0

    def testStoredWithWhoFromWhenAndWhy(self, loaded):
        _, stored, refused = makeValues(
            loaded, COLLECTION, [(101, 105)], store=True, acceptedBy="tester", log=quiet, now=lambda: LATER
        )
        rows = normalValues(loaded)
        assert len(rows) == len(stored) == 6 and refused.empty
        assert set(rows["accepted_by"]) == {"tester"}
        # Valid from the visit after the calibration frames: they judge what follows, not themselves.
        assert set(rows["valid_from"]) == {106}
        assert (pd.to_datetime(rows["accepted_at"]) == pd.Timestamp(LATER)).all()
        assert set(rows["reason"]) == {""}  # nothing shifted: nothing to give a reason for
        assert set(rows["n_frames"]) == {5} and set(rows["source_visits"]) == {"101-105"}
        b1 = rows[(rows["metric"] == "medFwhm") & (rows["camera"] == "b1")].iloc[0]
        assert b1["value"] == pytest.approx(NORMAL_B1)
        assert (b1["frame_type"], b1["selector"]) == ("quartz", "")
        # What was stored is what the gate reads back.
        read = readNormalValues(loaded, COLLECTION)
        assert len(read) == 6 and set(read["validFrom"]) == {106} and set(read["count"]) == {5}

    def testFramesFromValidFromOnAreJudgedAgain(self, loaded):
        assert statusOf(loaded)[(107, "b1")] == "PASS"  # before: no normal value, so no width check
        makeValues(
            loaded, COLLECTION, [(101, 105)], store=True, acceptedBy="tester", log=quiet, now=lambda: LATER
        )
        status = statusOf(loaded)
        assert [status[(visit, "b1")] for visit in FRAMES] == ["PASS", "FAIL", "WARN"]
        assert [status[(visit, "r1")] for visit in FRAMES] == ["PASS", "PASS", "PASS"]
        rows = gateOf(loaded).set_index(["pfs_visit_id", "camera_name"])
        failed = rows.loc[(107, "b1")]
        assert failed["qa_decided_by"] == "width"
        assert failed["qa_reason"] == "width 3.20 px is +5.3 % from normal 3.04 px, limit +5 %"
        assert CHECKS["width"].fail == 5.0
        judgedAgain = rows["decided_at"] == pd.Timestamp(LATER)
        assert sorted(set(rows.index[judgedAgain].get_level_values(0))) == FRAMES
        # Negative control: the frames before valid_from keep the verdict and the time of their load.
        earlier = rows[rows.index.get_level_values(0) < 106]
        assert len(earlier) == 10 and set(earlier["qa_status"]) == {"PASS"}
        assert (earlier["decided_at"] == pd.Timestamp(NOW)).all()

    def testValidFromGiven(self, loaded):
        """An explicit ``validFrom`` is where the values start to judge, and where judging again starts."""
        makeValues(
            loaded,
            COLLECTION,
            [(101, 105)],
            validFrom=104,
            store=True,
            acceptedBy="tester",
            log=quiet,
            now=lambda: LATER,
        )
        assert set(normalValues(loaded)["valid_from"]) == {104}
        rows = gateOf(loaded)
        judgedAgain = rows.loc[rows["decided_at"] == pd.Timestamp(LATER), "pfs_visit_id"]
        assert sorted(set(judgedAgain)) == [104, 105, 106, 107, 108]
        # Negative control: by default they would have started after the last calibration frame.
        assert max(CALIBRATION) + 1 == 106 != 104

    def testAShiftBeyondFailIsRefusedWithoutAReason(self, StubButler, quartzFrames, loaded):
        makeValues(
            loaded, COLLECTION, [(101, 105)], store=True, acceptedBy="tester", log=quiet, now=lambda: LATER
        )
        # A second calibration set, b1 now 8 % wider (beyond the width check's FAIL), and a frame after it.
        second = [201, 202, 203, 204, 205, 206]
        wider = round(NORMAL_B1 * 1.08, 4)
        load(StubButler(quartzFrames(second, width={"b1": wider, None: 3.30})), loaded, second)
        assert statusOf(loaded)[(206, "b1")] == "FAIL"  # against the first set's value

        compared, stored, refused = makeValues(
            loaded, COLLECTION, [(201, 205)], store=True, acceptedBy="tester", log=quiet, now=lambda: LATEST
        )
        made = compared.set_index(["metric", "camera", "frameType", "selector"])
        assert made.loc[B1, "status"] == "FAIL" and made.loc[B1, "shift"] == pytest.approx(8.0, abs=0.01)
        assert made.loc[B1, "previous"] == pytest.approx(NORMAL_B1)
        assert (made.drop(index=[B1])["status"] == "PASS").all()
        assert len(refused) == 1 and tuple(refused.iloc[0][["metric", "camera"]]) == ("medFwhm", "b1")
        rows = normalValues(loaded)
        new = rows[rows["valid_from"] == 206]
        assert len(new) == len(stored) == 5
        assert not ((new["metric"] == "medFwhm") & (new["camera"] == "b1")).any()
        # The value in force stays for the one refused: the frame after still fails against it.
        assert statusOf(loaded)[(206, "b1")] == "FAIL"

        # With a reason the shifted value is stored, the reason with it, and the frame is judged by it.
        why = "refocused after the warm-up"
        _, stored, refused = makeValues(
            loaded,
            COLLECTION,
            [(201, 205)],
            store=True,
            acceptedBy="tester",
            reason=why,
            log=quiet,
            now=lambda: LATEST,
        )
        assert refused.empty and len(stored) == 6
        new = normalValues(loaded)
        new = new[new["valid_from"] == 206].set_index(["metric", "camera"])
        assert len(new) == 6
        assert new.loc[("medFwhm", "b1"), "reason"] == why
        assert new.loc[("medFwhm", "b1"), "value"] == pytest.approx(wider)
        # Negative control: the values that did not shift carry no reason.
        assert set(new.drop(index=[("medFwhm", "b1")])["reason"]) == {""}
        assert statusOf(loaded)[(206, "b1")] == "PASS"
        # And the first set's values still judge the frames before the second's.
        assert statusOf(loaded)[(107, "b1")] == "FAIL"

    def testStoringTheSameKeyAgainReplaces(self, loaded):
        arguments = {"store": True, "log": quiet}
        makeValues(loaded, COLLECTION, [(101, 105)], acceptedBy="first", now=lambda: LATER, **arguments)
        ids = set(normalValues(loaded)["normal_value_id"])
        # The same frames and valid_from, by another name and from fewer frames.
        makeValues(
            loaded,
            COLLECTION,
            [(101, 104)],
            validFrom=106,
            acceptedBy="second",
            now=lambda: LATEST,
            **arguments,
        )
        rows = normalValues(loaded)
        assert len(rows) == 6 and set(rows["accepted_by"]) == {"second"} and set(rows["n_frames"]) == {4}
        assert set(rows["normal_value_id"]) == ids  # the rows themselves, replaced
        b1 = rows[(rows["metric"] == "medFwhm") & (rows["camera"] == "b1")].iloc[0]
        assert b1["value"] == pytest.approx(3.03)  # the median of 101 to 104
        # Negative control: another valid_from is another set of rows, and the first stays.
        makeValues(
            loaded,
            COLLECTION,
            [(101, 105)],
            validFrom=108,
            acceptedBy="third",
            now=lambda: LATEST,
            **arguments,
        )
        rows = normalValues(loaded)
        assert len(rows) == 12 and set(rows.loc[rows["valid_from"] == 106, "accepted_by"]) == {"second"}

    def testArmsRestricts(self, loaded):
        compared, stored, _ = makeValues(
            loaded, COLLECTION, [(101, 105)], arms=["b"], store=True, acceptedBy="tester", log=quiet
        )
        assert set(compared["camera"]) == {"b", "b1"} and len(compared) == 3
        rows = normalValues(loaded)
        assert len(rows) == len(stored) == 3 and set(rows["camera"]) == {"b", "b1"}
        # Negative control: without it the r arm's values are made too.
        everyArm, _, _ = makeValues(loaded, COLLECTION, [(101, 105)], log=quiet)
        assert set(everyArm["camera"]) == {"b", "b1", "r", "r1"}

    def testSeveralRanges(self, loaded):
        compared, _, _ = makeValues(
            loaded, COLLECTION, [(101, 102), (104, 105)], store=True, acceptedBy="tester", log=quiet
        )
        assert set(compared["count"]) == {4} and set(compared["sourceVisits"]) == {"101-102,104-105"}
        # Valid from the visit after the last of them.
        assert set(normalValues(loaded)["valid_from"]) == {106}

    def testAFrameFailingAFixedCheckIsLeftOut(self, StubButler, quartzFrames, qadb):
        """A calibration frame 2.5 px from the calibration is not one to take a normal value from."""
        width = {**WIDTH, (103, "b1"): 9.0}
        frames = quartzFrames(CALIBRATION, width=width, offset={(103, "b1"): 2.5, None: 0.0})
        load(StubButler(frames), qadb, CALIBRATION)
        assert statusOf(qadb)[(103, "b1")] == "FAIL"
        messages = []
        compared, _, _ = makeValues(qadb, COLLECTION, [(101, 105)], log=messages.append)
        made = compared.set_index(["metric", "camera", "frameType", "selector"])
        assert made.loc[B1, "count"] == 4 and made.loc[B1, "sourceVisits"] == "101-102,104-105"
        assert made.loc[B1, "value"] == pytest.approx(3.04)  # the median of 3.00, 3.02, 3.06 and 3.08
        assert "1 images fail a fixed check and are left out of the normal values" in messages
        # Negative control: the same visit's r1 image, which passes, is counted.
        assert made.loc[("medFwhm", "r1", "quartz", ""), "count"] == 5

    def testTheMArmOffsetIsStoredAndJudgesWith(self, StubButler, quartzFrames, qadb):
        """A camera's offset when read with the m arm has no frame type and no selector, and still judges.

        The b arm sits 0.30 px off in x on the visits read with m; a science frame read with m is judged
        on its offset less that.
        """
        withM = quartzFrames([301, 302, 303], cameras=("b1",), cameraSet="bmn", offset=0.30)
        withR = quartzFrames([304, 305, 306], cameras=("b1",), offset=0.0)
        science = [
            quartzFrames([310], cameras=("b1",), cameraSet="bmn", offset=0.70, obsType="science"),
            quartzFrames([311], cameras=("b1",), offset=0.70, obsType="science"),
            quartzFrames([312], cameras=("b1",), cameraSet="bmn", offset=0.95, obsType="science"),
        ]
        tables = {name: pd.concat([part[name] for part in (withM, withR, *science)]) for name in withM}
        load(StubButler(tables), qadb, [301, 302, 303, 304, 305, 306, 310, 311, 312])
        # Before: 0.70 px is beyond the sky frames' WARN of 0.5 px, read with m or not.
        status = statusOf(qadb)
        assert (status[(310, "b1")], status[(311, "b1")], status[(312, "b1")]) == ("WARN", "WARN", "WARN")

        compared, _, _ = makeValues(
            qadb, COLLECTION, [(301, 306)], store=True, acceptedBy="tester", log=quiet, now=lambda: LATER
        )
        offset = compared[compared["metric"] == "mArmOffsetX"].iloc[0]
        assert offset["value"] == pytest.approx(0.30) and offset["count"] == 3
        stored = normalValues(qadb)
        row = stored[stored["metric"] == "mArmOffsetX"].iloc[0]
        assert (row["camera"], row["frame_type"], row["selector"]) == ("b1", "", "")
        assert row["value"] == pytest.approx(0.30) and row["valid_from"] == 307

        rows = gateOf(qadb).set_index("pfs_visit_id")
        assert rows.loc[310, "qa_status"] == "PASS"  # 0.70 - 0.30 = 0.40 px
        assert rows.loc[312, "qa_status"] == "WARN"  # 0.95 - 0.30 = 0.65 px
        assert (
            rows.loc[312, "qa_reason"] == "x offset +0.65 px, limit 0.5 px (less the m-arm offset +0.30 px)"
        )
        # Negative control: the frame read with r has nothing subtracted, and still warns at 0.70 px.
        assert rows.loc[311, "qa_status"] == "WARN"
        assert rows.loc[311, "qa_reason"] == "x offset +0.70 px, limit 0.5 px"

    def testTooFewFramesAreNotStored(self, loaded):
        messages = []
        compared, stored, refused = makeValues(
            loaded, COLLECTION, [(101, 102)], store=True, acceptedBy="tester", log=messages.append
        )
        assert MIN_NORMAL_FRAMES == 3
        assert set(compared["status"]) == {"TOO_FEW"} and stored.empty and len(refused) == 6
        assert loaded.count("qa_normal_value", "collection_name = 'u/test/judge'") == 0
        assert not any("judged again" in m for m in messages)  # nothing stored, nothing to judge again


class TestJudgeStored:
    @pytest.fixture
    def judged(self, loaded, storeValues):
        """Return the database with b1's normal width stored, valid from 106, and the frames not judged by it yet."""
        storeValues(loaded, COLLECTION, {B1: NORMAL_B1}, validFrom=106)
        return loaded

    def testVerdictsFollowTheNormalValues(self, judged, storeValues):
        # The value alone changes no verdict: the gate rows are as loaded until they are judged again.
        assert statusOf(judged)[(107, "b1")] == "PASS"
        verdicts = judgeStored(judged, COLLECTION, 106, log=quiet, now=lambda: LATER)
        status = statusOf(judged)
        assert [status[(visit, "b1")] for visit in FRAMES] == ["PASS", "FAIL", "WARN"]
        assert len(verdicts) == 6 and list(verdicts["qaStatus"]) == [
            status[(row.visit, f"{row.arm}{row.spectrograph}")] for row in verdicts.itertuples()
        ]
        # A wider normal value in qadb: the same frames, judged again, now pass.
        storeValues(judged, COLLECTION, {B1: 3.20}, validFrom=106)
        assert statusOf(judged)[(107, "b1")] == "FAIL"  # negative control: not until judged again
        judgeStored(judged, COLLECTION, 106, log=quiet, now=lambda: LATEST)
        status = statusOf(judged)
        assert [status[(visit, "b1")] for visit in FRAMES] == ["PASS", "PASS", "PASS"]
        # A narrower one: every one of them is beyond FAIL.
        storeValues(judged, COLLECTION, {B1: 2.80}, validFrom=106)
        judgeStored(judged, COLLECTION, 106, log=quiet, now=lambda: LATEST)
        status = statusOf(judged)
        assert [status[(visit, "b1")] for visit in FRAMES] == ["FAIL", "FAIL", "FAIL"]
        assert [status[(visit, "r1")] for visit in FRAMES] == ["PASS", "PASS", "PASS"]

    def testTheVerdictsAreTheGates(self, judged):
        verdicts = judgeStored(judged, COLLECTION, 101, log=quiet, now=lambda: LATER)
        numbers = storedNumbers(judged, COLLECTION, [(101, 108)])
        expected = gate(**numbers, normalValues=readNormalValues(judged, COLLECTION))
        pd.testing.assert_frame_equal(verdicts, expected)
        stored = readGate(judged, COLLECTION)
        for column in ("frameType", "qaStatus", "qaDecidedBy", "qaReason"):
            assert stored[column].tolist() == expected[column].tolist(), column
        assert set(stored["mode"]) == {"backfill"}

    def testDryRunWritesNothing(self, judged):
        before = gateOf(judged)
        messages = []
        verdicts = judgeStored(judged, COLLECTION, 106, dryRun=True, log=messages.append, now=lambda: LATER)
        # The verdicts it would write come back: the FAIL is there.
        key = (verdicts["visit"] == 107) & (verdicts["arm"] == "b")
        assert verdicts.loc[key, "qaStatus"].tolist() == ["FAIL"]
        assert messages == ["6 images judged, nothing written: 4 PASS, 1 WARN, 1 FAIL"]
        pd.testing.assert_frame_equal(gateOf(judged), before)
        assert statusOf(judged)[(107, "b1")] == "PASS"
        # Negative control: without dryRun the same call does write it.
        judgeStored(judged, COLLECTION, 106, log=quiet, now=lambda: LATER)
        assert statusOf(judged)[(107, "b1")] == "FAIL"

    def testTheGateRowKeepsItsProcessing(self, StubButler, quartzFrames, qadb, storeValues):
        """Each image's row is replaced under the processing its numbers were loaded under."""
        runs = {(visit, arm, 1): "u/test/judge/second" for visit in (107, 108) for arm in "br"}
        load(StubButler(quartzFrames(FRAMES, width=WIDTH), runs=runs), qadb, FRAMES)
        storeValues(qadb, COLLECTION, {B1: NORMAL_B1}, validFrom=106)
        before = gateOf(qadb)
        judgeStored(qadb, COLLECTION, 106, log=quiet, now=lambda: LATER)
        after = gateOf(qadb)
        assert len(after) == len(before) == 6  # replaced: no second row for any image
        assert after["processing_id"].tolist() == before["processing_id"].tolist()
        assert (after["decided_at"] == pd.Timestamp(LATER)).all()
        assert after["qa_status"].tolist() != before["qa_status"].tolist()
        numbers = qadb.query_dataframe(
            "SELECT processing_id, pfs_visit_id, camera_name FROM image_quality_qa"
            " ORDER BY pfs_visit_id, camera_name"
        )
        assert after["processing_id"].tolist() == numbers["processing_id"].tolist()
        # Negative control: the frames are under two processings, so one id for all would have shown.
        assert after["processing_id"].nunique() == 2
        assert after.groupby("pfs_visit_id")["processing_id"].nunique().tolist() == [1, 1, 1]

    def testOnlyTheRangeIsJudged(self, judged):
        judgeStored(judged, COLLECTION, 107, 107, log=quiet, now=lambda: LATER)
        rows = gateOf(judged)
        again = rows["decided_at"] == pd.Timestamp(LATER)
        assert set(rows.loc[again, "pfs_visit_id"]) == {107} and again.sum() == 2
        status = statusOf(judged)
        assert status[(107, "b1")] == "FAIL"
        # Negative control: 108, outside the range, would warn and has not been judged.
        assert status[(108, "b1")] == "PASS"

    def testNoNumbers(self, judged):
        messages = []
        verdicts = judgeStored(judged, COLLECTION, 900, 999, log=messages.append)
        assert verdicts.empty and messages == [f"no numbers for visits 900 to 999 in {COLLECTION}"]
        assert judgeStored(judged, "u/test/typo", 101, log=quiet).empty

    def testModeIsRecorded(self, judged):
        judgeStored(judged, COLLECTION, 106, 106, mode="comparison", log=quiet, now=lambda: LATER)
        rows = gateOf(judged)
        assert set(rows.loc[rows["pfs_visit_id"] == 106, "mode"]) == {"comparison"}
        assert set(rows.loc[rows["pfs_visit_id"] != 106, "mode"]) == {"backfill"}

    def testTwilightsAreJudgedFromTheirStoredMetrics(self, twilight, qadb):
        """A twilight's verdict comes back from ``twilight_qa`` alone: no Butler, no opdb."""
        loadCollection(
            twilight.butler(),
            qadb,
            COLLECTION,
            visits=twilight.VISITS,
            opdb=twilight.OpDB(),
            readSequence=twilight.Sequences(),
            pipelineVersions=False,
            log=quiet,
            now=lambda: NOW,
        )
        atLoad = gateOf(qadb)
        assert set(atLoad["qa_status"]) == {"PASS", "FAIL", "UNKNOWN"}
        qadb.commit(
            "UPDATE qa_gate SET qa_status = 'UNKNOWN', qa_decided_by = '', qa_reason = 'lost'"
            " WHERE pfs_visit_id BETWEEN 501 AND 504"
        )
        assert set(gateOf(qadb)["qa_reason"]) == {"lost"}
        judgeStored(qadb, COLLECTION, 501, 504, log=quiet, now=lambda: LATER)
        again = gateOf(qadb)
        for column in ("processing_id", "frame_type", "qa_status", "qa_decided_by", "qa_reason"):
            assert again[column].tolist() == atLoad[column].tolist(), column
        failed = again[again["qa_status"] == "FAIL"]
        assert failed[["pfs_visit_id", "camera_name"]].to_numpy().tolist() == [[503, "r2"]]
