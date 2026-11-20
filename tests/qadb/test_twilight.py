"""Tests for twilights in the load: measured with their sequence once it is complete, and judged by the gate.

A twilight exposure is judged against the other exposures of its sequence, so
a visit loaded while its sequence runs has numbers and no verdict; the load of
the last visit measures the whole sequence, stores it in ``twilight_qa`` and
judges every visit of it. The sequence is synthetic (the ``twilight`` fixture
of ``conftest.py``: four exposures, ``twilightFiberFlux`` rows as
``twilightFluxQa`` stores them, one detector 10 % low in one exposure), the
Butler and the opdb are stubs, and the database is the ``qadb`` fixture.
"""

import dataclasses
import re
from datetime import datetime

import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import gate
from pfs.drp.qa.metrics.registry import CHECKS
from pfs.drp.qa.metrics.twilight import METRIC_COLUMNS, sequenceConsistency
from pfs.drp.qa.qadb.judge import judgeStored
from pfs.drp.qa.qadb.loader import TWILIGHT_FLUX, loadCollection, loadVisit, twilightMetrics
from pfs.drp.qa.qadb.reader import PROCESSING_ID, readGate
from pfs.drp.qa.qadb.tables import TWILIGHT_QA

COLLECTION = "u/test/twilight"
RUN = "u/test/qa/20261026T000000Z"
VISITS = [501, 502, 503, 504]
SEQUENCE = 54787
NOW = datetime(2026, 10, 26, 5, 0, 0)
LATER = datetime(2026, 10, 26, 5, 20, 0)
LATEST = datetime(2026, 10, 27, 9, 0, 0)
IMAGE = ["visit", "arm", "spectrograph"]


def load(butler, qadb, visits=VISITS, opdb=None, readSequence=None, now=NOW, **kwargs):
    """Load some visits; return the counts and what was logged."""
    messages = []
    counts = loadCollection(
        butler,
        qadb,
        COLLECTION,
        visits=visits,
        opdb=opdb,
        readSequence=readSequence,
        pipelineVersions=False,
        log=messages.append,
        now=lambda: now,
        **kwargs,
    )
    return counts, messages


def measure(twilight, butler, visits, sequences=None):
    """Run `twilightMetrics` on some visits; return the metrics and what was logged."""
    messages = []
    metrics = twilightMetrics(
        butler,
        twilight.OpDB(),
        COLLECTION,
        visits,
        log=messages.append,
        readSequence=twilight.Sequences() if sequences is None else sequences,
    )
    return metrics, messages


def gateOf(qadb) -> pd.DataFrame:
    return qadb.query_dataframe("SELECT * FROM qa_gate ORDER BY pfs_visit_id, camera_name")


@pytest.fixture
def expected(twilight):
    """Return the gate's verdicts on the whole sequence, measured from its per-fiber table."""
    metrics = sequenceConsistency(twilight.fibers().assign(sequence=twilight.SEQUENCE))
    verdicts = gate(iq=twilight.iq(), twilight=metrics)
    return verdicts.assign(camera=verdicts["arm"] + verdicts["spectrograph"].astype(str))


def testTheSyntheticSequenceHasOneFault(twilight, expected):
    """What the tests below rest on: the step fails its detector in its exposure, and nothing else does."""
    assert (twilight.VISITS, twilight.SEQUENCE) == (VISITS, SEQUENCE)
    failed = expected[expected["qaStatus"] == "FAIL"]
    assert failed[IMAGE].to_numpy().tolist() == [list(twilight.STEP[:3])] == [[503, "r", 2]]
    assert failed["qaDecidedBy"].tolist() == ["twilightLevel"]
    assert re.fullmatch(r"level offset -(9|10)\.\d\d %, limit 3 %", failed["qaReason"].iloc[0])
    assert set(expected.loc[expected["arm"] != "n", "qaStatus"]) == {"PASS", "FAIL"}
    # The n arm is measured and not judged.
    assert set(expected.loc[expected["arm"] == "n", "qaStatus"]) == {"UNKNOWN"}
    assert (expected["frameType"] == "twilight").all()


class TestTwilightMetrics:
    """`twilightMetrics`: the complete sequences of some twilight visits, measured."""

    def testCompleteSequenceIsMeasured(self, twilight):
        metrics, messages = measure(twilight, twilight.butler(), [504])
        assert len(metrics) == len(VISITS) * len(twilight.ARMS) * len(twilight.SPECTROGRAPHS) == 48
        assert set(metrics["visit"]) == set(VISITS)  # every visit of the sequence, not only the one asked for
        assert set(metrics.columns) == set(METRIC_COLUMNS) | {"run"}
        assert set(metrics["run"]) == {RUN}
        assert any(f"sequence {SEQUENCE}: 4 visits, all stored" in m for m in messages)
        # The sequence is the opdb's, not the header's.
        assert set(metrics["sequence"]) == {SEQUENCE}
        # Negative control: the per-fiber table said -1, as a header without W_SEQID gives.
        assert set(twilight.fibers()["sequence"]) == {-1}

    def testOneLookupPerSequence(self, twilight):
        """The visits of a sequence already found are not asked about again."""
        sequences = twilight.Sequences()
        metrics, _ = measure(twilight, twilight.butler(), VISITS, sequences)
        assert sequences.calls == [501]
        assert not metrics.duplicated(IMAGE).any() and len(metrics) == 48
        # Negative control: a visit the sequence found does not hold is asked about.
        sequences = twilight.Sequences(twilight.sequence(visits=(501, 502, 503)))
        measure(twilight, twilight.butler(), VISITS, sequences)
        assert sequences.calls == [501, 504]

    @pytest.mark.parametrize(
        ("answer", "why", "asksTheButler"),
        [
            ({"visits": ()}, "in no IIC sequence", False),
            ({"name": "Arc: HgCd"}, "is not a twilight", False),
            ({"finished": None}, "has not finished", True),
        ],
    )
    def testNotReady(self, twilight, answer, why, asksTheButler):
        butler, read = twilight.butler(), []
        get = butler.get
        butler.get = lambda ref, **kwargs: read.append(ref) or get(ref, **kwargs)
        metrics, messages = measure(twilight, butler, [504], twilight.Sequences(twilight.sequence(**answer)))
        assert metrics.empty and list(metrics.columns) == [*IMAGE, "run"]
        assert any(why in m for m in messages)
        # What is not a twilight is settled from the opdb alone; no per-fiber table is read for any of them.
        assert bool(butler.queries) == asksTheButler
        assert read == []

    def testWaitsForAVisitsTable(self, twilight):
        """A finished sequence with a reduced visit whose ``twilightFiberFlux`` is not stored yet is not measured."""
        metrics, messages = measure(twilight, twilight.butler(measured=VISITS[:3]), [503])
        assert metrics.empty
        assert any(f"sequence {SEQUENCE} is waiting for 504 (no tables yet)" in m for m in messages)
        # A visit the drpActor has not reduced either is said to be that.
        metrics, messages = measure(twilight, twilight.butler(measured=VISITS[:3], reduced=VISITS[:3]), [503])
        assert metrics.empty and any("is waiting for 504 (not reduced)" in m for m in messages)

    def testWaitsForACamerasTable(self, twilight):
        fibers = twilight.fibers()
        missing = (fibers["visit"] == 502) & (fibers["arm"] == "b") & (fibers["spectrograph"] == 3)
        metrics, messages = measure(twilight, twilight.butler(flux=fibers[~missing]), [504])
        assert metrics.empty
        assert any("is waiting for 502 (b3)" in m for m in messages)
        # Negative control: with that camera's table the same sequence is measured.
        assert len(measure(twilight, twilight.butler(flux=fibers), [504])[0]) == 48


class TestLoad:
    def testSequenceNotFinished(self, twilight, qadb):
        """A twilight loaded while its sequence runs has its numbers and no verdict."""
        sequences = twilight.Sequences(twilight.sequence(finished=None))
        counts, messages = load(
            twilight.butler(measured=[501]), qadb, [501], opdb=twilight.OpDB(), readSequence=sequences
        )
        assert sequences.calls == [501]
        assert counts["image_quality_qa"] == 12 and counts["qa_gate"] == 12
        assert "twilight_qa" not in counts and qadb.count("twilight_qa") == 0
        rows = gateOf(qadb)
        assert set(rows["qa_status"]) == {"UNKNOWN"} and set(rows["frame_type"]) == {"twilight"}
        assert set(rows["qa_reason"]) == {"nothing to judge was measured"}
        assert any(f"sequence {SEQUENCE} has not finished" in m for m in messages)

    def testMissingAVisitsFlux(self, twilight, qadb):
        """The sequence has finished, but a visit's ``twilightFiberFlux`` is not stored: still no verdict."""
        butler = twilight.butler(measured=VISITS[:3])
        counts, messages = load(
            butler, qadb, VISITS[:3], opdb=twilight.OpDB(), readSequence=twilight.Sequences()
        )
        assert counts["image_quality_qa"] == 36 and "twilight_qa" not in counts
        assert qadb.count("twilight_qa") == 0
        assert set(gateOf(qadb)["qa_status"]) == {"UNKNOWN"}
        assert any("is waiting for 504 (no tables yet)" in m for m in messages)

    def testTheLastVisitJudgesTheWholeSequence(self, twilight, qadb, expected):
        """The earlier visits' ``UNKNOWN`` rows are replaced when the last visit's load measures the sequence."""
        running = twilight.Sequences(twilight.sequence(finished=None))
        for index, visit in enumerate(VISITS[:3]):
            butler = twilight.butler(measured=VISITS[: index + 1], reduced=VISITS[: index + 1])
            load(butler, qadb, [visit], opdb=twilight.OpDB(), readSequence=running)
        before = gateOf(qadb)
        assert len(before) == 36 and set(before["qa_status"]) == {"UNKNOWN"}
        assert qadb.count("twilight_qa") == 0

        counts = loadVisit(
            twilight.butler(),
            qadb,
            COLLECTION,
            504,
            opdb=twilight.OpDB(),
            readSequence=twilight.Sequences(),
            pipelineVersions=False,
            log=lambda _: None,
            now=lambda: LATER,
        )
        # Numbers of the visit loaded; twilight metrics and verdicts of the whole sequence.
        assert counts["image_quality_qa"] == 12
        assert counts["twilight_qa"] == 48 == qadb.count("twilight_qa")
        assert counts["qa_gate"] == 48 and counts["pfs_visit"] == 4
        assert (
            qadb.query_scalars("SELECT DISTINCT pfs_visit_id FROM twilight_qa ORDER BY pfs_visit_id")
            == VISITS
        )
        assert set(qadb.query_scalars("SELECT DISTINCT iic_sequence_id FROM twilight_qa")) == {SEQUENCE}

        after = gateOf(qadb)
        # Replaced, not added to: still one row per image.
        assert len(after) == 48 and not after.duplicated(["pfs_visit_id", "camera_name"]).any()
        assert after["pfs_visit_id"].tolist() == expected["visit"].tolist()
        assert after["camera_name"].tolist() == expected["camera"].tolist()
        assert after["qa_status"].tolist() == expected["qaStatus"].tolist()
        assert after["qa_decided_by"].tolist() == expected["qaDecidedBy"].tolist()
        assert after["qa_reason"].tolist() == expected["qaReason"].tolist()
        assert set(after["frame_type"]) == {"twilight"}
        assert set(after["mode"]) == {"realtime"}
        assert (pd.to_datetime(after["decided_at"]) == pd.Timestamp(LATER)).all()
        earlier = after[(after["pfs_visit_id"] < 504) & ~after["camera_name"].str.startswith("n")]
        assert set(earlier["qa_status"]) == {"PASS", "FAIL"}
        failed = after[after["qa_status"] == "FAIL"]
        assert failed[["pfs_visit_id", "camera_name", "qa_decided_by"]].to_numpy().tolist() == [
            [503, "r2", "twilightLevel"]
        ]
        # Negative control: before the last visit those very rows were UNKNOWN, decided earlier.
        assert (pd.to_datetime(before["decided_at"]) == pd.Timestamp(NOW)).all()
        assert before.set_index(["pfs_visit_id", "camera_name"]).loc[(503, "r2"), "qa_status"] == "UNKNOWN"

    def testTwilightRowsHoldNumbersOnly(self, twilight, qadb):
        load(twilight.butler(), qadb, opdb=twilight.OpDB(), readSequence=twilight.Sequences())
        stored = qadb.query_dataframe("SELECT * FROM twilight_qa ORDER BY pfs_visit_id, camera_name")
        assert list(stored.columns) == list(TWILIGHT_QA.qadbColumns)
        assert not {"qa_status", "qa_decided_by", "qa_reason"} & set(stored.columns)
        step = stored[(stored["pfs_visit_id"] == 503) & (stored["camera_name"] == "r2")].iloc[0]
        assert step["twilight_level_offset"] == pytest.approx(-10.0, abs=0.5)
        assert step["n_exposures"] == 4 and step["seq_name"] == "Twilight sky"
        # Negative control: the same detector in another exposure sits at the sequence's level.
        other = stored[(stored["pfs_visit_id"] == 502) & (stored["camera_name"] == "r2")].iloc[0]
        assert abs(other["twilight_level_offset"]) < 0.5

    def testWithoutAnOpdbNoTwilightIsJudged(self, twilight, qadb):
        sequences = twilight.Sequences()
        butler = twilight.butler()
        counts, messages = load(butler, qadb, readSequence=sequences)
        assert sequences.calls == []  # no opdb to ask
        assert TWILIGHT_FLUX not in {datasetType for datasetType, _ in butler.queries}
        assert "twilight_qa" not in counts and qadb.count("twilight_qa") == 0
        assert counts["qa_gate"] == 48 and set(gateOf(qadb)["qa_status"]) == {"UNKNOWN"}
        assert "twilight: 4 twilight visit(s) not judged: no opdb to find their sequence in" in messages
        # Negative control: the same load with an opdb judges them.
        load(twilight.butler(), qadb, opdb=twilight.OpDB(), readSequence=twilight.Sequences())
        assert set(gateOf(qadb)["qa_status"]) == {"PASS", "FAIL", "UNKNOWN"}

    def testANonTwilightVisitNeverAsksForASequence(self, twilight, StubButler, fixtures, qadb):
        """An arc and a quartz with an opdb: descriptions are copied, and no sequence is looked up."""
        sequences, opdb = twilight.Sequences(), twilight.OpDB()
        butler = StubButler(fixtures)
        counts, _ = load(butler, qadb, [133025, 133040], opdb=opdb, readSequence=sequences)
        assert sequences.calls == [] and len(opdb.queries) == 1  # `ensureVisits`' query alone
        assert "twilight_qa" not in counts
        assert TWILIGHT_FLUX not in {datasetType for datasetType, _ in butler.queries}
        # Negative control: a twilight in the same load is asked about, and only it.
        both = twilight.butler()
        both.tables.update(
            iqQaMetrics=pd.concat([twilight.iq(), fixtures["iqQaMetrics"]], ignore_index=True),
            dmLineResidualSummary=fixtures["dmLineResidualSummary"],
            dmLineResidualMetrics=fixtures["dmLineResidualMetrics"],
        )
        counts, _ = load(both, qadb, [*VISITS, 133025, 133040], opdb=twilight.OpDB(), readSequence=sequences)
        assert sequences.calls == [501]
        assert counts["twilight_qa"] == 48 and counts["qa_gate"] == 52

    def testATwilightTheOpdbCallsSomethingElseIsNotJudged(self, twilight, qadb):
        """The opdb and the header must agree on what a twilight is: the sequence's type and name decide."""
        counts, messages = load(
            twilight.butler(),
            qadb,
            opdb=twilight.OpDB(),
            readSequence=twilight.Sequences(twilight.sequence(name="Sky flat")),
        )
        assert "twilight_qa" not in counts and set(gateOf(qadb)["qa_status"]) == {"UNKNOWN"}
        assert any("is not a twilight" in m for m in messages)

    def testLoadingTwiceChangesNothing(self, twilight, qadb):
        first, _ = load(twilight.butler(), qadb, opdb=twilight.OpDB(), readSequence=twilight.Sequences())
        rows = gateOf(qadb)
        second, _ = load(twilight.butler(), qadb, opdb=twilight.OpDB(), readSequence=twilight.Sequences())
        assert first == second
        assert qadb.count("twilight_qa") == first["twilight_qa"] == 48
        pd.testing.assert_frame_equal(gateOf(qadb), rows)

    def testTooFewExposuresIsNotMeasured(self, twilight, qadb):
        """A finished sequence of two exposures has no reference: nothing is measured, nothing judged."""
        butler = twilight.butler(measured=VISITS[:2], reduced=VISITS[:2])
        sequences = twilight.Sequences(twilight.sequence(visits=VISITS[:2]))
        counts, messages = load(butler, qadb, VISITS[:2], opdb=twilight.OpDB(), readSequence=sequences)
        assert "twilight_qa" not in counts and set(gateOf(qadb)["qa_status"]) == {"UNKNOWN"}
        assert any("2 visits, all stored" in m for m in messages)  # ready, and still too short to measure

    def testDryRun(self, twilight, tmp_path, expected):
        counts, _ = load(
            twilight.butler(), None, opdb=twilight.OpDB(), readSequence=twilight.Sequences(), dryRun=tmp_path
        )
        assert counts["twilight_qa"] == 48 and (tmp_path / "twilight_qa.parquet").exists()
        stored = pd.read_parquet(tmp_path / "qa_gate.parquet")
        assert stored["qa_status"].tolist() == expected["qaStatus"].tolist()

    def testTheGateRowSharesTheReductionsProcessing(self, twilight, qadb):
        """The twilight's verdict replaces the per-visit load's row because both sit under the visit's RUN."""
        load(twilight.butler(), qadb, opdb=twilight.OpDB(), readSequence=twilight.Sequences())
        gateIds = set(qadb.query_scalars("SELECT DISTINCT processing_id FROM qa_gate"))
        assert gateIds == set(qadb.query_scalars("SELECT DISTINCT processing_id FROM twilight_qa"))
        assert gateIds == set(qadb.query_scalars("SELECT DISTINCT processing_id FROM image_quality_qa"))
        assert len(gateIds) == 1


class TestFluxInAnotherRun:
    """``twilightFluxQa`` run alone into its own RUN: the fiber fluxes are not in ``iqQaMetrics``' RUN.

    Each visit is loaded as it is reduced, so its gate row is under ``iqQaMetrics``' processing. The last
    visit's load judges the earlier visits from tables of the other RUN: their verdicts must replace the
    rows they have, not add rows under the fluxes' processing.
    """

    FLUX_RUN = "u/test/twilight/flux"

    def butler(self, twilight, measured=VISITS):
        runs = {
            (TWILIGHT_FLUX, visit, arm, spectrograph): self.FLUX_RUN
            for visit in VISITS
            for arm in twilight.ARMS
            for spectrograph in twilight.SPECTROGRAPHS
        }
        return twilight.butler(measured=measured, runs=runs)

    @pytest.fixture
    def judged(self, twilight, qadb):
        """Return the database after the per-visit loads of the sequence, the last of which judges it."""
        running = twilight.Sequences(twilight.sequence(finished=None))
        for index, visit in enumerate(VISITS[:3]):
            butler = self.butler(twilight, measured=VISITS[: index + 1])
            load(butler, qadb, [visit], opdb=twilight.OpDB(), readSequence=running)
        load(
            self.butler(twilight),
            qadb,
            [504],
            opdb=twilight.OpDB(),
            readSequence=twilight.Sequences(),
            now=LATER,
        )
        return qadb

    def processings(self, qadb) -> pd.Series:
        return qadb.query_dataframe(
            "SELECT processing_id, run_name FROM data_processing_pipe2d WHERE collection_name = :c",
            params={"c": COLLECTION},
        ).set_index("run_name")["processing_id"]

    def testOneGateRowPerImage(self, judged, expected):
        ids = self.processings(judged)
        assert set(ids.index) == {RUN, self.FLUX_RUN}
        rows = gateOf(judged)
        # One row per image: 48, where a row added under the fluxes' processing for each image of the three
        # earlier visits would make 84.
        assert len(rows) == 48 == judged.count("qa_gate")
        assert rows.groupby(["pfs_visit_id", "camera_name"]).size().max() == 1
        assert set(rows["processing_id"]) == {ids[RUN]}
        # The rows are the judged ones, the earlier visits' included, decided at the last visit's load.
        assert rows["qa_status"].tolist() == expected["qaStatus"].tolist()
        assert rows["qa_reason"].tolist() == expected["qaReason"].tolist()
        assert (pd.to_datetime(rows["decided_at"]) == pd.Timestamp(LATER)).all()
        # Negative control: the numbers they were judged from do sit under the other RUN's processing, so the
        # gate rows did not land under the reduction's by sharing a RUN with them.
        assert set(judged.query_scalars("SELECT DISTINCT processing_id FROM twilight_qa")) == {
            ids[self.FLUX_RUN]
        }
        assert ids[self.FLUX_RUN] != ids[RUN]

    def testReadersTakeTheJudgedVerdicts(self, judged, expected):
        read = readGate(judged, COLLECTION)
        assert len(read) == 48
        assert read["qaStatus"].tolist() == expected["qaStatus"].tolist()
        assert read["qaDecidedBy"].tolist() == expected["qaDecidedBy"].tolist()
        assert set(read["run"]) == {RUN} and set(read[PROCESSING_ID]) == {self.processings(judged)[RUN]}
        failed = read[read["qaStatus"] == "FAIL"]
        assert failed[IMAGE].to_numpy().tolist() == [[503, "r", 2]]
        # Negative control: not the rows of the per-visit loads, which were all UNKNOWN.
        assert set(read.loc[read["arm"] != "n", "qaStatus"]) == {"PASS", "FAIL"}

    def testJudgingAgainChangesWhatReadersTake(self, judged, monkeypatch):
        """A limit changes; `judgeStored` replaces the very row `readGate` returns, for the earlier visits too."""
        key = (503, "r2")

        def read():
            rows = readGate(judged, COLLECTION)
            return rows.assign(camera=rows["arm"] + rows["spectrograph"].astype(str)).set_index(
                ["visit", "camera"]
            )

        before = read()
        assert before.loc[key, "qaStatus"] == "FAIL" and before.loc[key, "qaDecidedBy"] == "twilightLevel"
        # The level check now allows the 10 % step.
        loose = dataclasses.replace(CHECKS["twilightLevel"], warn=20.0, fail=30.0)
        monkeypatch.setitem(CHECKS, "twilightLevel", loose)
        # The limit alone changes nothing stored: the row is stale until judged again.
        assert read().loc[key, "qaStatus"] == "FAIL"

        verdicts = judgeStored(judged, COLLECTION, 501, 504, log=lambda _: None, now=lambda: LATEST)
        assert set(verdicts["qaStatus"]) == {"PASS", "UNKNOWN"}
        after = read()
        # What readers take is the verdict judged again, not the stale one.
        assert after.loc[key, "qaStatus"] == "PASS" and after.loc[key, "qaDecidedBy"] == ""
        assert "FAIL" not in set(after["qaStatus"])
        assert (after["decidedAt"] == pd.Timestamp(LATEST)).all()
        # Replaced in place: still one row per image, each under the processing it had.
        rows = gateOf(judged)
        assert len(rows) == 48 and rows.groupby(["pfs_visit_id", "camera_name"]).size().max() == 1
        assert after[PROCESSING_ID].tolist() == before[PROCESSING_ID].tolist()
        assert set(rows["qa_status"]) == {"PASS", "UNKNOWN"}

    def testJudgingAgainReplacesTheLatestOfTwoRows(self, judged, monkeypatch):
        """An image left with a gate row under each processing: judging again replaces the one readers take."""
        ids = self.processings(judged)
        # As a load that put the earlier visits' verdicts under the fluxes' processing left it: the rows of the
        # per-visit loads under the reduction's, UNKNOWN, and the judged ones beside them.
        judged.commit(
            "INSERT INTO qa_gate (processing_id, pfs_visit_id, camera_name, frame_type, qa_status,"
            " qa_decided_by, qa_reason, mode, decided_at)"
            " SELECT :flux, pfs_visit_id, camera_name, frame_type, qa_status, qa_decided_by, qa_reason,"
            " mode, decided_at FROM qa_gate WHERE processing_id = :iq AND pfs_visit_id < 504",
            params={"flux": int(ids[self.FLUX_RUN]), "iq": int(ids[RUN])},
        )
        judged.commit(
            "UPDATE qa_gate SET qa_status = 'UNKNOWN', qa_decided_by = '', qa_reason = 'as first loaded'"
            " WHERE processing_id = :iq AND pfs_visit_id < 504",
            params={"iq": int(ids[RUN])},
        )
        assert judged.count("qa_gate") == 84
        latest = max(ids[RUN], ids[self.FLUX_RUN])
        assert latest == ids[self.FLUX_RUN]  # registered at the last visit's load, so the later one
        stale = readGate(judged, COLLECTION)
        isStep = (stale["visit"] == 503) & (stale["arm"] == "r") & (stale["spectrograph"] == 2)
        assert stale.loc[isStep, "qaStatus"].tolist() == ["FAIL"]
        assert stale.loc[isStep, PROCESSING_ID].tolist() == [latest]

        loose = dataclasses.replace(CHECKS["twilightLevel"], warn=20.0, fail=30.0)
        monkeypatch.setitem(CHECKS, "twilightLevel", loose)
        judgeStored(judged, COLLECTION, 501, 504, log=lambda _: None, now=lambda: LATEST)
        again = readGate(judged, COLLECTION)
        assert again.loc[isStep, "qaStatus"].tolist() == ["PASS"]
        assert again[PROCESSING_ID].tolist() == stale[PROCESSING_ID].tolist()
        assert judged.count("qa_gate") == 84  # no third row
        # Negative control: the rows readers do not take, under the reduction's processing, were left alone:
        # written there instead, the verdict judged again would not have been read back.
        untouched = judged.query_dataframe(
            "SELECT qa_status, qa_reason FROM qa_gate WHERE processing_id = :iq AND pfs_visit_id < 504",
            params={"iq": int(ids[RUN])},
        )
        assert len(untouched) == 36 and set(untouched["qa_reason"]) == {"as first loaded"}
