"""Tests for the ``drpQa.py`` subcommands that work on qadb: ``values``, ``judge`` and ``load``'s arguments.

``values`` and ``judge`` run on the ``qadb`` fixture, given to them in place of
the `pfs.utils.database.qadb.QaDB` they would open. The frames are those of
``test_judge.py``: synthetic quartzes, calibration frames 101 to 105 and the
frames 106 to 108, whose b1 widths are a PASS, a FAIL and a WARN of the width
check once the calibration frames' median is the normal value. ``load`` needs a
Butler, so only its arguments are tested, and one load from a stand-in Butler.
"""

import sys
import types
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from pfs.drp.qa.qadb import cli
from pfs.drp.qa.qadb.cli import EXIT_REFUSED, judgeMain, loadMain, valuesMain
from pfs.drp.qa.qadb.loader import loadCollection

COLLECTION = "u/test/cli"
NOW = datetime(2026, 10, 26, 12, 0, 0)
CALIBRATION = [101, 102, 103, 104, 105]
FRAMES = [106, 107, 108]
B1_WIDTH = {101: 3.00, 102: 3.02, 103: 3.04, 104: 3.06, 105: 3.08, 106: 3.05, 107: 3.20, 108: 3.14}
NORMAL_B1 = 3.04
WIDTH = {**{(visit, "b1"): width for visit, width in B1_WIDTH.items()}, None: 3.30}
B1 = ("medFwhm", "b1", "quartz", "")


def load(butler, qadb, visits):
    loadCollection(
        butler, qadb, COLLECTION, visits=visits, pipelineVersions=False, log=lambda _: None, now=lambda: NOW
    )


@pytest.fixture
def db(StubButler, quartzFrames, qadb, monkeypatch):
    """Return the database the subcommands open, with the frames loaded; ``db.opened`` lists how it was asked for."""
    visits = [*CALIBRATION, *FRAMES]
    load(StubButler(quartzFrames(visits, width=WIDTH)), qadb, visits)
    qadb.opened = []
    monkeypatch.setattr("pfs.drp.qa.qadb.cli._qadb", lambda args: qadb.opened.append(args) or qadb)
    return qadb


def normalValues(qadb) -> pd.DataFrame:
    return qadb.query_dataframe(
        "SELECT * FROM qa_normal_value WHERE collection_name = :c ORDER BY valid_from, metric, camera",
        params={"c": COLLECTION},
    )


def gateOf(qadb) -> pd.DataFrame:
    rows = qadb.query_dataframe(
        "SELECT g.* FROM qa_gate g JOIN data_processing_pipe2d p ON p.processing_id = g.processing_id"
        " WHERE p.collection_name = :c ORDER BY g.pfs_visit_id, g.camera_name",
        params={"c": COLLECTION},
    )
    # SQLite returns the times as text, with a fraction of a second only when there is one.
    return rows.assign(decided_at=pd.to_datetime(rows["decided_at"], format="mixed"))


def statusOf(qadb) -> dict:
    rows = gateOf(qadb)
    return dict(
        zip(zip(rows["pfs_visit_id"], rows["camera_name"], strict=True), rows["qa_status"], strict=True)
    )


def values(*arguments):
    return valuesMain(["-c", COLLECTION, *arguments])


class TestValues:
    def testWithoutAcceptNothingIsStored(self, db, capsys):
        before = gateOf(db)
        assert values("--visits", "101-105") == 0
        out = capsys.readouterr().out
        assert "nothing stored: 6 values would be, with --accept" in out
        assert normalValues(db).empty
        pd.testing.assert_frame_equal(gateOf(db), before)
        # Each candidate is listed beside the value in force: there is none yet, so each is new.
        assert out.count("  NEW\n") == 6 and "\n6 NEW\n" in out
        # Negative control: with --accept the same command stores them.
        assert values("--visits", "101-105", "--accept", "tester") == 0
        assert len(normalValues(db)) == 6

    def testAcceptStoresWithThatName(self, db, capsys):
        assert values("--visits", "101-105", "--accept", "tester") == 0
        out = capsys.readouterr().out
        assert "6 values stored, accepted by tester" in out and "nothing stored" not in out
        stored = normalValues(db)
        assert len(stored) == 6 and set(stored["accepted_by"]) == {"tester"}
        assert set(stored["valid_from"]) == {106} and set(stored["n_frames"]) == {5}
        b1 = stored[(stored["metric"] == "medFwhm") & (stored["camera"] == "b1")].iloc[0]
        assert b1["value"] == pytest.approx(NORMAL_B1)
        # The frames the values cover were judged again, and said so.
        status = statusOf(db)
        assert [status[(visit, "b1")] for visit in FRAMES] == ["PASS", "FAIL", "WARN"]
        assert "qa_normal_value: 6 values valid from visit 106" in out
        assert "6 images judged again in qa_gate: 4 PASS, 1 WARN, 1 FAIL" in out

    def testAcceptWithoutANameIsTheUserRunningIt(self, db, monkeypatch):
        monkeypatch.setattr(cli.getpass, "getuser", lambda: "operator")
        assert values("--visits", "101-105", "--accept") == 0
        assert set(normalValues(db)["accepted_by"]) == {"operator"}

    def testAShiftBeyondFailIsRefusedWithoutAReason(self, db, StubButler, quartzFrames, capsys):
        assert values("--visits", "101-105", "--accept", "tester") == 0
        second = [201, 202, 203, 204, 205, 206]
        wider = round(NORMAL_B1 * 1.08, 4)  # 8 %: beyond the width check's FAIL of 5 %
        load(StubButler(quartzFrames(second, width={"b1": wider, None: 3.30})), db, second)
        capsys.readouterr()

        # Only looking: the exit status already says a value would be refused.
        assert values("--visits", "201-205") == EXIT_REFUSED == 1
        out = capsys.readouterr().out
        assert "1 values are shifted beyond FAIL and refused; give --reason to accept them" in out
        assert set(normalValues(db)["valid_from"]) == {106}

        assert values("--visits", "201-205", "--accept", "tester") == EXIT_REFUSED
        out = capsys.readouterr().out
        assert "5 values stored, accepted by tester" in out
        new = normalValues(db)
        new = new[new["valid_from"] == 206]
        assert len(new) == 5 and not ((new["metric"] == "medFwhm") & (new["camera"] == "b1")).any()
        assert statusOf(db)[(206, "b1")] == "FAIL"  # the value in force stays for the one refused

        assert values("--visits", "201-205", "--accept", "tester", "--reason", "refocused") == 0
        out = capsys.readouterr().out
        assert "6 values stored, accepted by tester" in out and "refused" not in out
        new = normalValues(db)
        new = new[new["valid_from"] == 206].set_index(["metric", "camera"])
        assert len(new) == 6 and new.loc[("medFwhm", "b1"), "reason"] == "refocused"
        assert new.loc[("medFwhm", "b1"), "value"] == pytest.approx(wider)
        # Negative control: the values that did not shift carry no reason.
        assert set(new.drop(index=[("medFwhm", "b1")])["reason"]) == {""}
        assert statusOf(db)[(206, "b1")] == "PASS"

    def testArmsReachMakeValues(self, db):
        assert values("--visits", "101-105", "--arms", "b", "--accept", "tester") == 0
        assert set(normalValues(db)["camera"]) == {"b", "b1"}
        # Negative control: several arms are each taken, as letters.
        assert values("--visits", "101-105", "--arms", "br", "--valid-from", "200", "--accept", "tester") == 0
        stored = normalValues(db)
        assert set(stored.loc[stored["valid_from"] == 200, "camera"]) == {"b", "b1", "r", "r1"}

    def testValidFromReachesMakeValues(self, db):
        assert values("--visits", "101-105", "--valid-from", "104", "--accept", "tester") == 0
        assert set(normalValues(db)["valid_from"]) == {104}
        rows = gateOf(db)
        judgedAgain = rows.loc[rows["decided_at"] != pd.Timestamp(NOW), "pfs_visit_id"]
        assert sorted(set(judgedAgain)) == [104, 105, 106, 107, 108]
        # Negative control: without it the values start after the calibration frames.
        assert values("--visits", "101-105", "--accept", "tester") == 0
        assert set(normalValues(db)["valid_from"]) == {104, 106}

    def testSeveralRangesAndSingleVisits(self, db):
        assert values("--visits", "101-102", "104", "105", "--accept", "tester") == 0
        stored = normalValues(db)
        assert set(stored["n_frames"]) == {4} and set(stored["source_visits"]) == {"101-102,104-105"}
        assert set(stored["valid_from"]) == {106}

    def testTooFewFramesAreSaidAndNotStored(self, db, capsys):
        assert values("--visits", "101-102", "--accept", "tester") == 0
        out = capsys.readouterr().out
        assert "6 values rest on too few frames and are left out; the values in force stay for them" in out
        assert "0 values stored, accepted by tester" in out and normalValues(db).empty

    def testAllListsTheValuesThatDidNotMove(self, db, capsys):
        assert values("--visits", "101-105", "--accept", "tester") == 0
        capsys.readouterr()
        # The same frames again, valid later: every value equals the one in force.
        assert values("--visits", "101-105", "--valid-from", "107") == 0
        moved = capsys.readouterr().out
        assert values("--visits", "101-105", "--valid-from", "107", "--all") == 0
        everything = capsys.readouterr().out
        assert moved.count("medFwhm") == 0 and everything.count("medFwhm") == 2
        assert "6 PASS" in moved and "6 PASS" in everything

    @pytest.mark.parametrize(
        ("visits", "message"),
        [
            ("5-3", "the first visit is after the last"),
            ("x", "is not a visit or FIRST-LAST"),
            ("3-", "is not a visit or FIRST-LAST"),
        ],
    )
    def testBadVisitsAreArgumentErrors(self, db, capsys, visits, message):
        with pytest.raises(SystemExit) as error:
            values("--visits", visits, "--accept", "tester")
        assert error.value.code == 2
        assert message in capsys.readouterr().err
        # Refused before the database is opened: nothing read, nothing stored.
        assert db.opened == [] and normalValues(db).empty

    def testTheCollectionAndTheVisitsAreRequired(self, db, capsys):
        for arguments in (["--visits", "101-105"], ["-c", COLLECTION]):
            with pytest.raises(SystemExit) as error:
                valuesMain(arguments)
            assert error.value.code == 2
        assert "required" in capsys.readouterr().err and db.opened == []

    def testTheDatabaseOptionsReachQadb(self, db):
        values("--visits", "101-105", "--host", "somewhere", "--port", "5555", "--dbname", "copy")
        (asked,) = db.opened
        assert (asked.host, asked.port, asked.user, asked.dbname) == ("somewhere", 5555, "pfs", "copy")


class TestJudge:
    @pytest.fixture
    def judged(self, db, storeValues):
        """Return the database with b1's normal width stored, valid from 106, and the frames not judged by it yet."""
        storeValues(db, COLLECTION, {B1: NORMAL_B1}, validFrom=106)
        return db

    def testDryRunWritesNothingAndJudgeDoes(self, judged, capsys):
        before = gateOf(judged)
        assert statusOf(judged)[(107, "b1")] == "PASS"
        assert judgeMain(["-c", COLLECTION, "--visits", "106", "108", "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "in 1 read(s), writing nothing" in out
        assert "6 images judged, nothing written: 4 PASS, 1 WARN, 1 FAIL" in out
        pd.testing.assert_frame_equal(gateOf(judged), before)

        assert judgeMain(["-c", COLLECTION, "--visits", "106", "108"]) == 0
        out = capsys.readouterr().out
        assert (
            "6 images judged again in qa_gate: 4 PASS, 1 WARN, 1 FAIL" in out and "nothing written" not in out
        )
        status = statusOf(judged)
        assert [status[(visit, "b1")] for visit in FRAMES] == ["PASS", "FAIL", "WARN"]
        rows = gateOf(judged)
        assert set(rows.loc[rows["decided_at"] != pd.Timestamp(NOW), "pfs_visit_id"]) == set(FRAMES)

    def testOnlyTheVisitsGiven(self, judged):
        assert judgeMain(["-c", COLLECTION, "--visits", "107", "107"]) == 0
        status = statusOf(judged)
        assert status[(107, "b1")] == "FAIL"
        # Negative control: 108 would warn, and was not judged.
        assert status[(108, "b1")] == "PASS"

    def testNothingLoadedIsSaid(self, judged, capsys):
        assert judgeMain(["-c", COLLECTION, "--visits", "900", "999"]) == 0
        assert f"no numbers for visits 900 to 999 in {COLLECTION}" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "arguments",
        [["-c", COLLECTION], ["--visits", "106", "108"], ["-c", COLLECTION, "--visits", "106"]],
    )
    def testArgumentErrors(self, judged, arguments, capsys):
        with pytest.raises(SystemExit) as error:
            judgeMain(arguments)
        assert error.value.code == 2 and judged.opened == []
        assert "usage:" in capsys.readouterr().err


class FakeButler:
    """Stand in for `lsst.daf.butler.Butler`: a stub over the tables given, knowing the collections given."""

    tables: dict = {}  # noqa: RUF012
    known: tuple = ()
    made: list = []  # noqa: RUF012

    def __new__(cls, repo, writeable=True):
        from pfs.drp.qa.qadb.loader import TABLES  # noqa: F401

        cls.made.append((repo, writeable))
        butler = cls.Stub(cls.tables, datasets={"packages": {"drp_qa": "w.2026.44"}})
        butler.collections = SimpleNamespace(query=lambda name: [name] if name in cls.known else [])
        return butler


@pytest.fixture
def butlers(monkeypatch, StubButler, fixtures):
    """Make ``from lsst.daf.butler import Butler`` give a stand-in over the fixtures; return its class."""
    FakeButler.Stub, FakeButler.tables, FakeButler.known, FakeButler.made = (
        StubButler,
        fixtures,
        (COLLECTION,),
        [],
    )
    module = types.ModuleType("lsst.daf.butler")
    module.Butler = FakeButler
    for name in ("lsst", "lsst.daf"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lsst.daf.butler", module)
    return FakeButler


class TestLoadArguments:
    """What ``load`` settles from its arguments, before any Butler is opened."""

    @pytest.fixture
    def opened(self, butlers, qadb, monkeypatch):
        """Return the list of databases ``load`` asked for; each is the ``qadb`` fixture."""
        opened = []
        monkeypatch.setattr("pfs.drp.qa.qadb.cli._qadb", lambda args: opened.append(args) or qadb)
        return opened

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (["-c", COLLECTION], "one of the arguments --visits --visit-range is required"),
            (
                ["-c", COLLECTION, "--visits", "1", "--visit-range", "1", "2"],
                "not allowed with argument --visits",
            ),
            (["--visits", "1"], "the following arguments are required: -c/--collection"),
            (["-c", COLLECTION, "--visit-range", "1"], "expected 2 arguments"),
            (["-c", COLLECTION, "--visits", "1", "--mode", "nightly"], "invalid choice: 'nightly'"),
        ],
    )
    def testArgumentErrors(self, opened, butlers, capsys, arguments, message):
        with pytest.raises(SystemExit) as error:
            loadMain(arguments)
        assert error.value.code == 2
        assert message in capsys.readouterr().err
        # Refused on the arguments alone: no Butler made, no database opened.
        assert butlers.made == [] and opened == []

    def testAnUnknownCollectionStopsBeforeTheDatabase(self, opened, butlers):
        with pytest.raises(SystemExit, match="No collection 'u/test/typo' in /work/datastore"):
            loadMain(["-c", "u/test/typo", "--visits", "133025"])
        # Negative control: these arguments did pass, and a read-only Butler was opened to look.
        assert butlers.made == [("/work/datastore", False)] and opened == []

    def testADryRunOpensNoDatabase(self, opened, butlers, tmp_path, capsys):
        arguments = ["-b", "/some/repo", "-c", COLLECTION, "--visit-range", "133025", "133040"]
        assert loadMain([*arguments, "--no-pipeline-versions", "--dry-run", str(tmp_path)]) == 0
        assert butlers.made == [("/some/repo", False)] and opened == []
        stored = pd.read_parquet(tmp_path / "qa_gate.parquet")
        assert len(stored) == 4 and set(stored["mode"]) == {"backfill"}
        assert "qa_gate: 4" in capsys.readouterr().out

    def testTheVisitsAndTheModeReachTheLoad(self, opened, butlers, qadb):
        arguments = ["-c", COLLECTION, "--visits", "133040", "--mode", "realtime", "--no-pipeline-versions"]
        assert loadMain(arguments) == 0
        assert len(opened) == 1
        rows = gateOf(qadb)
        assert set(rows["pfs_visit_id"]) == {133040} and set(rows["mode"]) == {"realtime"}
        # Negative control: the other visit of the collection, not asked for, was not loaded.
        assert qadb.count("image_quality_qa", "pfs_visit_id = 133025") == 0
