"""Tests for reading the gate back and summarizing it per visit (`readGate`, `pfs.drp.qa.qadb.visitSummary`).

Each test loads the fixtures with the loader, as qaActor does.
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from pfs.drp.qa.metrics.visitRollup import (
    EXIT_FAIL,
    EXIT_NO_DATA,
    EXIT_PASS,
    EXIT_WARN,
    ROLLUP_COLUMNS,
    exitCode,
)
from pfs.drp.qa.qadb import visitSummary as module
from pfs.drp.qa.qadb.loader import loadCollection
from pfs.drp.qa.qadb.reader import GATE_COLUMNS, collectionLoaded, readGate
from pfs.drp.qa.qadb.visitSummary import (
    EXIT_ERROR,
    EXIT_USAGE,
    formatSummary,
    main,
    summaryDocument,
    visitSummary,
)

COLLECTION = "u/test/qa"
VISITS = [133025, 133040]
NOW = datetime(2026, 10, 26, 12, 0, 0)
LATER = datetime(2026, 10, 26, 12, 5, 0)
IMAGE = ["visit", "arm", "spectrograph"]

#: Normal widths of the fixture's images: r1's arc is +10 % (FAIL), b1's quartz +2.4 % (PASS) and r1's
#: quartz +3.9 % (WARN).
WIDTHS = {
    ("medFwhm", "r1", "arc", "Argon"): 2.5,
    ("medFwhm", "b1", "quartz", ""): 3.1,
    ("medFwhm", "r1", "quartz", ""): 3.2,
}


def load(butler, qadb, now=NOW):
    loadCollection(
        butler, qadb, COLLECTION, visits=VISITS, pipelineVersions=False, log=lambda _: None, now=lambda: now
    )


@pytest.fixture
def datasets():
    return {"packages": {"drp_qa": "w.2026.44"}}


@pytest.fixture
def loaded(StubButler, fixtures, datasets, qadb):
    """Return a database with the fixtures loaded and no normal value: every image passes."""
    load(StubButler(fixtures, datasets=datasets), qadb)
    return qadb


@pytest.fixture
def judged(StubButler, fixtures, datasets, qadb, storeValues):
    """Return a database with the fixtures loaded and judged by `WIDTHS`: a FAIL on 133025, a WARN on 133040."""
    storeValues(qadb, COLLECTION, WIDTHS, validFrom=133000)
    load(StubButler(fixtures, datasets=datasets), qadb)
    return qadb


class OneAnswerOpDB:
    """An opdb that answers every query with one table, recording the parameters."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def query_dataframe(self, sql, /, *, params=None, conn=None):
        self.calls.append(params)
        return self.answer.copy()


def cameras(pairs):
    return pd.DataFrame(pairs, columns=["visit", "camera"])


def testReadGate(loaded):
    gate = readGate(loaded, COLLECTION)
    assert list(gate.columns) == list(GATE_COLUMNS)
    assert gate[IMAGE].values.tolist() == [[v, a, 1] for v in VISITS for a in "br"]
    assert gate["qaStatus"].tolist() == ["PASS"] * 4
    assert gate["frameType"].tolist() == ["arc", "arc", "quartz", "quartz"]
    assert set(gate["run"]) == {"u/test/qa/20261026T000000Z"}
    assert (gate["decidedAt"] == pd.Timestamp(NOW)).all()
    assert readGate(loaded, COLLECTION, first=133030)["visit"].unique().tolist() == [133040]
    assert readGate(loaded, "u/test/other").empty


def testReadGateTakesTheLatestRun(StubButler, fixtures, datasets, loaded):
    """An image measured again in a later RUN reads back once, with the later RUN's verdict."""
    before = readGate(loaded, COLLECTION)
    # The later RUN finds every trace 2.5 px from the calibration: beyond the offset check's FAIL.
    metrics = fixtures["dmLineResidualMetrics"].copy()
    offset = (
        (metrics["description"] == "Trace")
        & (metrics["reference"] == "calib")
        & (metrics["metric"] == "medianDeltaX")
    )
    metrics.loc[offset, "value"] = 2.5
    runs = {
        tuple(image): "u/test/qa/20261026T010000Z"
        for image in fixtures["iqQaMetrics"][IMAGE].itertuples(index=False)
    }
    again = {**fixtures, "dmLineResidualMetrics": metrics}
    load(StubButler(again, runs=runs, datasets=datasets), loaded, now=LATER)
    after = readGate(loaded, COLLECTION)
    assert len(after) == len(before) == 4 and loaded.count("qa_gate") == 8
    assert set(after["qaStatus"]) == {"FAIL"} and set(after["run"]) == {"u/test/qa/20261026T010000Z"}
    assert set(after["qaDecidedBy"]) == {"calibOffset"}
    # Negative control: the first RUN's verdicts were not FAIL, so this is the later row.
    assert set(before["qaStatus"]) == {"PASS"}


def testUnknownCollectionIsAnErrorNotNoData(loaded):
    assert collectionLoaded(loaded, COLLECTION)
    assert not collectionLoaded(loaded, "u/test/typo")
    with pytest.raises(LookupError, match="u/test/typo"):
        visitSummary(loaded, "u/test/typo")
    # Negative control: a loaded collection with no verdicts in the range is no data, not an error.
    rollup, unjudged = visitSummary(loaded, COLLECTION, 1, 2)
    assert rollup.empty and unjudged == [] and exitCode(rollup) == EXIT_NO_DATA


def testSummaryWithoutOpdb(judged):
    rollup, unjudged = visitSummary(judged, COLLECTION)
    gate = readGate(judged, COLLECTION)
    assert list(rollup.columns) == list(ROLLUP_COLUMNS)
    assert rollup["visit"].tolist() == VISITS and unjudged == []
    assert rollup["nImages"].tolist() == [2, 2]
    assert rollup["missingCameras"].isna().all()
    assert rollup["qaStatus"].tolist() == ["FAIL", "WARN"]
    for visit, status in zip(rollup["visit"], rollup["qaStatus"], strict=True):
        statuses = set(gate[gate["visit"] == visit]["qaStatus"])
        assert status == max(statuses, key=("UNKNOWN", "PASS", "WARN", "FAIL").index)
    # The camera and the check that decided each visit, with the gate's reason.
    assert rollup["decidedByCamera"].tolist() == ["r1", "r1"]
    assert rollup["qaDecidedBy"].tolist() == ["width", "width"]
    assert rollup.loc[0, "qaReason"] == "width 2.76 px is +10.3 % from normal 2.50 px, limit +5 %"
    assert exitCode(rollup) == EXIT_FAIL


def testFormatSummaryNamesTheCameraAndTheCheck(judged):
    rollup, _ = visitSummary(judged, COLLECTION)
    lines = formatSummary(rollup).splitlines()
    assert lines[0] == "2 visits judged: 1 FAIL, 1 WARN, 0 PASS, 0 UNKNOWN"
    assert lines[1] == (
        "133025 FAIL     2 images (F1 W0 P1 U0) r1 width:"
        " width 2.76 px is +10.3 % from normal 2.50 px, limit +5 %"
    )
    assert lines[2].startswith("133040 WARN     2 images (F0 W1 P1 U0) r1 width: width 3.33 px")
    assert "provisional" not in "\n".join(lines) and "imageQualityQa" not in "\n".join(lines)
    # A deciding image with no reason stored is named by its check alone.
    bare = formatSummary(rollup.assign(qaReason="")).splitlines()
    assert bare[1] == "133025 FAIL     2 images (F1 W0 P1 U0) r1 width"


def testAPassedVisitNamesNoDecider(loaded):
    rollup, _ = visitSummary(loaded, COLLECTION)
    assert rollup["qaStatus"].tolist() == ["PASS", "PASS"]
    assert formatSummary(rollup).splitlines()[1:] == ["133025 PASS     2 images", "133040 PASS     2 images"]
    assert exitCode(rollup) == EXIT_PASS


def testSummaryWithOpdb(loaded):
    """Cameras that exposed without a verdict are named; a visit with none is listed apart."""
    opdb = OneAnswerOpDB(
        cameras(
            [(133025, "b1"), (133025, "r1"), (133025, "n1"), (133030, "b1"), (133040, "b1"), (133040, "r1")]
        )
    )
    rollup, unjudged = visitSummary(loaded, COLLECTION, opdb=opdb)
    # An open range reads the opdb over the judged visits' span only.
    assert opdb.calls == [{"first": 133025, "last": 133040}]
    assert rollup["visit"].tolist() == VISITS
    assert rollup["missingCameras"].tolist() == ["n1", ""]
    assert rollup["nExpected"].tolist() == [3, 2]
    assert unjudged == [133030]
    text = formatSummary(rollup, unjudged)
    assert "2 visits judged" in text and "missing n1" in text and "1 visits with SpS exposures" in text

    given = OneAnswerOpDB(cameras([]))
    visitSummary(loaded, COLLECTION, 133000, 133100, opdb=given)
    assert given.calls == [{"first": 133000, "last": 133100}]


def testOpenEndWithNothingJudgedDoesNotReadTheOpdb(loaded):
    opdb = OneAnswerOpDB(cameras([(5, "b1")]))
    for ends in ({"first": 134000}, {"last": 1000}):
        rollup, unjudged = visitSummary(loaded, COLLECTION, opdb=opdb, **ends)
        assert rollup.empty and unjudged == []
    assert opdb.calls == []
    # Negative control: with both ends given the opdb says what exposed unjudged.
    assert visitSummary(loaded, COLLECTION, 1, 1000, opdb=opdb)[1] == [5] and len(opdb.calls) == 1


def testDocumentIsJson(loaded):
    rollup, _ = visitSummary(loaded, COLLECTION)
    document = json.loads(json.dumps(summaryDocument(COLLECTION, rollup, [133030])))
    assert document["collection"] == COLLECTION and document["unjudgedVisits"] == [133030]
    assert [visit["visit"] for visit in document["visits"]] == VISITS
    assert document["exitCode"] == exitCode(rollup)
    assert sum(document["counts"].values()) == 2


class TestMain:
    @pytest.fixture
    def run(self, monkeypatch, loaded, tmp_path):
        import pfs.utils.database.qadb as qadbModule

        made = []

        def fake(**kwargs):
            made.append(kwargs)
            return loaded

        monkeypatch.setattr(qadbModule, "QaDB", fake)
        return lambda *argv: (main(["-c", COLLECTION, *argv]), made)

    def testExitStatusIsTheWorstVerdict(
        self, run, loaded, capsys, tmp_path, storeValues, StubButler, fixtures
    ):
        status, made = run("--json", str(tmp_path / "s.json"))
        rollup, _ = visitSummary(loaded, COLLECTION)
        assert status == exitCode(rollup) == EXIT_PASS == 0
        assert made == [{"host": "pfsa-db", "port": 5436, "user": "pfs", "dbname": "qadb"}]
        assert "2 visits judged" in capsys.readouterr().out
        document = json.loads((tmp_path / "s.json").read_text())
        assert document["exitCode"] == status
        assert not {"decidedByTask", "provisional"} & set(document["visits"][0])
        # Judged with normal values, the same collection warns on one visit and fails on the other.
        storeValues(loaded, COLLECTION, WIDTHS, validFrom=133000)
        load(StubButler(fixtures), loaded, now=LATER)
        assert run("--visit-range", "133040", "133040")[0] == EXIT_WARN == 1
        assert run()[0] == EXIT_FAIL == 2

    def testNoDataAndErrorsAreNotVerdicts(self, run, monkeypatch, capsys):
        assert run("--visit-range", "1", "2")[0] == EXIT_NO_DATA == 3
        assert main(["-c", "u/test/typo"]) == EXIT_ERROR == 4
        assert "u/test/typo" in capsys.readouterr().err

        def broken(*args, **kwargs):
            raise ConnectionError("refused")

        monkeypatch.setattr(module, "visitSummary", broken)
        status = main(["-c", COLLECTION])
        # Negative control: a failed read is neither a pass nor a FAIL verdict.
        assert status == EXIT_ERROR and status not in (EXIT_PASS, EXIT_FAIL)
        assert "refused" in capsys.readouterr().err

    @pytest.mark.parametrize("argv", [["--visit-range", "1"], ["-c", COLLECTION, "--visit-range", "9", "1"]])
    def testBadArgumentsAreNotAFail(self, argv):
        with pytest.raises(SystemExit) as error:
            main(argv)
        assert error.value.code == EXIT_USAGE != EXIT_FAIL

    def testJsonThatCannotBeWrittenIsNotAWarn(self, run, tmp_path, capsys):
        status, _ = run("--json", str(tmp_path / "absent" / "s.json"))
        assert status == EXIT_ERROR != 1 and "absent" in capsys.readouterr().err
        # Negative control: the same summary, written where it can be, is a verdict.
        assert run("--json", str(tmp_path / "s.json"))[0] != EXIT_ERROR
