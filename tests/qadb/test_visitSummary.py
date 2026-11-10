"""Tests for reading the gate back and summarizing it per visit (`readGate`, `pfs.drp.qa.qadb.visitSummary`).

Each test loads the fixtures with the loader, as qaActor does.
"""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from pfs.drp.qa.metrics.visitRollup import EXIT_FAIL, EXIT_NO_DATA, EXIT_PASS, exitCode
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

SCRIPT = Path(__file__).parents[2] / "bin.src" / "qaVisitSummary.py"
COLLECTION = "u/test/qa"
VISITS = [133025, 133040]
NOW = datetime(2026, 10, 26, 12, 0, 0)
LATER = datetime(2026, 10, 26, 12, 5, 0)
IMAGE = ["visit", "arm", "spectrograph"]


def load(butler, qadb, now=NOW):
    loadCollection(
        butler, qadb, COLLECTION, visits=VISITS, pipelineVersions=False, log=lambda _: None, now=lambda: now
    )


@pytest.fixture
def datasets(configs):
    return {"packages": {"drp_qa": "w.2026.44"}, **configs}


@pytest.fixture
def loaded(StubButler, fixtures, datasets, qadb):
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
    assert set(gate["qaStatus"]) <= {"PASS", "WARN", "FAIL", "UNKNOWN"}
    assert set(gate["run"]) == {"u/test/qa/20261026T000000Z"}
    assert (gate["decidedAt"] == pd.Timestamp(NOW)).all()
    assert readGate(loaded, COLLECTION, first=133030)["visit"].unique().tolist() == [133040]
    assert readGate(loaded, "u/test/other").empty


def testReadGateTakesTheLatestRun(StubButler, fixtures, datasets, loaded):
    """An image judged again in a later RUN reads back once, with the later verdict."""
    before = readGate(loaded, COLLECTION)
    again = fixtures["iqQaMetrics"].assign(qaStatus="FAIL", qaDecidedBy="medFwhm", qaReason="rejudged")
    runs = {tuple(image): "u/test/qa/20261026T010000Z" for image in again[IMAGE].itertuples(index=False)}
    load(StubButler({**fixtures, "iqQaMetrics": again}, runs=runs, datasets=datasets), loaded, now=LATER)
    after = readGate(loaded, COLLECTION)
    assert len(after) == len(before) == 4
    assert set(after["qaStatus"]) == {"FAIL"} and set(after["run"]) == {"u/test/qa/20261026T010000Z"}
    # Negative control: the first RUN's verdicts were not all FAIL, so this is the later row.
    assert set(before["qaStatus"]) != {"FAIL"}


def testUnknownCollectionIsAnErrorNotNoData(loaded):
    assert collectionLoaded(loaded, COLLECTION)
    assert not collectionLoaded(loaded, "u/test/typo")
    with pytest.raises(LookupError, match="u/test/typo"):
        visitSummary(loaded, "u/test/typo")
    # Negative control: a loaded collection with no verdicts in the range is no data, not an error.
    rollup, unjudged = visitSummary(loaded, COLLECTION, 1, 2)
    assert rollup.empty and unjudged == [] and exitCode(rollup) == EXIT_NO_DATA


def testSummaryWithoutOpdb(loaded):
    rollup, unjudged = visitSummary(loaded, COLLECTION)
    gate = readGate(loaded, COLLECTION)
    assert rollup["visit"].tolist() == VISITS and unjudged == []
    assert rollup["nImages"].tolist() == [2, 2]
    assert rollup["missingCameras"].isna().all()
    for visit, status in zip(rollup["visit"], rollup["qaStatus"], strict=True):
        statuses = set(gate[gate["visit"] == visit]["qaStatus"])
        assert status == max(statuses, key=("UNKNOWN", "PASS", "WARN", "FAIL").index)


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

    def testExitStatusIsTheWorstVerdict(self, run, loaded, capsys, tmp_path):
        status, made = run("--json", str(tmp_path / "s.json"))
        rollup, _ = visitSummary(loaded, COLLECTION)
        assert status == exitCode(rollup)
        assert made == [{"host": "pfsa-db", "port": 5436, "user": "pfs", "dbname": "qadb"}]
        assert "2 visits judged" in capsys.readouterr().out
        assert json.loads((tmp_path / "s.json").read_text())["exitCode"] == status

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

    def testBadArgumentsAreNotAFail(self):
        with pytest.raises(SystemExit) as error:
            main(["--visit-range", "1"])
        assert error.value.code == EXIT_USAGE != EXIT_FAIL


def testScriptCallsMain():
    spec = importlib.util.spec_from_file_location("qaVisitSummary", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main is main
