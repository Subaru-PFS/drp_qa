"""Tests for the run comparison read from qadb (`pfs.drp.qa.qadb.compareRuns`).

Each test loads the fixtures with the loader, as qaActor does.
"""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

from pfs.drp.qa.metrics.runComparison import COMPARISON_COLUMNS, exitCode
from pfs.drp.qa.metrics.visitRollup import EXIT_FAIL, EXIT_NO_DATA, EXIT_PASS
from pfs.drp.qa.qadb import compareRuns as module
from pfs.drp.qa.qadb.compareRuns import (
    Reference,
    compareCollections,
    comparisonDocument,
    formatComparison,
    main,
    metricTables,
    parseReference,
)
from pfs.drp.qa.qadb.loader import loadCollection
from pfs.drp.qa.qadb.visitSummary import EXIT_ERROR, EXIT_USAGE

SCRIPT = Path(__file__).parents[2] / "bin.src" / "qaCompareRuns.py"
COLLECTION = "u/test/qa"
VISITS = [133025, 133040]
NOW = datetime(2026, 10, 26, 12, 0, 0)
EARLIER = "u/test/before"
BEFORE = Reference("before", VISITS[0], VISITS[1], EARLIER)
#: How far the run's widths sit above the earlier collection's: past the FAIL tolerance of ``medFwhm``.
WIDER = 1.1


@pytest.fixture
def loaded(StubButler, fixtures, configs, qadb):
    """Load the fixtures as the run, and as an earlier collection whose widths were narrower."""
    datasets = {"packages": {"drp_qa": "w.2026.44"}, **configs}
    narrower = fixtures["iqQaMetrics"].assign(medFwhm=fixtures["iqQaMetrics"]["medFwhm"] / WIDER)
    for collection, tables in ((COLLECTION, fixtures), (EARLIER, {**fixtures, "iqQaMetrics": narrower})):
        loadCollection(
            StubButler(tables, datasets=datasets),
            qadb,
            collection,
            visits=VISITS,
            pipelineVersions=False,
            log=lambda _: None,
            now=lambda: NOW,
        )
    return qadb


def testParseReference():
    assert parseReference("firstNight=153534-153600") == Reference("firstNight", 153534, 153600)
    assert parseReference("run30=u/me/run30:1-2") == Reference("run30", 1, 2, "u/me/run30")
    for bad in ("153534-153600", "a=1", "a=u/x:1", "a=2-1", "=1-2", "a=x-2"):
        with pytest.raises(ValueError, match=r"FIRST-LAST|ends before"):
            parseReference(bad)


def testMetricTablesCarryTheFrameType(loaded, fixtures):
    iq, residuals = metricTables(loaded, COLLECTION, VISITS[0], VISITS[0])
    assert set(iq["visit"]) == {VISITS[0]} and "medFwhm" in iq
    assert set(residuals["visit"]) == {VISITS[0]}
    assert residuals["obsType"].notna().all() and len(residuals)
    with pytest.raises(LookupError, match="u/test/typo"):
        metricTables(loaded, "u/test/typo", 1, 2)


def testCompareCollections(loaded):
    comparison = compareCollections(loaded, COLLECTION, *VISITS, [BEFORE])
    assert list(comparison.columns) == list(COMPARISON_COLUMNS) and len(comparison)
    assert set(comparison["referenceName"]) == {"before"}
    assert {"medFwhm", "robustRmsDeltaX"} <= set(comparison["metric"])
    widths = comparison[comparison["metric"] == "medFwhm"]
    assert len(widths) == 4 and set(widths["status"]) == {"FAIL"}
    assert widths["relative"].to_numpy() == pytest.approx(WIDER - 1, rel=1e-5)
    # Only the widths differ between the two collections.
    assert set(comparison[comparison["metric"] != "medFwhm"]["status"]) <= {"PASS", "UNKNOWN"}

    # An arc is not compared with a trace: the two visits share no population.
    apart = compareCollections(
        loaded, COLLECTION, VISITS[1], VISITS[1], [Reference("arc", VISITS[0], VISITS[0])]
    )
    assert len(apart) and (apart["nReference"] == 0).all() and set(apart["status"]) == {"UNKNOWN"}

    same = compareCollections(
        loaded,
        COLLECTION,
        VISITS[0],
        VISITS[0],
        [Reference("itself", VISITS[0], VISITS[0], COLLECTION)],
        knownGood=False,
    )
    assert (same["difference"] == 0).all() and set(same["status"]) <= {"PASS", "UNKNOWN"}
    assert (same["nRun"] == same["nReference"]).all()
    # Negative control: against the earlier collection the medians do differ.
    assert (comparison["difference"].dropna() != 0).any()

    with pytest.raises(LookupError, match="u/test/other"):
        compareCollections(loaded, COLLECTION, 1, 2, [Reference("a", 1, 2, "u/test/other")])


def testFormatAndDocument(loaded):
    comparison = compareCollections(loaded, COLLECTION, *VISITS, [BEFORE])
    flagged = comparison[comparison["status"].isin(["WARN", "FAIL"])]
    assert len(flagged) == 4
    text = formatComparison(comparison)
    assert text.startswith(f"against before: {len(comparison)} compared")
    assert len(text.splitlines()) == 1 + len(flagged)
    assert len(formatComparison(comparison, everything=True).splitlines()) == 1 + len(comparison)
    assert formatComparison(comparison.iloc[:0]) == "nothing to compare"

    document = json.loads(json.dumps(comparisonDocument(COLLECTION, tuple(VISITS), [BEFORE], comparison)))
    assert document["exitCode"] == exitCode(comparison) == EXIT_FAIL
    assert document["references"] == [{"name": "before", "collection": EARLIER, "visits": VISITS}]
    assert len(document["rows"]) == len(comparison) and sum(document["counts"].values()) == len(comparison)


class TestMain:
    ARGS = (
        "-c",
        COLLECTION,
        "--visit-range",
        "133025",
        "133040",
        "--reference",
        f"before={EARLIER}:133025-133040",
    )

    @pytest.fixture
    def made(self, monkeypatch, loaded):
        import pfs.utils.database.qadb as qadbModule

        made = []

        def fake(**kwargs):
            made.append(kwargs)
            return loaded

        monkeypatch.setattr(qadbModule, "QaDB", fake)
        return made

    def testExitStatusIsTheWorstStatus(self, made, loaded, capsys, tmp_path):
        status = main([*self.ARGS, "--json", str(tmp_path / "c.json")])
        assert status == exitCode(compareCollections(loaded, COLLECTION, *VISITS, [BEFORE])) == EXIT_FAIL
        assert made == [{"host": "pfsa-db", "port": 5436, "user": "pfs", "dbname": "qadb"}]
        assert "against before" in capsys.readouterr().out
        assert json.loads((tmp_path / "c.json").read_text())["exitCode"] == status

    def testReferenceDecidesTheStatus(self, made, capsys):
        """The same visits fail against the narrower collection and pass against themselves."""
        assert main(list(self.ARGS)) == EXIT_FAIL
        assert "FAIL    medFwhm b1" in capsys.readouterr().out
        # Negative control: nothing but the reference changed.
        assert main([*self.ARGS[:-1], "itself=133025-133040", "--all-reference-values"]) == EXIT_PASS

    def testNoDataAndErrorsAreNotVerdicts(self, made, monkeypatch, capsys):
        assert main(["-c", COLLECTION, "--visit-range", "1", "2", "--reference", "a=3-4"]) == EXIT_NO_DATA
        assert main(["-c", "u/test/typo", *self.ARGS[2:]]) == EXIT_ERROR
        assert "u/test/typo" in capsys.readouterr().err

        def broken(*args, **kwargs):
            raise ConnectionError("refused")

        monkeypatch.setattr(module, "compareCollections", broken)
        status = main(list(self.ARGS))
        # Negative control: a failed read is neither a pass nor a FAIL.
        assert status == EXIT_ERROR and status not in (EXIT_PASS, EXIT_FAIL)
        assert "refused" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "argv",
        [
            ["-c", COLLECTION, "--visit-range", "1", "2"],
            ["-c", COLLECTION, "--visit-range", "1", "2", "--reference", "nonsense"],
            [*ARGS, "--tolerance", "medFwhm=0.2,0.3"],
        ],
    )
    def testBadArgumentsAreNotAFail(self, argv):
        with pytest.raises(SystemExit) as error:
            main(argv)
        assert error.value.code == EXIT_USAGE != EXIT_FAIL


def testScriptCallsMain():
    spec = importlib.util.spec_from_file_location("qaCompareRuns", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main is main
