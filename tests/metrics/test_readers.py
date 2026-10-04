"""Tests for reading stored metrics from a Butler, against a stub."""

from dataclasses import dataclass, field

import pandas as pd
import pytest

from pfs.drp.qa.metrics.readers import chooseRun, readMetrics


@dataclass(frozen=True)
class StubRef:
    datasetType: str
    dataId: dict = field(hash=False)
    run: str = "run"

    def __hash__(self):
        return hash((self.datasetType, tuple(sorted(self.dataId.items())), self.run))


class StubButler:
    """Answers ``query_datasets`` from a dict of tables, keyed by data ID.

    Records each query so tests can check what was asked.
    """

    def __init__(self, tables: dict[tuple[int, str, int], pd.DataFrame]):
        self.tables = tables
        self.queries = []

    def query_datasets(self, datasetType, collections=None, where="", bind=None, **kwargs):
        self.queries.append(
            {"datasetType": datasetType, "collections": collections, "where": where, "bind": bind, **kwargs}
        )
        visits = set(bind["visits"])
        return [
            StubRef(
                datasetType, {"instrument": bind["instrumentName"], "visit": v, "arm": a, "spectrograph": s}
            )
            for (v, a, s) in self.tables
            if v in visits
        ]

    def get(self, ref):
        dataId = ref.dataId
        return self.tables[(dataId["visit"], dataId["arm"], dataId["spectrograph"])].copy()


def table(fwhm: float, withIds: tuple | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"medFwhm": [fwhm], "seqName": ["Arc: Neon"]})
    if withIds:
        frame["visit"], frame["arm"], frame["spectrograph"] = withIds
    return frame


@pytest.fixture
def butler() -> StubButler:
    return StubButler(
        {
            (101, "r", 1): table(2.7),
            (100, "b", 2): table(2.6, withIds=(100, "b", 2)),
            (100, "b", 1): table(2.5),
            (999, "b", 1): table(9.9),
        }
    )


def testReadsAndSorts(butler):
    metrics = readMetrics(butler, [101, 100, 100])
    assert metrics[["visit", "arm", "spectrograph"]].values.tolist() == [
        [100, "b", 1],
        [100, "b", 2],
        [101, "r", 1],
    ]
    assert metrics["medFwhm"].tolist() == [2.5, 2.6, 2.7]


def testOneQueryForEveryVisit(butler):
    """One registry query, find-first, bound rather than interpolated."""
    readMetrics(butler, [100, 101], collections=["u/someone/qa"])
    (query,) = butler.queries
    assert query["datasetType"] == "iqQaMetrics"
    assert query["collections"] == ["u/someone/qa"]
    assert query["bind"] == {"instrumentName": "PFS", "visits": (100, 101)}
    assert "100" not in query["where"]
    assert query["find_first"] is True


def testNothingFoundRaises(butler):
    with pytest.raises(LookupError, match="No iqQaMetrics datasets for 1 visits"):
        readMetrics(butler, [5])


def testNoVisitsRaises(butler):
    with pytest.raises(ValueError, match="No visits"):
        readMetrics(butler, [])


class TestChooseRun:
    PREFIX = "u/someone/qa-thresholds"
    # As `butler query-collections` prints them: a table, with other collections around.
    LISTING = (
        "Name                          Type",
        "----------------------------- ----",
        "u/someone/qa-thresholds/001   RUN",
        "u/someone/qa-thresholds/003   RUN",
        "u/someone/qa-thresholds/0004x RUN",
        "u/someone/qa-thresholds-old/9 RUN",
        "u/other/qa-thresholds/007     RUN",
    )

    def testNewRunTakesTheNextNumber(self):
        assert chooseRun(self.LISTING, self.PREFIX, newRun=True) == f"{self.PREFIX}/004"

    def testFirstRunIsOne(self):
        assert chooseRun([], self.PREFIX, newRun=True) == f"{self.PREFIX}/001"

    def testReadingTakesTheLatest(self):
        assert chooseRun(self.LISTING, self.PREFIX, newRun=False) == f"{self.PREFIX}/003"

    def testReadingWithNoRunsRaises(self):
        with pytest.raises(LookupError, match="RUN_PIPELINE"):
            chooseRun(self.LISTING[:2], self.PREFIX, newRun=False)

    def testExplicitNameWins(self):
        name = f"{self.PREFIX}/001"
        assert chooseRun(self.LISTING, self.PREFIX, newRun=True, runName=name) == name
