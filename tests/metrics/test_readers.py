"""Tests for reading stored metrics from a Butler, against a stub."""

from dataclasses import dataclass, field

import pandas as pd
import pytest

from pfs.drp.qa.metrics.readers import chooseRun, readMetrics, readPipelineVersions


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

    def testExplicitNameIsRead(self):
        name = f"{self.PREFIX}/001"
        assert chooseRun(self.LISTING, self.PREFIX, newRun=False, runName=name) == name

    def testExplicitNameCannotBeWrittenInto(self):
        with pytest.raises(ValueError, match="RUN_NAME"):
            chooseRun(self.LISTING, self.PREFIX, newRun=True, runName=f"{self.PREFIX}/001")

    def testErrorTextIsNotARun(self):
        """The CLI's error for a missing collection names it; that is not a listing."""
        error = [f"Collection '{self.PREFIX}/001' not found"]
        assert chooseRun(error, self.PREFIX, newRun=True) == f"{self.PREFIX}/001"


class StubURI:
    """Stands in for ``lsst.resources.ResourcePath``: a local file."""

    def __init__(self, path):
        self.ospath = str(path)

    def as_local(self):
        from contextlib import nullcontext

        return nullcontext(self)


def testReadPipelineVersions():
    """One query for every visit, each pfsArm's header read once, unrecorded versions null."""
    headers = {
        (101, "r", 1): {"VERSION_DATAMODEL": "a", "VERSION_OBS_PFS": "b", "VERSION_DRP_STELLA": "c"},
        (101, "b", 1): {"VERSION_DRP_STELLA": "c"},
    }
    butler = StubButler({key: table(2.7) for key in headers})
    butler.getURI = lambda ref: (ref.dataId["visit"], ref.dataId["arm"], ref.dataId["spectrograph"])
    versions = readPipelineVersions(butler, [101, 101], collections=["out"], readHeader=headers.__getitem__)
    assert [query["datasetType"] for query in butler.queries] == ["pfsArm"]
    assert butler.queries[0]["find_first"] and butler.queries[0]["collections"] == ["out"]
    assert versions[["visit", "arm", "spectrograph"]].values.tolist() == [[101, "b", 1], [101, "r", 1]]
    assert versions["versionDrpStella"].tolist() == ["c", "c"]
    assert versions["versionObsPfs"].iloc[1] == "b" and pd.isna(versions["versionObsPfs"].iloc[0])
    assert versions["versionDatamodel"].iloc[1] == "a" and pd.isna(versions["versionDatamodel"].iloc[0])
    assert readPipelineVersions(butler, []).empty


def testReadPipelineVersionsFromFits(tmp_path):
    """The default reader takes the primary header of the file, HIERARCH keys and all."""
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "pfsArm.fits"
    header = fits.Header()
    header["HIERARCH VERSION_DATAMODEL"] = "w.2026.40"
    header["HIERARCH VERSION_OBS_PFS"] = "w.2026.40"
    header["HIERARCH VERSION_DRP_STELLA"] = "w.2026.41-6-ge56938d2"
    fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU()]).writeto(path)
    butler = StubButler({(101, "r", 1): table(2.7)})
    butler.getURI = lambda ref: StubURI(path)
    versions = readPipelineVersions(butler, [101])
    assert versions[["versionDatamodel", "versionObsPfs", "versionDrpStella"]].values.tolist() == [
        ["w.2026.40", "w.2026.40", "w.2026.41-6-ge56938d2"]
    ]
