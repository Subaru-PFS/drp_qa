"""Tests for reading stored metrics from a Butler, against a stub."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas as pd
import pytest

from pfs.drp.qa.metrics.readers import (
    chooseRun,
    exportMetrics,
    readCalibValidity,
    readMetrics,
    readPipelineVersions,
    readReductionPipelines,
)


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
        if "visits" in bind:
            wanted = set(bind["visits"]).__contains__
        else:
            wanted = range(bind["firstVisit"], bind["lastVisit"] + 1).__contains__
        return [
            StubRef(
                datasetType, {"instrument": bind["instrumentName"], "visit": v, "arm": a, "spectrograph": s}
            )
            for (v, a, s) in self.tables
            if wanted(v)
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


def testVisitRange(butler):
    """Every visit of a range, e.g. a whole run, bound as its ends."""
    metrics = readMetrics(butler, visitRange=(100, 500))
    assert sorted(set(metrics["visit"])) == [100, 101]
    (query,) = butler.queries
    assert query["bind"] == {"instrumentName": "PFS", "firstVisit": 100, "lastVisit": 500}
    assert "100" not in query["where"]
    # Negative control: the range's ends are inclusive, and 999 lies beyond it.
    assert 999 in set(readMetrics(butler, visitRange=(100, 999))["visit"])


def testVisitsOrRange(butler):
    with pytest.raises(ValueError, match="one of visits and visitRange"):
        readMetrics(butler)
    with pytest.raises(ValueError, match="one of visits and visitRange"):
        readMetrics(butler, [100], visitRange=(100, 101))


def testProgress(butler, capsys):
    readMetrics(butler, [100, 101], progress=True)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "iqQaMetrics: reading 3 datasets for 2 visits (100-101)"
    assert lines[-1].startswith("iqQaMetrics: 3/3 ") and "ETA 0 s" in lines[-1]
    # Negative control: quiet by default.
    readMetrics(butler, [100, 101])
    assert capsys.readouterr().out == ""


def testExportMetrics(butler, tmp_path):
    paths = exportMetrics(
        butler, "u/someone/qa/008", ["iqQaMetrics", "dmDriftSummary"], tmp_path / "out", visits=[100, 101]
    )
    assert [path.name for path in paths] == [
        "iqQaMetrics-u_someone_qa_008.parquet",
        "dmDriftSummary-u_someone_qa_008.parquet",
    ]
    written = pd.read_parquet(paths[0])
    assert written[["visit", "arm", "spectrograph"]].values.tolist() == [
        [100, "b", 1],
        [100, "b", 2],
        [101, "r", 1],
    ]
    assert [query["collections"] for query in butler.queries] == [["u/someone/qa/008"]] * 2


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


class RunButler:
    """Holds pfsArms in RUN collections; a chain is searched first run first, as find_first does."""

    def __init__(self, runs: dict[str, dict[tuple, str]]):
        self.runs = runs  # run -> {(visit, arm, spectrograph): drp_stella version}
        self.queries = []

    def query_datasets(self, datasetType, collections=None, where="", bind=None, **kwargs):
        self.queries.append(list(collections))
        found = {}
        for run in collections:
            for key in self.runs.get(run, {}):
                if key[0] in set(bind["visits"]) and key not in found:
                    found[key] = run
        return [
            StubRef(datasetType, {"visit": v, "arm": a, "spectrograph": s}, run)
            for (v, a, s), run in found.items()
        ]

    def getURI(self, ref):
        dataId = ref.dataId
        return self.runs[ref.run][(dataId["visit"], dataId["arm"], dataId["spectrograph"])]


def testReductionPipelinesComeFromTheCalexpRun():
    """A later reduction of the detector, first in the chain, doesn't stand in for the one judged."""
    butler = RunButler(
        {
            "PFS/new": {(1, "b", 1): "w.2026.41"},  # reduced again after the verdict
            "PFS/old": {(1, "b", 1): "w.2026.38", (1, "r", 1): "w.2026.38"},
        }
    )
    readHeader = lambda version: {"VERSION_DRP_STELLA": version}  # noqa: E731
    metrics = pd.DataFrame(
        {
            "visit": [1, 1, 1],
            "arm": ["r", "b", "b"],
            "spectrograph": 1,
            "calexpRun": ["PFS/old", "PFS/old", None],
        },
        index=[7, 7, 3],
    )
    versions = readReductionPipelines(butler, metrics, readHeader=readHeader)
    assert versions.index.tolist() == [7, 7, 3]
    assert versions["versionDrpStella"].fillna("").tolist() == ["w.2026.38", "w.2026.38", ""]
    assert butler.queries == [["PFS/old"]]
    # The fallback searches a chain, so it finds the latest: only right for a single reduction.
    fromChain = readReductionPipelines(
        butler, metrics, fallback=["PFS/new", "PFS/old"], readHeader=readHeader
    )
    assert fromChain["versionDrpStella"].tolist() == ["w.2026.38", "w.2026.38", "w.2026.41"]
    # Metrics written before calexpRun: unknown, and nothing is read.
    butler.queries.clear()
    assert (
        readReductionPipelines(butler, metrics.drop(columns="calexpRun"), readHeader=readHeader)
        .isna()
        .all()
        .all()
    )
    assert butler.queries == []


class StubTime:
    def __init__(self, isot: str):
        self.tai = SimpleNamespace(isot=isot)


class StubRegistry:
    """Answers ``queryDatasetAssociations`` from a list of associations."""

    def __init__(self, associations):
        self.associations = associations
        self.queries = []

    def queryDatasetAssociations(self, datasetType, collections=None, flattenChains=False):
        self.queries.append((datasetType, collections, flattenChains))
        return iter(self.associations)


def association(calibId, collection, begin, end):
    timespan = (
        None
        if begin == "run"
        else SimpleNamespace(begin=begin and StubTime(begin), end=end and StubTime(end))
    )
    return SimpleNamespace(ref=SimpleNamespace(id=calibId), collection=collection, timespan=timespan)


def testReadCalibValidity():
    registry = StubRegistry(
        [
            association("a", "PFS/calib", "2026-01-01T00:00:00.000", "2026-06-01T00:00:00.000"),
            association("a", "PFS/calib/old", "2025-01-01T00:00:00.000", None),
            association("b", "PFS/calib", "2026-06-01T00:00:00.000", None),
            association("a", "PFS/calib/run", "run", None),  # a RUN collection: no timespan
        ]
    )
    validity = readCalibValidity(SimpleNamespace(registry=registry), ["a"], "PFS/calib")
    assert validity.to_dict("records") == [
        {
            "calibId": "a",
            "collection": "PFS/calib/old",
            "validFrom": "2025-01-01T00:00:00.000",
            "validTo": None,
        },
        {
            "calibId": "a",
            "collection": "PFS/calib",
            "validFrom": "2026-01-01T00:00:00.000",
            "validTo": "2026-06-01T00:00:00.000",
        },
    ]
    assert registry.queries == [("detectorMap_calib", "PFS/calib", True)]
    assert readCalibValidity(SimpleNamespace(registry=registry), [], "PFS/calib").empty
