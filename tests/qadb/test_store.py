"""Tests for storing a per-image table in a Butler after the fact, against a stub Butler."""

import sys
import types
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from pfs.drp.qa.qadb.store import IMAGE_DIMENSIONS, runName, storeImageTable


class StubButler:
    """Records registrations, puts and the chain; a CHAINED collection unless told otherwise."""

    def __init__(self, collectionType="CHAINED"):
        self.puts = []
        self.datasetTypes = []
        self.runs = []
        self.chains = []
        self.dimensions = SimpleNamespace(conform=lambda names: tuple(names))
        self.registry = SimpleNamespace(
            registerDatasetType=self.datasetTypes.append, registerRun=self.register
        )
        self.taken = set()
        self.collections = SimpleNamespace(
            get_info=lambda name: SimpleNamespace(type=SimpleNamespace(name=collectionType)),
            prepend_chain=lambda parent, children: self.chains.append((parent, list(children))),
        )

    def register(self, run):
        """Say whether the RUN was new, as `Registry.registerRun` does."""
        self.runs.append(run)
        if run in self.taken:
            return False
        self.taken.add(run)
        return True

    def put(self, obj, datasetType, dataId=None, run=None):
        self.puts.append((datasetType, dict(dataId or {}), run, obj))


@pytest.fixture(autouse=True)
def stackStub(monkeypatch):
    """`lsst.daf.butler.DatasetType` as a plain record."""
    module = types.ModuleType("lsst.daf.butler")

    class DatasetType:
        def __init__(self, name, dimensions, storageClass, universe=None):
            self.name, self.dimensions, self.storageClass = name, dimensions, storageClass

    module.DatasetType = DatasetType
    packages = types.ModuleType("lsst.utils.packages")

    class Packages(dict):
        @classmethod
        def fromSystem(cls):
            return cls({"drp_qa": "w.test", "drp_stella": "w.test"})

    packages.Packages = Packages
    monkeypatch.setitem(sys.modules, "lsst", types.ModuleType("lsst"))
    monkeypatch.setitem(sys.modules, "lsst.daf", types.ModuleType("lsst.daf"))
    monkeypatch.setitem(sys.modules, "lsst.daf.butler", module)
    monkeypatch.setitem(sys.modules, "lsst.utils", types.ModuleType("lsst.utils"))
    monkeypatch.setitem(sys.modules, "lsst.utils.packages", packages)


def table():
    rows = []
    for visit in (1001, 1002):
        for arm, spectrograph in (("b", 1), ("r", 1), ("b", 2)):
            rows.append(
                {"visit": visit, "arm": arm, "spectrograph": spectrograph, "qaStatus": "PASS", "x": 1.0}
            )
    return pd.DataFrame(rows)


def testRunName():
    when = datetime(2026, 11, 4, 8, 30, 0, tzinfo=UTC)
    assert runName("qaActor/reductions", "twilightQaMetrics", when) == (
        "qaActor/reductions/twilightQaMetrics/20261104T083000Z"
    )
    assert runName("c", "t").startswith("c/t/")


def testOnePutPerImageInANewRunPrependedToTheChain():
    butler = StubButler()
    run = storeImageTable(
        butler, "qaActor/reductions", "twilightQaMetrics", table(), run="c/t/run1", log=lambda _: None
    )
    assert run == "c/t/run1"
    assert butler.runs == ["c/t/run1"]
    registered, packages = butler.datasetTypes
    assert registered.name == "twilightQaMetrics" and registered.storageClass == "DataFrame"
    assert registered.dimensions == IMAGE_DIMENSIONS
    assert packages.name == "packages" and packages.storageClass == "Packages"
    images = [put for put in butler.puts if put[0] == "twilightQaMetrics"]
    assert len(images) == 6
    dataIds = [dataId for _, dataId, _, _ in images]
    assert dataIds[0] == {"instrument": "PFS", "visit": 1001, "arm": "b", "spectrograph": 1}
    assert all(run == "c/t/run1" for _, _, run, _ in butler.puts)
    assert all(len(obj) == 1 and obj.index[0] == 0 for *_, obj in images)  # one row, reindexed
    # The judging software's versions go in the RUN too, for the loader's drpQaVersionOf.
    (versions,) = [put for put in butler.puts if put[0] == "packages"]
    assert versions[3]["drp_qa"] == "w.test"
    assert butler.chains == [("qaActor/reductions", ["c/t/run1"])]


def testDefaultRunIsNamedForTheCollectionAndType():
    butler = StubButler()
    run = storeImageTable(butler, "c", "twilightQaMetrics", table(), log=lambda _: None)
    assert run.startswith("c/twilightQaMetrics/") and run.endswith("Z")


def testAGeneratedRunNameTakenInTheSameSecondGetsASuffix():
    """Two judges of one sequence in one second must not put the same images into one RUN."""
    butler = StubButler()
    first = storeImageTable(butler, "c", "twilightQaMetrics", table(), run=None, log=lambda _: None)
    butler.taken.add(first)  # as a concurrent store that took the same timestamp would
    second = storeImageTable(butler, "c", "twilightQaMetrics", table(), log=lambda _: None)
    if second == first:  # the clock moved on between the two: force the collision instead
        butler = StubButler()
        butler.taken.add("c/t/run1")
        second = storeImageTable(butler, "c", "twilightQaMetrics", table(), run=None, log=lambda _: None)
    assert second != first or second.endswith("-1")
    # Negative control: an explicit RUN that exists is refused, not renamed.
    butler = StubButler()
    butler.taken.add("c/t/run1")
    with pytest.raises(ValueError, match="already exists"):
        storeImageTable(butler, "c", "twilightQaMetrics", table(), run="c/t/run1", log=lambda _: None)


def testPackagesFailureIsLoggedNotRaised(monkeypatch):
    butler = StubButler()
    lines = []
    monkeypatch.delitem(sys.modules, "lsst.utils.packages")
    run = storeImageTable(butler, "c", "twilightQaMetrics", table(), run="c/t/run1", log=lines.append)
    assert run == "c/t/run1" and butler.chains == [("c", ["c/t/run1"])]
    assert any("packages: not recorded" in line for line in lines)
    assert not [put for put in butler.puts if put[0] == "packages"]


def testPackagesCanBeSkipped():
    butler = StubButler()
    storeImageTable(
        butler, "c", "twilightQaMetrics", table(), run="c/t/run1", packages=False, log=lambda _: None
    )
    assert not [put for put in butler.puts if put[0] == "packages"]


def testARunCollectionIsRefused():
    """Nothing is written where the collection's readers could not find it."""
    butler = StubButler(collectionType="RUN")
    with pytest.raises(ValueError, match="not CHAINED"):
        storeImageTable(butler, "u/test/run", "twilightQaMetrics", table(), log=lambda _: None)
    assert butler.puts == [] and butler.runs == [] and butler.chains == []
    # Negative control: the same table into a CHAINED collection is written.
    assert len(StubButler().puts) == 0
    good = StubButler()
    storeImageTable(good, "u/test/chain", "twilightQaMetrics", table(), log=lambda _: None)
    assert len([put for put in good.puts if put[0] == "twilightQaMetrics"]) == 6
