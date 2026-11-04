"""Tests for storing a per-image table in a Butler after the fact (INSTRM-3089), against a stub Butler."""

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
            registerDatasetType=self.datasetTypes.append, registerRun=self.runs.append
        )
        self.collections = SimpleNamespace(
            get_info=lambda name: SimpleNamespace(type=SimpleNamespace(name=collectionType)),
            prepend_chain=lambda parent, children: self.chains.append((parent, list(children))),
        )

    def put(self, obj, datasetType, dataId=None, run=None):
        self.puts.append((datasetType, dict(dataId), run, obj))


@pytest.fixture(autouse=True)
def stackStub(monkeypatch):
    """`lsst.daf.butler.DatasetType` as a plain record."""
    module = types.ModuleType("lsst.daf.butler")

    class DatasetType:
        def __init__(self, name, dimensions, storageClass, universe=None):
            self.name, self.dimensions, self.storageClass = name, dimensions, storageClass

    module.DatasetType = DatasetType
    monkeypatch.setitem(sys.modules, "lsst", types.ModuleType("lsst"))
    monkeypatch.setitem(sys.modules, "lsst.daf", types.ModuleType("lsst.daf"))
    monkeypatch.setitem(sys.modules, "lsst.daf.butler", module)


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
    (registered,) = butler.datasetTypes
    assert registered.name == "twilightQaMetrics" and registered.storageClass == "DataFrame"
    assert registered.dimensions == IMAGE_DIMENSIONS
    assert len(butler.puts) == 6
    dataIds = [dataId for _, dataId, _, _ in butler.puts]
    assert dataIds[0] == {"instrument": "PFS", "visit": 1001, "arm": "b", "spectrograph": 1}
    assert all(run == "c/t/run1" and name == "twilightQaMetrics" for name, _, run, _ in butler.puts)
    assert all(len(obj) == 1 and obj.index[0] == 0 for *_, obj in butler.puts)  # one row, reindexed
    assert butler.chains == [("qaActor/reductions", ["c/t/run1"])]


def testDefaultRunIsNamedForTheCollectionAndType():
    butler = StubButler()
    run = storeImageTable(butler, "c", "twilightQaMetrics", table(), log=lambda _: None)
    assert run.startswith("c/twilightQaMetrics/") and run.endswith("Z")


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
    assert len(good.puts) == 6
