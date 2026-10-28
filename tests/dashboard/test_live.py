"""Tests for the live Tier 1 store (`pfs.drp.qa.dashboard.live.QadbTier1Store`), with a stand-in for qadb."""

import pandas as pd
import pytest

from pfs.drp.qa.dashboard.live import QadbTier1Store
from pfs.drp.qa.dashboard.serving import taskThresholds


def image(visit, arm, status, medFwhm, decidedAt, seqName="Arc: Neon"):
    return {
        "visit": visit,
        "arm": arm,
        "spectrograph": 1,
        "obsType": "arc",
        "seqName": seqName,
        "illumination": "full",
        "traceOnly": False,
        "medFwhm": medFwhm,
        # The other metrics the thresholds name: not measured here.
        **dict.fromkeys(
            ("medDxCenter", "dxCenterRms", "pctFlagged", "linesPerFiber", "nLines"), float("nan")
        ),
        "qaStatus": status,
        "qaReason": "" if status == "PASS" else "medFwhm",
        "calexpRun": "",
        "decidedAt": decidedAt,
    }


def tables(images):
    """Return what `readQadb` returns for some images: no line residuals."""
    iq = pd.DataFrame(images)
    visits = sorted(set(iq["visit"])) if len(iq) else []
    return {
        "iqQaMetrics": iq,
        "dmLineResidualSummary": pd.DataFrame(columns=["visit", "arm", "spectrograph", "decidedAt"]),
        "dmLineResidualMetrics": pd.DataFrame(columns=["visit", "arm", "spectrograph", "decidedAt"]),
        "visitTimes": pd.DataFrame(
            {
                "pfs_visit_id": visits,
                "time_exp_start": [
                    pd.Timestamp("2026-11-01 19:00") + pd.Timedelta(minutes=v % 100) for v in visits
                ],
                "iic_sequence_id": pd.array([pd.NA] * len(visits), dtype="Int64"),
            }
        ),
    }


class FakeQadb:
    """Answers ``read(since)`` from loads, each a list of images at one time."""

    def __init__(self):
        self.loads = []
        self.calls = []

    def read(self, since):
        self.calls.append(since)
        # As qadb holds them: each image's latest load.
        latest = {(row["visit"], row["arm"], row["spectrograph"]): row for load in self.loads for row in load}
        return tables([row for row in latest.values() if since is None or row["decidedAt"] >= since])


@pytest.fixture
def clock():
    now = [0.0]
    tick = lambda: now[0]  # noqa: E731
    tick.advance = lambda seconds: now.__setitem__(0, now[0] + seconds)
    return tick


@pytest.fixture
def layers():
    return taskThresholds()[0]


def testRefreshReplacesRejudgedImagesAndAddsNewOnes(clock, layers):
    qadb = FakeQadb()
    qadb.loads.append([image(1001, "b", "PASS", 2.5, "t1"), image(1001, "r", "PASS", 2.7, "t1")])
    store = QadbTier1Store(qadb.read, layers, {"collection": "c"}, minInterval=10, clock=clock)
    assert store.version == 1 and store.since == "t1"
    assert store.images()["status"].tolist() == ["PASS", "PASS"]

    # Within minInterval nothing is read.
    qadb.loads.append([image(1002, "b", "PASS", 2.5, "t2")])
    assert not store.refresh() and len(qadb.calls) == 1

    clock.advance(10)
    qadb.loads.append([image(1001, "b", "FAIL", 9.0, "t3")])
    assert store.refresh()
    assert qadb.calls[-1] == "t1"
    images = store.images().set_index(["visit", "arm"])
    assert len(images) == 3
    assert images.loc[(1001, "b"), "status"] == "FAIL"  # replaced, not duplicated
    assert images.loc[(1001, "r"), "status"] == "PASS"  # kept
    assert store.since == "t3"
    fwhm = store.trend("iqQaMetrics", "medFwhm")
    assert len(fwhm) == 3 and fwhm.set_index(["visit", "arm"]).loc[(1001, "b"), "value"] == 9.0


def testSequencesAreNumberedOverTheWholeTable(clock, layers):
    """Visits read apart but of one sequence share its key; another seqName starts a new one."""
    qadb = FakeQadb()
    qadb.loads.append([image(1001, "b", "PASS", 2.5, "t1")])
    store = QadbTier1Store(qadb.read, layers, {}, minInterval=0, clock=clock)
    qadb.loads.append(
        [image(1002, "b", "PASS", 2.5, "t2"), image(1003, "b", "PASS", 3.0, "t2", seqName="Trace")]
    )
    store.refresh()
    visits = store.visits().set_index("visit")
    assert visits.loc[1001, "sequenceKey"] == visits.loc[1002, "sequenceKey"]
    # Negative control: a new seqName is a new sequence.
    assert visits.loc[1003, "sequenceKey"] != visits.loc[1002, "sequenceKey"]
    assert visits["timeExpStart"].notna().all()


def testEmptyUntilTheFirstLoad(clock, layers):
    qadb = FakeQadb()
    store = QadbTier1Store(qadb.read, layers, {}, minInterval=0, clock=clock)
    assert store.version == 0 and store.visits().empty and store.since is None
    assert not store.refresh()
    qadb.loads.append([image(1001, "b", "WARN", 2.9, "t1")])
    assert store.refresh() and store.images()["status"].tolist() == ["WARN"]


def testTaskThresholdsIncludeTheIisLayer():
    """The IIS file, imageQualityQa's default extraThresholdsFiles, is a layer, as the task judges."""
    layers, sources = taskThresholds()
    files = [layer.name for layer in layers["iqQaMetrics"] if hasattr(layer, "name")]
    assert files[:2] == ["iqQaThresholds-run25.yaml", "iqQaThresholds-iis-run30.yaml"]
    assert "iqQaThresholds-iis-run30.yaml" in sources["iqQaMetrics"]


def testUnchangedRefreshIsNotNew(clock, layers):
    """``since`` is inclusive, so the last load comes back on every read: that is not new data."""
    qadb = FakeQadb()
    qadb.loads.append([image(1001, "b", "PASS", 2.5, "t1")])
    store = QadbTier1Store(qadb.read, layers, {}, minInterval=0, clock=clock)
    assert not store.refresh() and not store.refresh()
    assert store.version == 1 and qadb.calls[-1] == "t1"

    # Negative control: an image loaded in the same instant, read only now, is new.
    qadb.loads.append([image(1001, "r", "WARN", 2.9, "t1")])
    assert store.refresh() and store.version == 2
    assert sorted(store.images()["arm"]) == ["b", "r"]
    assert not store.refresh()
