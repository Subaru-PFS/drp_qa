"""Tests for the qadb store (`pfs.drp.qa.dashboard.live.QadbMetricsStore`), with a stand-in for qadb."""

import pandas as pd
import pytest

from pfs.drp.qa.dashboard.live import QadbMetricsStore, qadbProvenance
from pfs.drp.qa.dashboard.qadbSource import writeSnapshot
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
    store = QadbMetricsStore(qadb.read, layers, {"collection": "c"}, minInterval=10, clock=clock)
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
    store = QadbMetricsStore(qadb.read, layers, {}, minInterval=0, clock=clock)
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
    store = QadbMetricsStore(qadb.read, layers, {}, minInterval=0, clock=clock)
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
    store = QadbMetricsStore(qadb.read, layers, {}, minInterval=0, clock=clock)
    assert not store.refresh() and not store.refresh()
    assert store.version == 1 and qadb.calls[-1] == "t1"

    # Negative control: an image loaded in the same instant, read only now, is new.
    qadb.loads.append([image(1001, "r", "WARN", 2.9, "t1")])
    assert store.refresh() and store.version == 2
    assert sorted(store.images()["arm"]) == ["b", "r"]
    assert not store.refresh()


def testSnapshotServesWhatQadbHeld(tmp_path, clock, layers):
    """A snapshot of a read gives the live store's table and the provenance written, and never refreshes."""
    qadb = FakeQadb()
    qadb.loads.append([image(1001, "b", "PASS", 2.5, "t1"), image(1002, "r", "FAIL", 9.0, "t1")])
    liveStore = QadbMetricsStore(qadb.read, layers, {}, minInterval=0, clock=clock)
    provenance = qadbProvenance("c (qadb snapshot)", builtAt="2026-10-28T10:00:00-10:00")
    store = QadbMetricsStore.fromSnapshot(writeSnapshot(qadb.read(None), tmp_path / "snap", provenance))

    pd.testing.assert_frame_equal(store.images(), liveStore.images())
    assert store.provenance == provenance
    assert provenance["thresholds"]["file"] == "iqQaThresholds-run25.yaml"
    assert not store.live and not store.refresh()
    # Negative control: the store reading qadb itself is live.
    assert liveStore.live


class FakeOpdb:
    """Answers ``readOpdb(visits)``: their times and sequences, and the notes on them and their sequences."""

    def __init__(self, sequences):
        self.sequences = sequences  # visit -> IIC sequence
        self.visitNotes = []  # (visit, camera, note, source)
        self.sequenceNotes = []  # (sequence, note)
        self.calls = []
        self.down = False

    def read(self, visits):
        self.calls.append(list(visits))
        if self.down:
            raise ConnectionError("opdb tunnel closed")
        times = pd.DataFrame(
            {
                "pfs_visit_id": visits,
                "time_exp_start": [
                    pd.Timestamp("2026-11-01 20:00") + pd.Timedelta(minutes=v % 100) for v in visits
                ],
                "iic_sequence_id": pd.array([self.sequences.get(v) for v in visits], dtype="Int64"),
            }
        )
        onVisits = [
            (v, pd.NA, camera, note, source) for v, camera, note, source in self.visitNotes if v in visits
        ]
        sequences = set(times["iic_sequence_id"].dropna())
        onSequences = [
            (pd.NA, s, None, note, "obslog_sequence") for s, note in self.sequenceNotes if s in sequences
        ]
        notes = pd.DataFrame(
            onVisits + onSequences, columns=["pfs_visit_id", "iic_sequence_id", "camera", "note", "source"]
        )
        notes = notes.astype({"pfs_visit_id": "Int64", "iic_sequence_id": "Int64"}).assign(
            data_flag=pd.array([pd.NA] * len(notes), dtype="Int64")
        )
        return times, notes


def testOpdbGivesSequencesTimesAndLateNotes(clock, layers):
    """The opdb's sequences and times replace qadb's, and a note written after the visit's load appears."""
    qadb = FakeQadb()
    qadb.loads.append([image(1001, "b", "PASS", 2.5, "t1"), image(1002, "b", "PASS", 2.5, "t1")])
    opdb = FakeOpdb({1001: 77, 1002: 78})
    store = QadbMetricsStore(qadb.read, layers, {}, minInterval=0, clock=clock, readOpdb=opdb.read)
    visits = store.visits().set_index("visit")
    assert (visits["sequenceSource"] == "opdb").all()
    assert visits.loc[1001, "sequenceKey"] == 77 and visits.loc[1002, "sequenceKey"] == 78
    assert visits.loc[1001, "timeExpStart"] == pd.Timestamp("2026-11-01 20:01")
    assert store.notes().empty and store.version == 1 and store.hasOpdb

    opdb.visitNotes.append((1001, "r1", "ghost", "sps_annotation"))
    opdb.sequenceNotes.append((78, "Dome light on"))
    assert store.refresh() and store.version == 2
    notes = store.notes()
    assert notes[["visit", "arm", "spectrograph", "note"]].values.tolist() == [
        [1001, "r", 1, "ghost"],
        [1002, "", 0, "Dome light on"],
    ]
    assert not store.refresh()

    # Negative control: without the opdb, one seqName is one inferred sequence, and there are no notes.
    bare = QadbMetricsStore(qadb.read, layers, {}, minInterval=0, clock=clock)
    visits = bare.visits().set_index("visit")
    assert (visits["sequenceSource"] == "inferred").all()
    assert visits.loc[1001, "sequenceKey"] == visits.loc[1002, "sequenceKey"]
    assert bare.notes().empty and not bare.hasOpdb


def testOpdbIsReadForRecentVisitsAfterTheFirst(clock, layers):
    qadb = FakeQadb()
    qadb.loads.append([image(v, "b", "PASS", 2.5, "t1") for v in (1001, 1002, 1003)])
    opdb = FakeOpdb({})
    store = QadbMetricsStore(
        qadb.read, layers, {}, minInterval=0, clock=clock, readOpdb=opdb.read, recentVisits=2
    )
    store.refresh()
    assert opdb.calls == [[1001, 1002, 1003], [1002, 1003]]


def testSnapshotKeepsTheOpdbNotes(tmp_path, clock, layers):
    qadb = FakeQadb()
    qadb.loads.append([image(1001, "b", "PASS", 2.5, "t1")])
    opdb = FakeOpdb({1001: 77})
    opdb.visitNotes.append((1001, None, "Cloudy", "obslog"))
    tables = dict(qadb.read(None))
    tables["opdbVisitTimes"], tables["notes"] = opdb.read([1001])
    store = QadbMetricsStore.fromSnapshot(writeSnapshot(tables, tmp_path / "snap", {"collection": "c"}))
    assert store.notes()["note"].tolist() == ["Cloudy"]
    assert store.visits()["sequenceSource"].tolist() == ["opdb"]


def testOpdbFailureKeepsTheNewImages(clock, layers):
    """A failed opdb read still shows the images qadb just gave; the opdb's answer comes on the next refresh."""
    qadb = FakeQadb()
    qadb.loads.append([image(1001, "b", "PASS", 2.5, "t1")])
    opdb = FakeOpdb({1001: 77, 1002: 78})
    store = QadbMetricsStore(qadb.read, layers, {}, minInterval=0, clock=clock, readOpdb=opdb.read)

    opdb.down = True
    qadb.loads.append([image(1002, "b", "FAIL", 9.0, "t2")])
    assert store.refresh()
    visits = store.visits().set_index("visit")
    assert visits.loc[1002, "sequenceSource"] == "inferred"
    assert store.images().set_index("visit").loc[1002, "status"] == "FAIL"

    opdb.down = False
    assert store.refresh()
    assert store.visits().set_index("visit").loc[1002, "sequenceKey"] == 78
