"""Every known fault reaches its verdict, and the frames known to be good are not failed.

The numbers are stored ones of calibration and engineering visits
(``data/makeGateFixtures.py``, ``makeDriftFixture.py``, ``makeTwilightFixture.py``).
The normal values are made here as a run's are (`pfs.drp.qa.metrics.normalValues`):

- from Run25's stable calibration set, valid from its first visit;
- from Run29's quartz, valid from Run29, where the b2 slit had been refocused: its width
  is refused until it is accepted with that reason;
- from Run30's nightly frames lit through the engineering fibers, for those frames.

Run27's frames fall between: they are judged by Run25's values, as a run is judged by
the previous one's until its own are accepted.
"""

from pathlib import Path

import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import gate, judge, longLines, measurements
from pfs.drp.qa.metrics.normalValues import accepted, candidates, compare

DATA = Path(__file__).parent / "data"

#: First and last visit of the calibration frames each set of normal values is made from.
RUN25, RUN29, RUN30 = (133025, 135850), (144587, 147830), (148039, 150896)

#: The faults: what, visits (first, last), arms, spectrographs, check, metric, the verdict every image with the
#: number measured must reach, and at least how many images that is.
FAULTS = [
    # Run27: SM1 out of focus. Lines and traces are wide, and scatter about the visit's own solution.
    ("SM1 defocus, arcs", (140005, 140138), "brn", (1,), "width", "medFwhm", "FAIL", 40),
    ("SM1 defocus, quartz", (140032, 140148), "brn", (1,), "width", "medFwhm", "FAIL", 30),
    ("SM1 defocus, quartz traces", (140139, 140148), "brn", (1,), "scatterX", "robustRmsDeltaX", "FAIL", 30),
    ("SM1 defocus, arc lines", (140130, 140135), "brn", (1,), "scatterX", "robustRmsDeltaX", "FAIL", 15),
    # Run27 was reduced against a calibration built later: SM3 sits +1.2 px off in x, SM2 -3.2 px in y.
    ("SM3 offset in x", (139971, 140148), "brn", (3,), "calibOffset", "medianDeltaX", "WARN", 80),
    ("SM2 offset in y", (139971, 140138), "br", (2,), "calibOffset", "medianDeltaY", "FAIL", 30),
    # Run30: fibers obstructed, so each lit fiber measures fewer lines and rows.
    (
        "obstructed, arcs",
        (150115, 150115),
        "brn",
        (1, 2, 3, 4),
        "linesPerFiberArc",
        "linesPerFiber",
        "FAIL",
        12,
    ),
    (
        "obstructed, arcs",
        (150641, 150641),
        "brn",
        (1, 2, 3, 4),
        "linesPerFiberArc",
        "linesPerFiber",
        "FAIL",
        12,
    ),
    (
        "obstructed, quartz",
        (150116, 150116),
        "brn",
        (1, 2, 3, 4),
        "linesPerFiberQuartz",
        "linesPerFiber",
        "FAIL",
        12,
    ),
    (
        "obstructed, quartz",
        (150642, 150642),
        "brn",
        (1, 2, 3, 4),
        "linesPerFiberQuartz",
        "linesPerFiber",
        "FAIL",
        12,
    ),
    # Run21 twilights: one detector's level steps against the sequence's other exposures.
    ("twilight level, b1", (121949, 121952), "b", (1,), "twilightLevel", "twilightLevelOffset", "FAIL", 4),
    ("twilight level, b4", (123320, 123320), "b", (4,), "twilightLevel", "twilightLevelOffset", "FAIL", 1),
    # Run30: n4's lamp header read off on an engineering-fiber arc, reduced against the wrong line list.
    ("trace centre, n4", (148829, 148829), "n", (4,), "traceCentreOffset", "medDxCenter", "WARN", 1),
]


def _between(table, visits):
    return table[table["visit"].between(*visits)]


@pytest.fixture(scope="module")
def numbers():
    """Return the stored numbers: ``iqQaMetrics``, the line residuals and the twilight metrics."""
    iq = pd.concat(
        [
            pd.read_parquet(DATA / "iqQaMetrics-validation.parquet"),
            pd.read_parquet(DATA / "iqQaMetrics-iis-run30.parquet"),
        ],
        ignore_index=True,
    )
    drift = pd.read_parquet(DATA / "dmDriftMetrics-validation.parquet")
    keep = ["visit", "arm", "spectrograph", "cameraSet", "obsType", "reference", "description", "statusType"]
    metrics = ["medianDeltaX", "robustRmsDeltaX", "medianDeltaY", "robustRmsDeltaY"]
    return iq, longLines(drift[[*keep, *metrics]]), pd.read_parquet(DATA / "twilightMetrics.parquet")


@pytest.fixture(scope="module")
def normalValues(numbers):
    """Return the normal values of the three runs, made and accepted in turn."""
    iq, lines, _ = numbers
    full, iis = iq[iq["illumination"] != "iis"], iq[iq["illumination"] == "iis"]
    stored = []
    for visits, frames, reason in ((RUN25, full, ""), (RUN29, full, "b2 slit refocused"), (RUN30, iis, "")):
        inForce = pd.concat(stored, ignore_index=True) if stored else None
        compared = compare(candidates(_between(frames, visits), _between(lines, visits)), inForce, visits[0])
        stored.append(accepted(compared, "test", visits[0], "test", reason=reason)[0])
    return pd.concat(stored, ignore_index=True)


@pytest.fixture(scope="module")
def judged(numbers, normalValues):
    return judge(measurements(*numbers), normalValues)


@pytest.fixture(scope="module")
def verdicts(numbers, normalValues):
    return gate(*numbers, normalValues=normalValues)


@pytest.mark.parametrize("fault", FAULTS, ids=[f"{fault[0]} {fault[1][0]}" for fault in FAULTS])
def testFaultReachesItsVerdict(judged, verdicts, fault):
    _, visits, arms, spectrographs, check, metric, expect, atLeast = fault
    rows = _between(judged, visits)
    rows = rows[rows["arm"].isin(list(arms)) & rows["spectrograph"].isin(spectrographs)]
    rows = rows[(rows["check"] == check) & (rows["metric"] == metric)]
    reached = ("FAIL",) if expect == "FAIL" else ("WARN", "FAIL")
    images = rows.groupby(["visit", "arm", "spectrograph"])["status"].agg(lambda s: s.isin(reached).any())
    # The check runs on these images, and gives every one its verdict.
    assert len(images) >= atLeast, f"only {len(images)} images judged"
    assert images.all(), images[~images]
    # And the image's verdict is at least that.
    image = verdicts.merge(images.reset_index()[["visit", "arm", "spectrograph"]])
    assert len(image) == len(images) and image["qaStatus"].isin(reached).all()


def testLooserLimitsMissTheDefocus(numbers, normalValues):
    """Negative control: the SM1 defocus is caught by the width's tolerance, not by any tolerance."""
    widths = normalValues["metric"] == "medFwhm"
    loose = normalValues.assign(value=normalValues["value"].where(~widths, normalValues["value"] * 3))
    rows = judge(measurements(*numbers), loose)
    rows = _between(rows, (140005, 140148))
    rows = rows[(rows["spectrograph"] == 1) & (rows["check"] == "width")]
    assert len(rows) > 70 and (rows["status"] == "PASS").all()


def testRefusedUntilAccepted(numbers):
    """Run29's b2 quartz is 5 % wider than Run25's: its value is refused without a reason, and kept with one."""
    iq, lines, _ = numbers
    full = iq[iq["illumination"] != "iis"]
    run25 = accepted(
        compare(candidates(_between(full, RUN25), _between(lines, RUN25)), None, RUN25[0]),
        "test",
        RUN25[0],
        "test",
    )[0]
    compared = compare(candidates(_between(full, RUN29), _between(lines, RUN29)), run25, RUN29[0])
    failed = compared[compared["status"] == "FAIL"]
    assert failed[["metric", "camera", "frameType"]].to_dict("records") == [
        {"metric": "medFwhm", "camera": "b2", "frameType": "quartz"}
    ]
    assert failed["shift"].iloc[0] == pytest.approx(5.25, abs=0.01)
    stored, refused = accepted(compared, "test", RUN29[0], "test")
    assert len(refused) == 1 and len(stored) == len(compared) - 1
    # Without it, Run29's b2 quartz is judged by Run25's value and fails.
    quartz = _between(full, RUN29)
    b2 = (quartz["arm"] == "b") & (quartz["spectrograph"] == 2)
    without = gate(iq=quartz[b2], normalValues=pd.concat([run25, stored]))
    assert (without["qaStatus"] == "FAIL").sum() > 10
    kept = accepted(compared, "test", RUN29[0], "test", reason="b2 slit refocused")[0]
    assert not (gate(iq=quartz[b2], normalValues=pd.concat([run25, kept]))["qaStatus"] == "FAIL").any()


class TestKnownGood:
    def testRun25StableSet(self, verdicts):
        """Run25's arcs and quartz: none fails, and the warnings are these."""
        lamp = _between(verdicts, RUN25)
        lamp = lamp[lamp["frameType"].isin(["arc", "quartz"])]
        assert len(lamp) == 616 and not (lamp["qaStatus"] == "FAIL").any()
        assert lamp["qaStatus"].value_counts().to_dict() == {"PASS": 563, "WARN": 53}
        warned = lamp[lamp["qaStatus"] == "WARN"]
        # SM3 after its calibration block sits 1.1 to 1.5 px from the calibration in y; three b-arm Xenon
        # arcs scatter in x; one m1 Argon arc is 3.3 % wide.
        assert warned["qaDecidedBy"].value_counts().to_dict() == {
            "calibOffset": 49,
            "scatterX": 3,
            "width": 1,
        }
        offset = warned[warned["qaDecidedBy"] == "calibOffset"]
        assert set(offset["spectrograph"]) == {3} and offset["visit"].min() >= 135830

    def testWidthOfTheStableSet(self, judged):
        """Every width is judged against its own camera and lamp, but b-arm Argon's, which is not judged."""
        rows = _between(judged, RUN25)
        rows = rows[rows["check"] == "width"]
        assert rows["status"].value_counts().to_dict() == {"PASS": 575, "WARN": 1}
        assert rows["normal"].notna().all()
        assert not ((rows["arm"] == "b") & (rows["selector"] == "Argon")).any()
        warned = rows[rows["status"] == "WARN"].iloc[0]
        assert (warned["visit"], warned["arm"], warned["spectrograph"]) == (135841, "m", 1)

    def testRun29Quartz(self, verdicts):
        quartz = _between(verdicts, RUN29)
        quartz = quartz[quartz["frameType"] == "quartz"]
        assert len(quartz) == 272 and (quartz["qaStatus"] == "PASS").all()

    def testEngineeringFiberFrames(self, verdicts):
        """Run30's nightly engineering-fiber frames, judged by their own normal values."""
        frames = verdicts[verdicts["frameType"].isin(["iisArc", "iisQuartz"])]
        assert len(frames) == 540 and not frames["frameType"].isna().any()
        assert frames["qaStatus"].value_counts().to_dict() == {"PASS": 532, "WARN": 7, "FAIL": 1}
        flagged = frames[frames["qaStatus"] != "PASS"]
        # The trace centre is the known fault. The rest are widths: n-arm Ne arcs, whose 15 faint lines per
        # fiber give a width that scatters 2 to 4 % from night to night, and b2 on the night SM2 moved.
        assert flagged["qaDecidedBy"].value_counts().to_dict() == {"width": 7, "traceCentreOffset": 1}
        widths = flagged[flagged["qaDecidedBy"] == "width"]
        assert widths["arm"].value_counts().to_dict() == {"n": 6, "b": 1}
        # No unknown: every one of these frames has normal values of its own.
        assert not (frames["qaStatus"] == "UNKNOWN").any()

    def testTwilights(self, verdicts):
        """Twilights but the Run21 faults and one Run29 sequence pass; the n arm is not judged."""
        twilight = verdicts[verdicts["frameType"] == "twilight"]
        n = twilight["arm"] == "n"
        assert n.sum() > 200 and (twilight.loc[n, "qaStatus"] == "UNKNOWN").all()
        judgedArms = twilight[~n]
        assert not (judgedArms["qaStatus"] == "UNKNOWN").any()
        failed = judgedArms[judgedArms["qaStatus"] == "FAIL"]
        assert sorted(set(failed["visit"])) == [121949, 121950, 121951, 121952, 123320, 145073]
        assert (judgedArms["qaStatus"] == "WARN").sum() == 1


def testEveryImageHasOneVerdict(numbers, verdicts):
    iq, lines, twilight = numbers
    keys = ["visit", "arm", "spectrograph"]
    images = pd.concat([iq[keys], lines[keys], twilight[keys]]).drop_duplicates()
    assert len(verdicts) == len(images) and not verdicts.duplicated(keys).any()
    assert set(verdicts["qaStatus"]) == {"PASS", "WARN", "FAIL", "UNKNOWN"}
