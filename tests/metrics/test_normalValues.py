"""Normal values: made from calibration frames, compared with those in force, accepted or refused."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import gate, judge, measurements
from pfs.drp.qa.metrics.normalValues import (
    CANDIDATE_COLUMNS,
    NORMAL_COLUMNS,
    accepted,
    candidates,
    compare,
    fixedFailures,
    formatComparison,
    visitRanges,
)
from pfs.drp.qa.metrics.registry import MIN_NORMAL_FRAMES


def iqRows(widths, arm="r", spectrograph=1, firstVisit=100, seqName="Arc: Neon", obsType="arc", **values):
    """Return ``iqQaMetrics`` rows, one visit per width."""
    return pd.DataFrame(
        {
            "visit": np.arange(firstVisit, firstVisit + len(widths)),
            "arm": arm,
            "spectrograph": spectrograph,
            "obsType": obsType,
            "illumination": "full",
            "seqName": seqName,
            "traceOnly": False,
            "medFwhm": widths,
            "linesPerFiber": 80.0,
        }
        | values
    )


def offsetRows(values, arm, cameraSet, metric, description, obsType, firstVisit=100, spectrograph=1):
    """Return ``dmLineResidualMetrics`` rows against the calibration, one visit per value."""
    return pd.DataFrame(
        {
            "visit": np.arange(firstVisit, firstVisit + len(values)),
            "arm": arm,
            "spectrograph": spectrograph,
            "cameraSet": cameraSet,
            "obsType": obsType,
            "reference": "calib",
            "description": description,
            "statusType": "BOTH",
            "metric": metric,
            "value": values,
        }
    )


@pytest.mark.parametrize(
    "visits, text",
    [
        ([], ""),
        ([5], "5"),
        ([3, 1, 2], "1-3"),
        ([1, 2, 4, 7, 8, 8], "1-2,4,7-8"),
        (np.array([10, 12]), "10,12"),
    ],
)
def testVisitRanges(visits, text):
    assert visitRanges(visits) == text


class TestCandidates:
    def testMedianPerCameraFrameTypeAndLamp(self):
        iq = pd.concat(
            [
                iqRows([2.0, 2.2, 2.1, 9.0]),
                iqRows([3.0, 3.1, 3.2], spectrograph=2),
                iqRows([2.5, 2.6, 2.7], seqName="Arc: Argon", firstVisit=200),
                iqRows([3.3, 3.4, 3.5], seqName="Trace", obsType="trace", firstVisit=300),
            ]
        )
        made = candidates(iq)
        assert list(made.columns) == list(CANDIDATE_COLUMNS)
        width = made[made["metric"] == "medFwhm"].set_index(["camera", "frameType", "selector"])
        assert width["value"].to_dict() == pytest.approx(
            {
                ("r1", "arc", "Ne"): 2.15,
                ("r2", "arc", "Ne"): 3.1,
                ("r1", "arc", "Ar"): 2.6,
                ("r1", "quartz", ""): 3.4,
            }
        )
        row = width.loc[("r1", "arc", "Ne")]
        assert (row["count"], row["firstVisit"], row["lastVisit"], row["sourceVisits"]) == (
            4,
            100,
            103,
            "100-103",
        )
        # Lines per lit fiber are kept per arm: the two spectrographs' images together, over four exposures.
        perFiber = made[made["metric"] == "linesPerFiber"].set_index(["camera", "frameType", "selector"])
        assert perFiber.loc[("r", "arc", "Ne"), "count"] == 4
        assert set(perFiber.index) == {("r", "arc", "Ne"), ("r", "arc", "Ar"), ("r", "quartz", "")}

    def testCountIsExposuresNotImages(self):
        """One exposure of four spectrographs is one frame: no per-arm normal value comes from it."""
        one = pd.concat([iqRows([2.0], spectrograph=spectrograph) for spectrograph in (1, 2, 3, 4)])
        made = compare(candidates(one), None, 200).set_index("metric")
        assert made.loc["linesPerFiber", "count"] == 1 and made.loc["linesPerFiber", "status"] == "TOO_FEW"
        assert accepted(made.reset_index(), "collection", 200, "someone")[0].empty
        # Negative control: images of two spectrographs over three exposures are enough.
        three = pd.concat(
            [iqRows([2.0] * MIN_NORMAL_FRAMES, spectrograph=spectrograph) for spectrograph in (1, 2)]
        )
        made = compare(candidates(three), None, 200).set_index("metric")
        assert made.loc["linesPerFiber", "count"] == MIN_NORMAL_FRAMES
        assert made.loc["linesPerFiber", "status"] == "NEW"

    def testOnlyWhatTheGateWouldJudge(self):
        """Science frames, unmeasured widths and widths on too few lines give no normal value."""
        iq = pd.concat(
            [
                iqRows([2.0] * 3, obsType="science", seqName="x"),
                iqRows([2.0] * 3, traceOnly=True, firstVisit=200),
                iqRows([2.0] * 3, linesPerFiber=2.0, firstVisit=300, seqName="Arc: Argon"),
                iqRows([np.nan] * 3, firstVisit=400, seqName="Arc: Xenon"),
            ]
        )
        made = candidates(iq)
        assert made[made["metric"] == "medFwhm"].empty
        assert set(made["selector"]) == {"Ne", "Ar", "Xe"}  # their lines per lit fiber

    def testExcludedImagesAreLeftOut(self):
        iq = iqRows([2.0, 2.2, 2.1, 9.0])
        exclude = pd.DataFrame({"visit": [103], "arm": ["r"], "spectrograph": [1]})
        made = candidates(iq, exclude=exclude).set_index("metric")
        assert made.loc["medFwhm", "value"] == pytest.approx(2.1) and made.loc["medFwhm", "count"] == 3
        # Negative control: with it, the median moves.
        assert candidates(iq).set_index("metric").loc["medFwhm", "value"] == pytest.approx(2.15)
        # Another camera's image of the same visit excludes nothing.
        other = exclude.assign(spectrograph=2)
        assert candidates(iq, exclude=other).set_index("metric").loc["medFwhm", "count"] == 4

    def testScatterPerSpecies(self):
        lines = pd.concat(
            [
                offsetRows([0.05, 0.06, 0.07], "r", "brn", "robustRmsDeltaX", species, "arc").assign(
                    reference="visit", statusType="RESERVED"
                )
                for species in ("NeI", "HgI", "Trace")
            ]
        )
        made = candidates(iqRows([2.0] * 3), lines)
        scatter = made[made["metric"] == "robustRmsDeltaX"]
        assert list(scatter["selector"]) == ["NeI"] and scatter["value"].iloc[0] == pytest.approx(0.06)

    def testEmpty(self):
        assert candidates().empty and list(candidates().columns) == list(CANDIDATE_COLUMNS)


class TestMArmOffsets:
    """The b arm in x from the quartz, the n arm in y from the Neon arcs: with m minus with r."""

    @staticmethod
    def tables(withM=(-0.52, -0.5, -0.51), withR=(0.01, -0.01, 0.0)):
        iq = pd.concat(
            [
                iqRows([3.0] * 6, arm="b", seqName="Trace", obsType="trace"),
                iqRows([3.0] * 6, arm="n", seqName="Arc: Neon", firstVisit=200),
            ]
        )
        lines = pd.concat(
            [
                offsetRows(withR, "b", "brn", "medianDeltaX", "Trace", "trace"),
                offsetRows(withM, "b", "bmn", "medianDeltaX", "Trace", "trace", firstVisit=103),
                offsetRows([0.02, 0.0, 0.01], "n", "brn", "medianDeltaY", "NeI", "arc", firstVisit=200),
                offsetRows([-0.13, -0.12, -0.14], "n", "bmn", "medianDeltaY", "NeI", "arc", firstVisit=203),
            ]
        )
        return iq, lines

    def testWithMMinusWithR(self):
        made = candidates(*self.tables()).set_index("metric")
        x, y = made.loc["mArmOffsetX"], made.loc["mArmOffsetY"]
        assert (x["camera"], x["frameType"], x["selector"], x["count"]) == ("b1", "", "", 3)
        assert x["value"] == pytest.approx(-0.51) and y["value"] == pytest.approx(-0.14)
        assert y["camera"] == "n1" and x["sourceVisits"] == "100-105"
        # Negative control: the sign is with m minus with r, not the reverse.
        assert x["value"] != pytest.approx(0.51)

    def testCountIsTheModeWithFewerVisits(self):
        made = candidates(*self.tables(withM=(-0.5, -0.5))).set_index("metric")
        assert made.loc["mArmOffsetX", "count"] == 2
        # Too few visits in one mode: compared as TOO_FEW, and not stored.
        compared = compare(made.reset_index(), None, 300).set_index("metric")
        assert (
            compared.loc["mArmOffsetX", "status"] == "TOO_FEW"
            and compared.loc["mArmOffsetY", "status"] == "NEW"
        )

    def testNeedsBothModes(self):
        iq, lines = self.tables()
        made = candidates(iq, lines[lines["cameraSet"] == "bmn"])
        assert not made["metric"].str.startswith("mArmOffset").any()

    def testOtherSpeciesAndArmsGiveNone(self):
        iq, lines = self.tables()
        lines = lines.assign(
            description=lines["description"].replace({"NeI": "ArI"}), arm=lines["arm"].replace({"b": "r"})
        )
        iq = iq.assign(arm=iq["arm"].replace({"b": "r"}))
        assert not candidates(iq, lines)["metric"].str.startswith("mArmOffset").any()

    def testTheGateUsesWhatIsMade(self):
        """Offsets made here, stored, bring a science frame read with m back inside its limit."""
        made = candidates(*self.tables())
        stored, _ = accepted(compare(made, None, 300), "collection", 300, "someone")
        science = pd.DataFrame(
            {
                "visit": [400],
                "arm": ["b"],
                "spectrograph": [1],
                "cameraSet": ["bmn"],
                "obsType": ["science"],
                "reference": ["calib"],
                "description": ["Trace"],
                "statusType": ["BOTH"],
                "metric": ["medianDeltaX"],
                "value": [-0.6],
            }
        )
        assert gate(lines=science, normalValues=stored).iloc[0]["qaStatus"] == "PASS"
        assert gate(lines=science).iloc[0]["qaStatus"] == "WARN"
        # Not before they are valid.
        assert gate(lines=science.assign(visit=299), normalValues=stored).iloc[0]["qaStatus"] == "WARN"


class TestCompare:
    @staticmethod
    def inForce(value=2.0, metric="medFwhm", camera="r1", selector="Ne", validFrom=1, count=10):
        return pd.DataFrame(
            [
                {
                    "metric": metric,
                    "camera": camera,
                    "frameType": "arc",
                    "selector": selector,
                    "value": value,
                    "count": count,
                    "validFrom": validFrom,
                }
            ]
        )

    @pytest.mark.parametrize(
        "widths, status, shift",
        [
            ([2.0, 2.02, 2.04], "PASS", 1.0),
            ([2.06] * 3, "WARN", 3.0),
            ([2.2] * 3, "FAIL", 10.0),
            ([1.0] * 3, "PASS", -50.0),
        ],
    )
    def testWidthAgainstTheValueInForce(self, widths, status, shift):
        compared = compare(candidates(iqRows(widths)), self.inForce(), 200).set_index("metric")
        row = compared.loc["medFwhm"]
        assert row["status"] == status and row["previous"] == 2.0 and row["shift"] == pytest.approx(shift)

    def testFewerLinesIsTheBadDirection(self):
        inForce = self.inForce(100.0, "linesPerFiber", "r")
        compared = compare(candidates(iqRows([np.nan] * 3, linesPerFiber=70.0)), inForce, 200)
        assert compared.set_index("metric").loc["linesPerFiber", "status"] == "WARN"
        # Negative control: as many more lines is no fault.
        compared = compare(candidates(iqRows([np.nan] * 3, linesPerFiber=130.0)), inForce, 200)
        assert compared.set_index("metric").loc["linesPerFiber", "status"] == "PASS"

    def testNewAndTooFew(self):
        made = candidates(iqRows([2.0] * (MIN_NORMAL_FRAMES - 1)))
        assert set(compare(made, self.inForce(), 200)["status"]) == {"TOO_FEW"}
        made = candidates(iqRows([2.0] * MIN_NORMAL_FRAMES))
        assert set(compare(made, None, 200)["status"]) == {"NEW"}
        assert set(compare(made, self.inForce(camera="r2"), 200)["status"]) == {"NEW"}

    def testComparedWithWhatIsInForceBeforeValidFrom(self):
        """Rerunning after acceptance compares with the previous values, not with those just stored."""
        inForce = pd.concat([self.inForce(2.0, validFrom=1), self.inForce(2.2, validFrom=200)])
        made = candidates(iqRows([2.2] * 3))
        assert compare(made, inForce, 200).set_index("metric").loc["medFwhm", "previous"] == 2.0
        assert compare(made, inForce, 201).set_index("metric").loc["medFwhm", "previous"] == 2.2

    def testMArmOffsetShiftIsInPixelsAndNeverRefused(self):
        made = candidates(*TestMArmOffsets.tables())
        inForce = self.inForce(-0.3, "mArmOffsetX", "b1", "").assign(frameType="")
        row = compare(made, inForce, 300).set_index("metric").loc["mArmOffsetX"]
        assert row["status"] == "PASS" and row["shift"] == pytest.approx(-0.21)

    def testFormat(self):
        compared = compare(candidates(iqRows([2.2] * 3)), self.inForce(), 200)
        text = formatComparison(compared)
        assert "medFwhm" in text and "+10.00 %" in text and "FAIL" in text and "1 FAIL, 1 NEW" in text
        assert "linesPerFiber" not in formatComparison(compared[compared["status"] == "FAIL"])
        assert "medFwhm" not in formatComparison(compared[compared["status"] != "FAIL"], onlyShifted=False)


class TestAccepted:
    def setup_method(self):
        self.compared = compare(
            candidates(pd.concat([iqRows([2.2] * 3), iqRows([2.0] * 2, spectrograph=2)])),
            TestCompare.inForce(),
            200,
        )

    def testFailIsRefusedWithoutAReason(self):
        stored, refused = accepted(self.compared, "collection", 200, "someone", when=datetime(2026, 1, 2))
        assert list(stored.columns) == list(NORMAL_COLUMNS)
        # The shifted width is refused and the two-frame one left out: the previous values stay in force.
        assert sorted(refused["status"]) == ["FAIL", "TOO_FEW"]
        assert "medFwhm" not in set(stored["metric"]) and len(stored) == 1
        row = stored.iloc[0]
        assert (row["collection"], row["validFrom"], row["acceptedBy"], row["reason"]) == (
            "collection",
            200,
            "someone",
            "",
        )
        assert row["acceptedAt"] == datetime(2026, 1, 2)

    def testFailIsStoredWithItsReason(self):
        stored, refused = accepted(self.compared, "collection", 200, "someone", reason="refocused")
        assert list(refused["status"]) == ["TOO_FEW"]
        reasons = stored.set_index("metric")["reason"].to_dict()
        # The reason is recorded on the value it was needed for, and only there.
        assert reasons == {"medFwhm": "refocused", "linesPerFiber": ""}

    def testAcceptedValuesJudge(self):
        stored, _ = accepted(self.compared, "collection", 200, "someone", reason="refocused")
        frames = iqRows([2.2, 2.2], firstVisit=199)
        inForce = pd.concat(
            [TestCompare.inForce().assign(collection="collection"), stored], ignore_index=True
        )
        result = gate(iq=frames, normalValues=inForce).set_index("visit")["qaStatus"]
        assert result.to_dict() == {199: "FAIL", 200: "PASS"}


def testFixedFailures():
    iq = iqRows([9.0, 2.0], linesPerFiber=80.0)
    lines = offsetRows([5.0, 0.1], "r", "brn", "medianDeltaX", "Trace", "arc")
    normal = TestCompare.inForce()
    judged = judge(measurements(iq, lines), normal)
    failed = fixedFailures(judged)
    # Visit 100 fails its offset, a fixed check; its width failing too is what a normal value is for.
    assert failed.to_dict("records") == [{"visit": 100, "arm": "r", "spectrograph": 1}]
    onlyWidth = fixedFailures(judge(measurements(iq), normal))
    assert onlyWidth.empty
