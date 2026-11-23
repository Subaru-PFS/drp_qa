"""The gate, on synthetic numbers: every rule, and what gets no verdict."""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import STATUS_ORDER, VERDICT_COLUMNS, gate, judge, longLines, measurements, worst
from pfs.drp.qa.metrics.registry import MIN_NORMAL_FRAMES

VISIT = 1000


def iqRow(
    arm="r", spectrograph=1, visit=VISIT, obsType="arc", seqName="Arc: Neon", illumination="full", **values
):
    """Return one ``iqQaMetrics`` row."""
    row = {
        "visit": visit,
        "arm": arm,
        "spectrograph": spectrograph,
        "obsType": obsType,
        "illumination": illumination,
        "seqName": seqName,
        "traceOnly": False,
        "medFwhm": np.nan,
        "medDxCenter": np.nan,
        "linesPerFiber": np.nan,
    }
    return row | values


def lineRow(metric, value, description="Trace", reference="calib", statusType="BOTH", **keys):
    """Return one ``dmLineResidualMetrics`` row."""
    row = {
        "visit": VISIT,
        "arm": "r",
        "spectrograph": 1,
        "cameraSet": "brn",
        "obsType": "science",
        "reference": reference,
        "description": description,
        "statusType": statusType,
        "metric": metric,
        "value": value,
    }
    return row | keys


def normalRow(metric, camera, frameType, selector, value, count=10, validFrom=1):
    """Return one normal-value row."""
    return {
        "metric": metric,
        "camera": camera,
        "frameType": frameType,
        "selector": selector,
        "value": value,
        "count": count,
        "validFrom": validFrom,
    }


def frame(*rows):
    return pd.DataFrame(list(rows))


def verdict(**tables):
    """Return the one image's verdict row."""
    result = gate(**tables)
    assert len(result) == 1, result
    return result.iloc[0]


def testEmpty():
    assert list(gate().columns) == list(VERDICT_COLUMNS) and gate().empty
    assert measurements().empty
    assert worst([]) == "UNKNOWN" and worst(["PASS", "FAIL", "WARN"]) == "FAIL"
    assert STATUS_ORDER == ("UNKNOWN", "PASS", "WARN", "FAIL")


class TestScience:
    """Science and all-sky frames: the offset from the calibration and the sky-line residual."""

    @pytest.mark.parametrize(
        "value, status",
        [(0.49, "PASS"), (0.5, "WARN"), (-0.5, "WARN"), (0.99, "WARN"), (1.0, "FAIL"), (-3, "FAIL")],
    )
    def testOffsetInX(self, value, status):
        """A value at a limit has crossed it, in either direction."""
        row = verdict(lines=frame(lineRow("medianDeltaX", value)))
        assert row["qaStatus"] == status and row["frameType"] == "science"
        assert row["qaDecidedBy"] == ("" if status == "PASS" else "skyOffset")
        if status != "PASS":
            assert f"x offset {value:+.2f} px, limit {0.5 if status == 'WARN' else 1:g} px" == row["qaReason"]

    def testAllSkyIsJudgedLikeScience(self):
        row = verdict(lines=frame(lineRow("medianDeltaX", 0.7, obsType="allsky")))
        assert (row["qaStatus"], row["frameType"]) == ("WARN", "allsky")
        # Negative control: the same offset on an arc is within the calibration frames' 1 px.
        arc = gate(iq=frame(iqRow()), lines=frame(lineRow("medianDeltaX", 0.7, obsType="arc")))
        assert arc.iloc[0]["qaStatus"] == "PASS"

    def testOffsetInYIsTheSpeciesFurthestOff(self):
        lines = frame(
            lineRow("medianDeltaY", 0.2, "OH"),
            lineRow("medianDeltaY", -0.6, "O2"),
            lineRow("medianDeltaY", 0.55, "NaI"),
            lineRow("medianDeltaX", 0.1),
        )
        row = verdict(lines=lines)
        assert row["qaStatus"] == "WARN" and row["qaReason"] == "y offset (O2) -0.60 px, limit 0.5 px"
        judged = judge(measurements(lines=lines))
        # One species is judged in y, not each: the others leave no row.
        assert sorted(judged["part"]) == ["x", "y O2"]

    def testOnlyGoodLinesAgainstTheCalibration(self):
        """RESERVED or USED rows against the calibration, and rows against the visit, are not the offset."""
        for keys in ({"statusType": "RESERVED"}, {"statusType": "USED"}, {"reference": "visit"}):
            row = verdict(lines=frame(lineRow("medianDeltaX", 5.0, **keys)))
            assert row["qaStatus"] == "UNKNOWN", keys
        assert verdict(lines=frame(lineRow("medianDeltaX", 5.0, statusType="UNFITTED")))["qaStatus"] == "FAIL"

    @pytest.mark.parametrize(
        "arm, species, value, status",
        [
            ("r", "OH", 0.49, "PASS"),
            ("r", "OH", -0.5, "WARN"),
            ("n", "OH", 1.0, "FAIL"),
            ("m", "OH", 0.6, "WARN"),
            ("b", "OI", 1.2, "PASS"),
            ("b", "OI", 1.5, "WARN"),
            ("b", "OI", -2.5, "FAIL"),
        ],
    )
    def testSkyLineResidual(self, arm, species, value, status):
        lines = frame(lineRow("medianDeltaYKms", value, species, "visit", "RESERVED", arm=arm))
        row = verdict(lines=lines)
        assert row["qaStatus"] == status
        if status != "PASS":
            assert row["qaDecidedBy"] == ("skyLineResidualBlue" if arm == "b" else "skyLineResidual")
            assert row["qaReason"].startswith(f"{'[O I]' if arm == 'b' else 'OH'} residual {value:+.2f} km/s")

    def testEachArmHasItsSpecies(self):
        """The b arm is judged on [O I] and the others on OH; the other species' residuals are stored only."""
        for arm, species in (("b", "OH"), ("r", "OI"), ("r", "O2"), ("n", "NaI")):
            lines = frame(lineRow("medianDeltaYKms", 9.0, species, "visit", "RESERVED", arm=arm))
            assert verdict(lines=lines)["qaStatus"] == "UNKNOWN", (arm, species)
        # Nor the fitted lines, nor the residual against the calibration.
        for keys in ({"statusType": "BOTH"}, {"reference": "calib", "statusType": "RESERVED"}):
            lines = frame(lineRow("medianDeltaYKms", 9.0, "OH", **({"reference": "visit"} | keys)))
            assert verdict(lines=lines)["qaStatus"] == "UNKNOWN", keys

    def testJudgedOnWhatWasMeasured(self):
        """Without a sky-line residual an image is judged on its offsets alone; with neither it is UNKNOWN."""
        assert verdict(lines=frame(lineRow("medianDeltaX", 0.1)))["qaStatus"] == "PASS"
        unmeasured = verdict(lines=frame(lineRow("medianDeltaX", np.nan)))
        assert (
            unmeasured["qaStatus"] == "UNKNOWN" and unmeasured["qaReason"] == "nothing to judge was measured"
        )

    def testWidthAndScatterAreNotJudged(self):
        iq = frame(iqRow(obsType="science", seqName="x", medFwhm=9.0, linesPerFiber=1.0, medDxCenter=5.0))
        lines = frame(
            lineRow("robustRmsDeltaX", 9.0, "Trace", "visit", "RESERVED"),
            lineRow("robustRmsDeltaY", 9.0, "OH", "visit", "RESERVED"),
        )
        normal = frame(normalRow("medFwhm", "r1", "science", "x", 1.0))
        assert verdict(iq=iq, lines=lines, normalValues=normal)["qaStatus"] == "UNKNOWN"


class TestMArmOffset:
    """On a visit read with m, the b arm's x and the n arm's y are measured from where they sit in that mode."""

    NORMAL = frame(
        normalRow("mArmOffsetX", "b1", "", "", -0.51, count=4),
        normalRow("mArmOffsetY", "n1", "", "", -0.14, count=4),
    )

    def testSubtractedBeforeTheLimit(self):
        lines = frame(lineRow("medianDeltaX", -0.62, arm="b", cameraSet="bmn"))
        assert verdict(lines=lines, normalValues=self.NORMAL)["qaStatus"] == "PASS"
        # Negative control: uncorrected, the same frame warns.
        assert verdict(lines=lines)["qaStatus"] == "WARN"
        # And the wrong sign would fail it.
        wrong = self.NORMAL.assign(value=-self.NORMAL["value"])
        assert verdict(lines=lines, normalValues=wrong)["qaStatus"] == "FAIL"

    def testTheReasonSaysSo(self):
        lines = frame(lineRow("medianDeltaX", -1.2, arm="b", cameraSet="bm"))
        row = verdict(lines=lines, normalValues=self.NORMAL)
        assert row["qaReason"] == "x offset -0.69 px, limit 0.5 px (less the m-arm offset -0.51 px)"

    def testNArmInY(self):
        lines = frame(lineRow("medianDeltaY", -0.6, "OH", arm="n", cameraSet="bmn"))
        assert verdict(lines=lines, normalValues=self.NORMAL)["qaStatus"] == "PASS"
        assert verdict(lines=lines)["qaStatus"] == "WARN"

    def testOnlyWithMAndOnlyThoseArmsAndAxes(self):
        for keys in (
            {"arm": "b", "cameraSet": "brn"},  # read with r
            {"arm": "r", "cameraSet": "bmn"},  # not the b arm
            {"arm": "n", "cameraSet": "bmn"},  # the n arm's offset is in y
        ):
            lines = frame(lineRow("medianDeltaX", -0.62, **keys))
            normal = pd.concat([self.NORMAL, frame(normalRow("mArmOffsetX", "n1", "", "", -0.5))])
            assert verdict(lines=lines, normalValues=normal)["qaStatus"] == "WARN", keys

    def testCalibrationFramesAreNotCorrected(self):
        lines = frame(lineRow("medianDeltaX", -1.2, arm="b", cameraSet="bm", obsType="trace"))
        row = verdict(
            iq=frame(iqRow(arm="b", obsType="trace", seqName="Trace")), lines=lines, normalValues=self.NORMAL
        )
        assert row["qaStatus"] == "WARN" and "m-arm" not in row["qaReason"]


class TestCalibrationFrames:
    def testOffsets(self):
        iq = frame(iqRow())
        for value, status in ((0.99, "PASS"), (1.0, "WARN"), (-2.0, "FAIL")):
            lines = frame(lineRow("medianDeltaX", value, obsType="arc"))
            assert verdict(iq=iq, lines=lines)["qaStatus"] == status
            lines = frame(lineRow("medianDeltaY", value, "NeI", obsType="arc"))
            row = verdict(iq=iq, lines=lines)
            assert row["qaStatus"] == status and row["qaDecidedBy"] == (
                "" if status == "PASS" else "calibOffset"
            )

    def testLampFramesNeedTheirIllumination(self):
        """Line residuals of an arc with no ``iqQaMetrics`` row are not judged: it may be lit through the IIS."""
        lines = frame(lineRow("medianDeltaX", 5.0, obsType="arc"))
        assert verdict(lines=lines)["qaStatus"] == "UNKNOWN"
        assert verdict(iq=frame(iqRow()), lines=lines)["qaStatus"] == "FAIL"
        assert verdict(iq=frame(iqRow(illumination="iis")), lines=lines)["qaStatus"] == "UNKNOWN"
        # Another image's row does not vouch for this one.
        other = gate(iq=frame(iqRow(spectrograph=2)), lines=lines)
        assert other.set_index("spectrograph").loc[1, "qaStatus"] == "UNKNOWN"

    def testTraceCentreOffsetOnEngineeringFiberFrames(self):
        for value, status in ((0.5, "PASS"), (-1.0, "WARN"), (2.5, "FAIL")):
            iq = frame(iqRow(illumination="iis", seqName="ImageQuality (Ne)", medDxCenter=value))
            row = verdict(iq=iq)
            assert (row["qaStatus"], row["frameType"]) == (status, "iisArc")
        # Negative control: on a full-slit frame it is not judged; the line residuals give the offset there.
        assert verdict(iq=frame(iqRow(medDxCenter=2.5)))["qaStatus"] == "UNKNOWN"


class TestRelative:
    """Checks against the camera's normal value."""

    NORMAL = frame(
        normalRow("medFwhm", "r1", "arc", "Ne", 2.0),
        normalRow("medFwhm", "r1", "quartz", "", 3.0),
        normalRow("linesPerFiber", "r", "arc", "Ne", 80.0),
        normalRow("linesPerFiber", "r", "quartz", "", 4000.0),
        normalRow("robustRmsDeltaX", "r1", "arc", "NeI", 0.05),
        normalRow("robustRmsDeltaX", "r1", "quartz", "Trace", 0.02),
        normalRow("robustRmsDeltaY", "r1", "arc", "NeI", 0.04),
    )

    @pytest.mark.parametrize("width, status", [(2.059, "PASS"), (2.06, "WARN"), (2.1, "FAIL"), (1.0, "PASS")])
    def testWidth(self, width, status):
        row = verdict(iq=frame(iqRow(medFwhm=width, linesPerFiber=80.0)), normalValues=self.NORMAL)
        assert row["qaStatus"] == status
        if status == "FAIL":
            assert row["qaReason"] == "width 2.10 px is +5.0 % from normal 2.00 px, limit +5 %"

    @pytest.mark.parametrize(
        "count, status", [(61.0, "PASS"), (60.0, "WARN"), (40.0, "FAIL"), (160.0, "PASS")]
    )
    def testLinesPerFiberOnAnArc(self, count, status):
        """Fewer lines is the bad direction: more than normal passes."""
        row = verdict(iq=frame(iqRow(linesPerFiber=count)), normalValues=self.NORMAL)
        assert row["qaStatus"] == status
        if status == "WARN":
            assert row["qaReason"] == "lines per lit fiber 60.0 is -25.0 % from normal 80.0, limit -25 %"

    def testLinesPerFiberOnAQuartzIsTighter(self):
        quartz = iqRow(obsType="trace", seqName="Trace", linesPerFiber=3880.0)  # -3 %
        row = verdict(iq=frame(quartz), normalValues=self.NORMAL)
        assert (row["qaStatus"], row["qaDecidedBy"]) == ("WARN", "linesPerFiberQuartz")
        # Negative control: the same -3 % on an arc is well inside its -25 %.
        assert verdict(iq=frame(iqRow(linesPerFiber=77.6)), normalValues=self.NORMAL)["qaStatus"] == "PASS"

    def testScatter(self):
        iq = frame(iqRow())
        for metric, value, status in (
            ("robustRmsDeltaX", 0.059, "PASS"),
            ("robustRmsDeltaX", 0.06, "WARN"),
            ("robustRmsDeltaX", 0.075, "FAIL"),
            ("robustRmsDeltaY", 0.059, "PASS"),
            ("robustRmsDeltaY", 0.06, "WARN"),
            ("robustRmsDeltaY", 0.1, "FAIL"),
        ):
            lines = frame(lineRow(metric, value, "NeI", "visit", "RESERVED", obsType="arc"))
            assert verdict(iq=iq, lines=lines, normalValues=self.NORMAL)["qaStatus"] == status, (
                metric,
                value,
            )

    def testScatterOnlyOfHeldBackLinesAgainstTheVisit(self):
        iq = frame(iqRow())
        for keys in (
            {"statusType": "BOTH"},
            {"statusType": "USED"},
            {"reference": "calib", "statusType": "BOTH"},
        ):
            lines = frame(
                lineRow("robustRmsDeltaX", 9.0, "NeI", **({"reference": "visit"} | keys), obsType="arc")
            )
            assert verdict(iq=iq, lines=lines, normalValues=self.NORMAL)["qaStatus"] == "UNKNOWN", keys

    def testSpeciesNotJudged(self):
        """Mercury and cadmium, sky species and an arc's trace centroids are stored, not judged."""
        iq = frame(iqRow())
        normal = pd.concat(
            [
                self.NORMAL,
                frame(
                    *(
                        normalRow("robustRmsDeltaX", "r1", "arc", s, 0.01)
                        for s in ("HgI", "CdI", "Trace", "OH")
                    )
                ),
            ]
        )
        for species in ("HgI", "CdI", "Trace", "OH"):
            lines = frame(lineRow("robustRmsDeltaX", 9.0, species, "visit", "RESERVED", obsType="arc"))
            assert verdict(iq=iq, lines=lines, normalValues=normal)["qaStatus"] == "UNKNOWN", species
        # A quartz is judged on its trace centroids, in x only.
        quartz = frame(iqRow(obsType="trace", seqName="Trace"))
        lines = frame(lineRow("robustRmsDeltaX", 0.05, "Trace", "visit", "RESERVED", obsType="trace"))
        assert verdict(iq=quartz, lines=lines, normalValues=self.NORMAL)["qaStatus"] == "FAIL"
        lines = frame(lineRow("robustRmsDeltaY", 9.0, "Trace", "visit", "RESERVED", obsType="trace"))
        assert verdict(iq=quartz, lines=lines, normalValues=self.NORMAL)["qaStatus"] == "UNKNOWN"

    def testNoNormalValueIsUnknown(self):
        iq = frame(iqRow(medFwhm=9.0, linesPerFiber=80.0))
        row = verdict(iq=iq)
        assert row["qaStatus"] == "UNKNOWN" and row["qaReason"] == "no normal value"
        # Another camera's, lamp's or frame type's value is none.
        for wrong in (
            normalRow("medFwhm", "r2", "arc", "Ne", 2.0),
            normalRow("medFwhm", "r1", "arc", "Ar", 2.0),
            normalRow("medFwhm", "r1", "iisArc", "Ne", 2.0),
            normalRow("medFwhm", "r", "arc", "Ne", 2.0),
        ):
            assert verdict(iq=iq, normalValues=frame(wrong))["qaStatus"] == "UNKNOWN", wrong
        assert (
            verdict(iq=iq, normalValues=frame(normalRow("medFwhm", "r1", "arc", "Ne", 2.0)))["qaStatus"]
            == "FAIL"
        )

    def testTooFewFramesIsNoNormalValue(self):
        iq = frame(iqRow(medFwhm=9.0, linesPerFiber=80.0))
        few = frame(normalRow("medFwhm", "r1", "arc", "Ne", 2.0, count=MIN_NORMAL_FRAMES - 1))
        assert verdict(iq=iq, normalValues=few)["qaStatus"] == "UNKNOWN"
        enough = frame(normalRow("medFwhm", "r1", "arc", "Ne", 2.0, count=MIN_NORMAL_FRAMES))
        assert verdict(iq=iq, normalValues=enough)["qaStatus"] == "FAIL"

    def testLatestValueAtOrBeforeTheVisit(self):
        """The previous run's value stays in force until the new one is valid."""
        normal = frame(
            normalRow("medFwhm", "r1", "arc", "Ne", 2.0, validFrom=500),
            normalRow("medFwhm", "r1", "arc", "Ne", 2.2, validFrom=VISIT),
            normalRow("medFwhm", "r1", "arc", "Ne", 3.0, validFrom=VISIT + 1),
        )
        rows = [iqRow(medFwhm=2.2, linesPerFiber=80.0, visit=visit) for visit in (499, 999, VISIT, VISIT + 1)]
        result = gate(iq=frame(*rows), normalValues=normal).set_index("visit")["qaStatus"]
        assert result.to_dict() == {499: "UNKNOWN", 999: "FAIL", VISIT: "PASS", VISIT + 1: "PASS"}
        # A value with too few frames does not replace the one in force.
        normal.loc[1, "count"] = 1
        assert gate(iq=frame(rows[2]), normalValues=normal).iloc[0]["qaStatus"] == "FAIL"

    def testWidthOnTooFewLinesIsNotJudged(self):
        """b-arm Argon: under 5 lines per lit fiber the width rests on a handful of lines."""
        sparse = frame(iqRow(medFwhm=9.0, linesPerFiber=4.9))
        normal = pd.concat(
            [self.NORMAL, frame(normalRow("linesPerFiber", "r", "arc", "Ne", 4.0))], ignore_index=True
        )
        normal = normal.drop(index=2)
        row = verdict(iq=sparse, normalValues=normal)
        assert row["qaStatus"] == "UNKNOWN" and "under 5 lines per lit fiber" in row["qaReason"]
        # Negative control: with the lines, the same width fails. A quartz has no such floor.
        assert (
            verdict(iq=frame(iqRow(medFwhm=9.0, linesPerFiber=80.0)), normalValues=self.NORMAL)["qaStatus"]
            == "FAIL"
        )

    def testAFrameShortOfLinesStillFailsItsCount(self):
        """The floor is on the normal value, so obstructed fibers are caught, not excused."""
        row = verdict(iq=frame(iqRow(medFwhm=2.0, linesPerFiber=2.0)), normalValues=self.NORMAL)
        assert (row["qaStatus"], row["qaDecidedBy"]) == ("FAIL", "linesPerFiberArc")

    def testWidthReadFromFiberProfilesIsNotMeasured(self):
        iq = frame(iqRow(medFwhm=9.0, linesPerFiber=80.0, traceOnly=True))
        normal = self.NORMAL[self.NORMAL["metric"] == "medFwhm"]
        assert verdict(iq=iq, normalValues=normal)["qaStatus"] == "UNKNOWN"

    def testEngineeringFiberFramesHaveTheirOwnValues(self):
        iq = frame(iqRow(illumination="iis", seqName="ImageQuality (Ne)", medFwhm=2.0, linesPerFiber=80.0))
        # The full-slit values do not judge it.
        assert verdict(iq=iq, normalValues=self.NORMAL)["qaStatus"] == "UNKNOWN"
        own = frame(
            normalRow("medFwhm", "r1", "iisArc", "Ne", 1.8),
            normalRow("linesPerFiber", "r", "iisArc", "Ne", 81.0),
        )
        row = verdict(iq=iq, normalValues=own)
        assert (row["qaStatus"], row["qaDecidedBy"]) == ("FAIL", "width")


class TestTwilight:
    @staticmethod
    def row(arm="b", level=0.0, rms=0.5, patch=0.5, tooFaint=False):
        return {
            "visit": VISIT,
            "arm": arm,
            "spectrograph": 1,
            "tooFaint": tooFaint,
            "twilightLevelOffset": level,
            "twilightResidRms": rms,
            "twilightPatch": patch,
        }

    @pytest.mark.parametrize(
        "values, status, check",
        [
            ({}, "PASS", ""),
            ({"level": -2.0}, "WARN", "twilightLevel"),
            ({"level": 3.0}, "FAIL", "twilightLevel"),
            ({"rms": 1.5}, "WARN", "twilightScatter"),
            ({"rms": 2.5}, "FAIL", "twilightScatter"),
            ({"patch": 2.5}, "WARN", "twilightPatch"),
            ({"patch": 4.0}, "FAIL", "twilightPatch"),
            ({"level": np.inf}, "FAIL", "twilightLevel"),
        ],
    )
    def testLimits(self, values, status, check):
        row = verdict(twilight=frame(self.row(**values)))
        assert (row["qaStatus"], row["qaDecidedBy"], row["frameType"]) == (status, check, "twilight")

    def testTooFaintIsJudgedOnItsLevelAlone(self):
        faint = self.row(rms=9.0, patch=9.0, tooFaint=True)
        assert verdict(twilight=frame(faint))["qaStatus"] == "PASS"
        assert verdict(twilight=frame(faint | {"twilightLevelOffset": 5.0}))["qaStatus"] == "FAIL"
        # Negative control: bright enough, the same scatter fails.
        assert verdict(twilight=frame(faint | {"tooFaint": False}))["qaStatus"] == "FAIL"

    def testNArmIsNotJudged(self):
        row = verdict(twilight=frame(self.row(arm="n", level=9.0, rms=9.0, patch=9.0)))
        assert row["qaStatus"] == "UNKNOWN"
        assert verdict(twilight=frame(self.row(arm="m", level=9.0)))["qaStatus"] == "FAIL"


class TestImage:
    def testAnImageTakesItsWorstCheck(self):
        iq = frame(iqRow(medFwhm=2.07, linesPerFiber=30.0))
        lines = frame(lineRow("medianDeltaX", 1.5, obsType="arc"))
        row = verdict(iq=iq, lines=lines, normalValues=TestRelative.NORMAL)
        assert row["qaStatus"] == "FAIL" and row["qaDecidedBy"] == "linesPerFiberArc"
        # Every crossing is in the reason, WARNs included.
        assert row["qaReason"].count(";") == 2
        for text in ("width 2.07 px", "lines per lit fiber 30.0", "x offset +1.50 px"):
            assert text in row["qaReason"]

    def testNoFrameTypeIsUnknownNeverPass(self):
        for obsType in ("unknown", None):
            row = verdict(
                iq=frame(iqRow(obsType=obsType, medFwhm=2.0, linesPerFiber=80.0)),
                normalValues=TestRelative.NORMAL,
            )
            assert row["qaStatus"] == "UNKNOWN" and row["qaReason"] == "no frame type"
            assert row["frameType"] is None

    def testAnUnclassifiedSequenceTypeIsNamedInTheReason(self):
        """A sequence type the registry lacks is untyped, and the reason says which it is."""
        row = verdict(
            iq=frame(iqRow(obsType="unknown", seqType="rampObject", medFwhm=2.0, linesPerFiber=80.0)),
            normalValues=TestRelative.NORMAL,
        )
        assert (row["frameType"], row["qaStatus"]) == (None, "UNKNOWN")
        assert row["qaReason"] == "no frame type for sequence type rampObject"
        # Negative control: it is not judged as the arc its numbers would pass for.
        asArc = verdict(
            iq=frame(iqRow(seqType="scienceArc", medFwhm=2.0, linesPerFiber=80.0)),
            normalValues=TestRelative.NORMAL,
        )
        assert asArc["qaStatus"] == "PASS" != row["qaStatus"]

    def testTheFirstTableThatGivesATypeGivesIt(self):
        """A table whose header cannot type the frame does not leave a judged image without a frame type."""
        iq = frame(iqRow(obsType="unknown"))
        lines = frame(lineRow("medianDeltaX", 0.7))
        row = verdict(iq=iq, lines=lines)
        assert (row["frameType"], row["qaStatus"], row["qaDecidedBy"]) == ("science", "WARN", "skyOffset")
        # Negative control: when iqQaMetrics does type the frame, its type stands, illumination included.
        typed = verdict(
            iq=frame(iqRow(illumination="iis")), lines=frame(lineRow("medianDeltaX", 0.7, obsType="arc"))
        )
        assert (typed["frameType"], typed["qaStatus"]) == ("iisArc", "UNKNOWN")

    def testOneRowPerImageOfAnyTable(self):
        iq = frame(iqRow(spectrograph=1), iqRow(spectrograph=2, medFwhm=2.0, linesPerFiber=80.0))
        lines = frame(
            lineRow("medianDeltaX", 0.1, spectrograph=3), lineRow("medianDeltaX", 0.1, spectrograph=3)
        )
        twilight = frame(TestTwilight.row(arm="r") | {"spectrograph": 4})
        result = gate(iq=iq, lines=lines, twilight=twilight, normalValues=TestRelative.NORMAL)
        assert list(result["spectrograph"]) == [1, 2, 3, 4]
        assert list(result["qaStatus"]) == ["UNKNOWN", "PASS", "PASS", "PASS"]
        assert list(result["frameType"]) == ["arc", "arc", "science", "twilight"]

    def testIndexAndDtypesDoNotMatter(self):
        iq = frame(iqRow(medFwhm=2.2, linesPerFiber=80.0)).astype({"visit": "int32", "spectrograph": "int16"})
        iq.index = [7]
        normal = TestRelative.NORMAL.astype({"validFrom": "float64"})
        assert verdict(iq=iq, normalValues=normal)["qaStatus"] == "FAIL"


class TestSequenceType:
    """What a frame is comes from its sequence type; a frame type that is not judged says why."""

    #: A dark as Run31 stored them: the pfsConfig's engineering fibers GOOD, a width and a trace-centre offset.
    DARK = {  # noqa: RUF012
        "obsType": "unknown",
        "seqType": "darks",
        "seqName": "Morning Dark",
        "illumination": "iis",
        "medFwhm": 15.0,
        "medDxCenter": 6.0,
    }

    def testADarkIsNamedAndNotJudged(self):
        row = verdict(iq=frame(iqRow(**self.DARK)), normalValues=TestRelative.NORMAL)
        assert (row["frameType"], row["qaStatus"], row["qaDecidedBy"]) == ("dark", "UNKNOWN", "")
        assert row["qaReason"] == "dark frames are not judged: No lamp or sky is exposed."
        # Negative control: the same numbers on an engineering-fiber quartz fail its trace-centre check, so it
        # is the type that keeps the dark from a verdict, not the numbers.
        quartz = verdict(iq=frame(iqRow(**self.DARK | {"seqType": "scienceTrace"})))
        assert (quartz["frameType"], quartz["qaStatus"], quartz["qaDecidedBy"]) == (
            "iisQuartz",
            "FAIL",
            "traceCentreOffset",
        )

    @pytest.mark.parametrize(
        "seqType, kind",
        [
            ("fiberProfiles", "fiberProfile"),
            ("dotRoach", "dotRoach"),
            ("dotRoach_continued", "dotRoach"),
            ("slitThroughFocus", "throughFocus"),
            ("flats_binned", "engineering"),
            ("scienceTrace_windowed", "windowed"),
        ],
    )
    def testAFrameThatDiffersByDesignIsNotJudgedAsAQuartz(self, seqType, kind):
        """Dithered, refocused, binned or part-read frames get no quartz verdict, whatever was measured on them."""
        lines = frame(lineRow("medianDeltaX", 2.5, obsType="unknown", seqType=seqType))
        iq = frame(iqRow(obsType="unknown", seqType=seqType, seqName="x", medFwhm=9.0, linesPerFiber=1.0))
        normal = frame(
            normalRow("medFwhm", "r1", "quartz", "", 3.2),
            normalRow("linesPerFiber", "r", "quartz", "", 4000.0),
        )
        row = verdict(iq=iq, lines=lines, normalValues=normal)
        assert (row["frameType"], row["qaStatus"]) == (kind, "UNKNOWN")
        assert row["qaReason"].startswith(f"{kind} frames are not judged: ")
        # Negative control: as a quartz the same numbers fail three checks.
        quartz = verdict(
            iq=iq.assign(seqType="scienceTrace"),
            lines=lines.assign(seqType="scienceTrace", obsType="trace"),
            normalValues=normal,
        )
        assert quartz["qaStatus"] == "FAIL" and quartz["qaReason"].count(";") == 2

    def testTheReasonNamesTheSequenceTypeWhicheverTableHasIt(self):
        """An image no table types is named by the first table that read its sequence type."""
        iq = frame(iqRow(obsType="unknown", seqType=""))
        lines = frame(lineRow("medianDeltaX", 0.1, obsType="unknown", seqType="rampObject"))
        assert verdict(iq=iq, lines=lines)["qaReason"] == "no frame type for sequence type rampObject"
        assert verdict(iq=iq.drop(columns="seqType"), lines=lines)["qaReason"].endswith("rampObject")
        # Negative control: with no table saying, none is named.
        assert verdict(iq=iq, lines=lines.assign(seqType=None))["qaReason"] == "no frame type"

    def testTheSequenceTypeDecidesNotTheObservationType(self):
        """``obsType`` says how the frame was measured; with a sequence type stored it types nothing."""
        measuredAsTrace = iqRow(obsType="trace", seqType="fiberProfiles", medFwhm=9.0, linesPerFiber=1.0)
        assert verdict(iq=frame(measuredAsTrace))["frameType"] == "fiberProfile"
        heuristic = iqRow(obsType="unknown", seqType="scienceArc", medFwhm=2.0, linesPerFiber=80.0)
        row = verdict(iq=frame(heuristic), normalValues=TestRelative.NORMAL)
        assert (row["frameType"], row["qaStatus"]) == ("arc", "PASS")

    def testRowsStoredWithoutASequenceTypeKeepTheirType(self):
        """Rows from before ``seqType`` was a column, or with it null, are typed by ``obsType`` as they were."""
        for missing in ({}, {"seqType": None}, {"seqType": ""}, {"seqType": float("nan")}):
            row = verdict(
                iq=frame(iqRow(medFwhm=2.0, linesPerFiber=80.0, **missing)), normalValues=TestRelative.NORMAL
            )
            assert (row["frameType"], row["qaStatus"]) == ("arc", "PASS")

    def testALineTableTellsATwilightFromScienceWithoutTheName(self):
        """The line residuals carry no sequence name: their ``obsType``, made from it, splits a science sequence."""
        twilight = frame(lineRow("medianDeltaX", 0.7, seqType="scienceObject", obsType="twilight"))
        row = verdict(lines=twilight)
        assert (row["frameType"], row["qaStatus"]) == ("twilight", "UNKNOWN")
        # Negative control: a science frame with the same offset is judged on it.
        science = verdict(lines=frame(lineRow("medianDeltaX", 0.7, seqType="scienceObject")))
        assert (science["frameType"], science["qaStatus"]) == ("science", "WARN")


class TestLamp:
    """The lamp a normal value is kept per: the header's when stored, the sequence name's otherwise."""

    NORMAL = frame(
        normalRow("medFwhm", "r1", "iisArc", "Kr", 2.0),
        normalRow("medFwhm", "r1", "iisArc", "Ne", 1.0),
        normalRow("medFwhm", "r1", "quartz", "", 3.2),
    )

    def iis(self, **values):
        return iqRow(illumination="iis", seqName="ImageQuality", medFwhm=2.0, linesPerFiber=80.0, **values)

    def testTheHeaderLampJudgesAFrameWhoseNameHasNone(self):
        row = verdict(iq=frame(self.iis(lamps="Kr_eng")), normalValues=self.NORMAL)
        assert (row["frameType"], row["qaStatus"]) == ("iisArc", "PASS")
        # Negative control: the same width against the other lamp's normal value fails, so the lamp matters...
        assert verdict(iq=frame(self.iis(lamps="Ne_eng")), normalValues=self.NORMAL)["qaStatus"] == "FAIL"
        # ...and with neither a header lamp nor one in the name there is no normal value to judge by.
        for lamps in ({}, {"lamps": ""}, {"lamps": None}):
            unnamed = verdict(iq=frame(self.iis(**lamps)), normalValues=self.NORMAL)
            assert (unnamed["qaStatus"], unnamed["qaReason"]) == ("UNKNOWN", "no normal value")

    def testTheHeaderLampWinsOverTheName(self):
        named = self.iis(lamps="Kr_eng") | {"seqName": "ImageQuality (Ne)"}
        assert verdict(iq=frame(named), normalValues=self.NORMAL)["qaStatus"] == "PASS"
        # Negative control: by its name alone the frame is judged as Neon, and fails.
        byName = self.iis() | {"seqName": "ImageQuality (Ne)"}
        assert verdict(iq=frame(byName), normalValues=self.NORMAL)["qaStatus"] == "FAIL"

    def testEitherSpellingOfALampIsOneKey(self):
        """``Arc: Neon`` and a header's ``Ne`` find the same normal value."""
        normal = frame(normalRow("medFwhm", "r1", "arc", "Ne", 2.0))
        for row in (
            iqRow(medFwhm=2.0, linesPerFiber=80.0),
            iqRow(seqName="x", lamps="Ne", medFwhm=2.0, linesPerFiber=80.0),
        ):
            assert verdict(iq=frame(row), normalValues=normal)["qaStatus"] == "PASS"
        # Negative control: another lamp's header finds none.
        other = iqRow(seqName="x", lamps="Ar", medFwhm=2.0, linesPerFiber=80.0)
        assert verdict(iq=frame(other), normalValues=normal)["qaStatus"] == "UNKNOWN"

    def testAValueStoredUnderAnotherSpellingStillJudges(self):
        """Normal values stored as ``Neon`` or, for a quartz, ``Halogen`` judge the frames they were made for."""
        stored = frame(
            normalRow("medFwhm", "r1", "arc", "Neon", 2.0),
            normalRow("linesPerFiber", "r", "arc", "Neon", 80.0),
            normalRow("medFwhm", "r1", "iisQuartz", "Halogen", 3.0),
        )
        arc = verdict(iq=frame(iqRow(medFwhm=2.2, linesPerFiber=30.0)), normalValues=stored)
        assert arc["qaStatus"] == "FAIL" and arc["qaReason"].count(";") == 1
        quartz = iqRow(obsType="trace", illumination="iis", seqName="ImageQuality (Halogen)", medFwhm=3.2)
        assert verdict(iq=frame(quartz), normalValues=stored)["qaDecidedBy"] == "width"
        # Negative control: another lamp's value, however spelled, judges neither number.
        other = stored.assign(selector=["Argon", "Ar", "Halogen"])
        unjudged = verdict(iq=frame(iqRow(medFwhm=2.2, linesPerFiber=30.0)), normalValues=other)
        assert (unjudged["qaStatus"], unjudged["qaReason"]) == ("UNKNOWN", "no normal value")
        # A species, the key of a scatter check, is not a lamp and is matched as it is spelled.
        lines = frame(lineRow("robustRmsDeltaX", 0.2, "H", "visit", "RESERVED", obsType="arc"))
        byName = frame(normalRow("robustRmsDeltaX", "r1", "arc", "Hydrogen", 0.1))
        assert verdict(iq=frame(iqRow()), lines=lines, normalValues=byName)["qaStatus"] == "UNKNOWN"
        bySpecies = frame(normalRow("robustRmsDeltaX", "r1", "arc", "H", 0.1))
        assert verdict(iq=frame(iqRow()), lines=lines, normalValues=bySpecies)["qaStatus"] == "FAIL"

    def testAQuartzIsKeptPerCameraWhateverItsLampSays(self):
        for lamps in ({}, {"lamps": "Quartz"}):
            quartz = iqRow(obsType="trace", seqName="Trace", medFwhm=3.2, **lamps)
            assert verdict(iq=frame(quartz), normalValues=self.NORMAL)["qaStatus"] == "PASS"
        # Negative control: an arc's lamp is part of its key, so a lamp it has no value for is not judged.
        arc = iqRow(seqName="x", lamps="Quartz", medFwhm=3.2, linesPerFiber=80.0)
        assert verdict(iq=frame(arc), normalValues=self.NORMAL)["qaStatus"] == "UNKNOWN"


def testLongLines():
    wide = pd.DataFrame(
        {
            "visit": [VISIT],
            "arm": ["r"],
            "spectrograph": [1],
            "obsType": ["science"],
            "reference": ["calib"],
            "description": ["Trace"],
            "statusType": ["BOTH"],
            "medianDeltaX": [0.7],
            "robustRmsDeltaX": [0.02],
        }
    )
    long = longLines(wide)
    assert (
        sorted(long["metric"]) == ["medianDeltaX", "robustRmsDeltaX"] and "medianDeltaX" not in long.columns
    )
    assert gate(lines=long).iloc[0]["qaStatus"] == "WARN"


class TestSlitOffsetInvariance:
    """The visit's slit offsets against the calibration's (reference ``slit``, calibration frames)."""

    def _row(self, value, metric="maxSpatialOffsetDelta"):
        line = lineRow(
            metric, value, description="SlitOffset", reference="slit", statusType="ALL", obsType="arc"
        )
        return gate(iq=frame(iqRow()), lines=frame(line)).iloc[0]

    @pytest.mark.parametrize(
        "value, status",
        [(0.0, "PASS"), (0.019, "PASS"), (0.02, "WARN"), (0.05, "WARN"), (0.10, "FAIL"), (0.5, "FAIL")],
    )
    def testInvariance(self, value, status):
        """Offsets equal to the calibration pass; any real change warns, a large one fails."""
        row = self._row(value)
        assert row["qaStatus"] == status
        assert row["qaDecidedBy"] == ("" if status == "PASS" else "slitOffsetInvariance")

    def testSpectralIsJudgedToo(self):
        assert self._row(0.2, metric="maxSpectralOffsetDelta")["qaStatus"] == "FAIL"

    def testCompanionCountIsNotJudged(self):
        # nSlitOffsetChanged rides along for context but has no check: alone it leaves no verdict.
        row = self._row(7.0, metric="nSlitOffsetChanged")
        assert row["qaStatus"] == "UNKNOWN"

    def testNotJudgedOnScienceFrames(self):
        """Negative control: slitOffsetInvariance is a calibration-frame check; science is left ungated."""
        line = lineRow(
            "maxSpatialOffsetDelta", 0.5, description="SlitOffset", reference="slit", statusType="ALL"
        )
        assert gate(lines=frame(line)).iloc[0]["qaStatus"] == "UNKNOWN"
