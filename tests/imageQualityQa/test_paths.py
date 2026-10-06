"""Which measurement path ``imageQualityQa`` takes, and what it then judges."""

import logging

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import configThresholds

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
TRACE = ("scienceTrace", "Trace")
ARC = ("scienceArc", "Arc: Neon")


def traceEntry(warn: float, fail: float, arm: str = "r") -> pd.DataFrame:
    """Return a thresholds table with one ``medFwhm`` entry for trace quanta of ``arm``."""
    return pd.DataFrame(
        [
            {
                "metric": "medFwhm",
                "obsType": "trace",
                "arm": arm,
                "warn": warn,
                "fail": fail,
                "higherIsWorse": True,
                "absolute": False,
                "provenance": "test",
            }
        ]
    )


def configTable(iqqa) -> pd.DataFrame:
    """Return the task config's thresholds as a table."""
    return configThresholds(iqqa.ImageQualityQaConfig())


def testTraceFallsBackWhenCalexpUnusable(runTask, fakes, caplog):
    """A calexp that measures nothing hands over to fiberProfiles, which is not judged."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)
    calexp = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*TRACE))
    profiles = fakes.FiberProfiles(detectorMap, sigma=1.2)
    with caplog.at_level(logging.INFO):
        outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp, fiberProfiles=profiles)

    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["traceOnly"]
    assert metrics["medFwhm"] == pytest.approx(FWHM_FACTOR * 1.2)
    # The profile path flags nothing, so its flag rate would be zero by construction.
    assert np.isnan(metrics["pctFlagged"])
    # nLines counts the rows of ``lines``, not the 50 profile swaths.
    assert metrics["nLines"] == len(lines)
    assert len(outputs.iqQaData) == 50 and outputs.iqQaData["traceOnly"].all()
    assert metrics["qaStatus"] == "UNKNOWN"
    assert "fiberProfiles" in metrics["qaReason"]

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("calexp unusable" in message and "of 80 cross-dispersion" in message for message in warnings)
    assert any("fiber profile calibration widths" in message for message in warnings)

    # Negative control: with no fiberProfiles the same quantum is sparse.
    outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert not metrics["traceOnly"] and np.isnan(metrics["medFwhm"])
    assert (metrics["qaStatus"], metrics["qaReason"]) == ("UNKNOWN", "no metric measured")


def testProfileFwhmIsNotJudged(iqqa, runTask, fakes):
    """A trace entry that would fail the profile FWHM gives no verdict on it; a calexp one fails."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)
    thresholds = [traceEntry(warn=1.0, fail=2.0), configTable(iqqa)]

    unusable = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*TRACE))
    profiles = fakes.FiberProfiles(detectorMap, sigma=1.2)
    outputs = runTask(
        lines, detectorMap=detectorMap, calexp=unusable, fiberProfiles=profiles, thresholds=thresholds
    )
    assert outputs.iqQaMetrics["medFwhm"].iloc[0] > 2.0
    assert outputs.iqQaMetrics["qaStatus"].iloc[0] == "UNKNOWN"

    # Negative control: the same width measured from the calexp fails.
    usable = fakes.Exposure(fakes.traceImage(detectorMap, sigma=1.2), fakes.header(*TRACE))
    outputs = runTask(
        lines, detectorMap=detectorMap, calexp=usable, fiberProfiles=profiles, thresholds=thresholds
    )
    metrics = outputs.iqQaMetrics.iloc[0]
    assert not metrics["traceOnly"]
    assert (metrics["qaStatus"], metrics["qaDecidedBy"]) == ("FAIL", "medFwhm")


def testTraceFwhmIsGatedPerArm(iqqa, runTask, fakes):
    """A trace entry for one arm judges that arm's trace quanta, not its arcs or other arms."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)
    calexp = fakes.Exposure(fakes.traceImage(detectorMap, sigma=1.3), fakes.header(*TRACE))
    thresholds = [traceEntry(warn=2.5, fail=2.8, arm="r"), configTable(iqqa)]

    outputs = runTask(lines, arm="r", detectorMap=detectorMap, calexp=calexp, thresholds=thresholds)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "trace" and not metrics["traceOnly"]
    assert metrics["medFwhm"] == pytest.approx(FWHM_FACTOR * 1.3, rel=0.02)
    assert metrics["qaStatus"] == "FAIL"
    assert "fail threshold 2.8px" in metrics["qaReason"]

    # The b arm falls through to the config's 3.2/3.5.
    outputs = runTask(lines, arm="b", detectorMap=detectorMap, calexp=calexp, thresholds=thresholds)
    assert outputs.iqQaMetrics["qaStatus"].iloc[0] == "PASS"

    # An r-arm arc with the same FWHM is not a trace.
    arc = fakes.arcLines(detectorMap, numPerFiber=30, fwhm=FWHM_FACTOR * 1.3)
    arcExposure = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*ARC))
    outputs = runTask(arc, arm="r", detectorMap=detectorMap, calexp=arcExposure, thresholds=thresholds)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "arc" and metrics["qaStatus"] == "PASS"


def testNothingMeasuredIsUnknown(runTask, fakes):
    """A dark is not judged, though the same lines on an arc pass."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)

    dark = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header("scienceDark", "Dark"))
    metrics = runTask(lines, detectorMap=detectorMap, calexp=dark).iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "dark"
    assert (metrics["qaStatus"], metrics["qaReason"]) == ("UNKNOWN", "no metric measured")

    arc = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*ARC))
    metrics = runTask(lines, detectorMap=detectorMap, calexp=arc).iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "arc" and metrics["qaStatus"] == "PASS"
    assert metrics["nLines"] == len(lines) and metrics["pctFlagged"] == 0.0


def testTwilightIsNotMeasured(runTask, fakes):
    """Twilight is its own class and, until sky lines are measured, unassessed."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)
    calexp = fakes.Exposure(
        fakes.traceImage(detectorMap, sigma=1.2), fakes.header("scienceObject", "Twilight sky")
    )
    metrics = runTask(lines, detectorMap=detectorMap, calexp=calexp).iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "twilight"
    assert metrics["qaStatus"] == "UNKNOWN" and np.isnan(metrics["medFwhm"])
    assert metrics["nLines"] == len(lines)


@pytest.mark.parametrize(
    ("seqType", "seqName", "obsType"),
    [
        ("scienceObject", "Twilight sky", "twilight"),
        ("scienceObject_windowed", "twilight", "twilight"),
        ("scienceObject", "sky flat", "allsky"),
        ("scienceObject", "Field 3", "science"),
        ("scienceDark", "Twilight sky", "dark"),
        ("scienceTrace", "Trace", "trace"),
        ("scienceArc", "Arc: HgCd", "arc"),
        ("biases", "Bias", "unknown"),
    ],
)
def testClassifyVisit(iqqa, fakes, seqType, seqName, obsType):
    """``W_SEQTYP`` sets the class; ``W_SEQNAM`` splits scienceObject."""
    task = iqqa.ImageQualityQaTask(config=iqqa.ImageQualityQaConfig())
    calexp = fakes.Exposure(np.zeros((2, 2)), fakes.header(seqType, seqName))
    assert task._classifyVisit(calexp, None)[0] == obsType
