"""Judging an image on the fibers its pfsConfig lights (PIPE2D-1935).

A design lighting one fiber group hides the others behind their black dots
(``fiberStatus`` BLACKSPOT). ``lines`` has rows only for the ``GOOD`` fibers, and
on an arc also holds trace centroids whose number follows the illumination, so
``linesPerFiber`` counts unflagged lamp lines (trace centroids on a quartz) per
lit fiber, and ``pctFlagged`` is taken over lamp lines alone.
"""

import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

TRACE = ("scienceTrace", "Trace")
ARC = ("scienceArc", "Arc: Neon")


def lowerIsWorse(metric: str, warn: float, fail: float, seqName: str) -> pd.DataFrame:
    """Return a thresholds table with one lower-is-worse entry for ``seqName``."""
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "seqName": seqName,
                "warn": warn,
                "fail": fail,
                "higherIsWorse": False,
                "absolute": False,
                "provenance": "test",
            }
        ]
    )


def arcWithTraces(fakes, detectorMap, fibers, linesPerFiber=30, flagged=0, tracesPerFiber=40):
    """Return an arc's ``lines``: lamp lines and, as fitDetectorMap adds them, trace centroids."""
    lamp = fakes.arcLines(detectorMap, linesPerFiber, flagged=flagged, fibers=fibers, description="NeI")
    traces = fakes.arcLines(detectorMap, tracesPerFiber, fibers=fibers, description="Trace")
    return pd.concat([lamp, traces], ignore_index=True)


def testLitFibers(iqqa, fakes):
    """GOOD fibers on the detector are lit; hidden ones are possible; engineering and broken are neither."""
    detectorMap = fakes.DetectorMap(numFibers=6)
    pfsConfig = fakes.PfsConfig(
        [1, 2, 3, 4, 5, 6, 99],  # 99 is on another spectrograph
        status=[1, 1, 4, 4, 2, 1, 1],
        targetType=[9, 4, 4, 9, 9, 5, 9],  # 2 is unassigned but GOOD: lit; 6 is engineering
    )
    lit, possible = iqqa.litFibers(pfsConfig, detectorMap.fiberId)
    assert lit == {1, 2}
    assert possible == 4  # 1, 2 and the hidden 3, 4
    assert iqqa.litFibers(None, detectorMap.fiberId) == (None, None)


def testOneGroupArcIsCountedPerLitFiber(runTask, fakes):
    """A one-group arc has a quarter of the rows of a full one, and the same lines per lit fiber."""
    detectorMap = fakes.DetectorMap(numFibers=20)
    calexp = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*ARC))
    thresholds = [lowerIsWorse("linesPerFiber", warn=28, fail=25, seqName="Arc: Neon")]

    full = arcWithTraces(fakes, detectorMap, detectorMap.fiberId, tracesPerFiber=60)
    allLit = fakes.PfsConfig(detectorMap.fiberId)
    fullMetrics = runTask(
        full, detectorMap=detectorMap, calexp=calexp, pfsConfig=allLit, thresholds=thresholds
    ).iqQaMetrics.iloc[0]

    litIds, pfsConfig = fakes.oneGroup(detectorMap)
    # Fewer trace centroids per fiber on the one-group frame, as on 149398 against 149883.
    group = arcWithTraces(fakes, detectorMap, litIds, tracesPerFiber=20)
    groupMetrics = runTask(
        group, detectorMap=detectorMap, calexp=calexp, pfsConfig=pfsConfig, thresholds=thresholds
    ).iqQaMetrics.iloc[0]

    assert (fullMetrics["nLitFibers"], fullMetrics["nFibers"]) == (20, 20)
    assert (groupMetrics["nLitFibers"], groupMetrics["nFibers"]) == (5, 20)
    assert fullMetrics["linesPerFiber"] == groupMetrics["linesPerFiber"] == 30
    assert groupMetrics["nTraceRows"] == 5 * 20
    assert groupMetrics["qaStatus"] == "PASS"
    # Negative control: the raw row count is a sixth of the full frame's, so a count of
    # rows or a scaling by lit fibers alone would fail it.
    assert groupMetrics["nLines"] * 4 < fullMetrics["nLines"] * 0.6


def testObstructedArcFailsPerLitFiber(runTask, fakes):
    """An arc whose lit fibers measure few lines fails, however many fibers the design lights."""
    detectorMap = fakes.DetectorMap(numFibers=20)
    calexp = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*ARC))
    thresholds = [lowerIsWorse("linesPerFiber", warn=28, fail=25, seqName="Arc: Neon")]
    litIds, pfsConfig = fakes.oneGroup(detectorMap)
    # 30 lines per fiber, 15 of them unmeasured (flagged) on each of the 5 lit fibers.
    lines = arcWithTraces(fakes, detectorMap, litIds, flagged=75)
    metrics = runTask(
        lines, detectorMap=detectorMap, calexp=calexp, pfsConfig=pfsConfig, thresholds=thresholds
    ).iqQaMetrics.iloc[0]
    assert metrics["linesPerFiber"] == 15
    assert (metrics["qaStatus"], metrics["qaDecidedBy"]) == ("FAIL", "linesPerFiber")


def testPctFlaggedIsOverLampLines(runTask, fakes):
    """Trace centroids, rarely flagged, don't dilute the flag rate of the lamp lines."""
    detectorMap = fakes.DetectorMap(numFibers=20)
    calexp = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*ARC))
    litIds, pfsConfig = fakes.oneGroup(detectorMap)
    few = arcWithTraces(fakes, detectorMap, litIds, flagged=75, tracesPerFiber=10)
    many = arcWithTraces(fakes, detectorMap, litIds, flagged=75, tracesPerFiber=200)
    fewRate = runTask(few, detectorMap=detectorMap, calexp=calexp, pfsConfig=pfsConfig).iqQaMetrics
    manyRate = runTask(many, detectorMap=detectorMap, calexp=calexp, pfsConfig=pfsConfig).iqQaMetrics
    assert fewRate["pctFlagged"].iloc[0] == manyRate["pctFlagged"].iloc[0] == pytest.approx(50.0)
    # Negative control: over every row, the rates would differ with the number of traces.
    assert 100.0 * 75 / len(few) != pytest.approx(100.0 * 75 / len(many))


def testOneGroupQuartzIsMeasuredOnLitFibers(runTask, fakes):
    """A one-group quartz is measured from the calexp, and judged per lit fiber."""
    detectorMap = fakes.DetectorMap(numFibers=20)
    litIds, pfsConfig = fakes.oneGroup(detectorMap)
    calexp = fakes.Exposure(fakes.traceImage(detectorMap, sigma=1.3, fibers=litIds), fakes.header(*TRACE))
    lines = fakes.arcLines(detectorMap, 380, fibers=litIds, description="Trace")
    profiles = fakes.FiberProfiles(detectorMap, sigma=1.0)
    thresholds = [lowerIsWorse("linesPerFiber", warn=370, fail=360, seqName="Trace")]

    metrics = runTask(
        lines,
        detectorMap=detectorMap,
        calexp=calexp,
        pfsConfig=pfsConfig,
        fiberProfiles=profiles,
        thresholds=thresholds,
    ).iqQaMetrics.iloc[0]
    assert not metrics["traceOnly"]  # the calexp, not fiberProfiles
    assert metrics["linesPerFiber"] == 380
    assert metrics["qaStatus"] == "PASS"

    # Negative control: without the pfsConfig the dark fibers flag most samples, and
    # the width falls back to fiberProfiles; nor can the count be judged.
    metrics = runTask(
        lines, detectorMap=detectorMap, calexp=calexp, fiberProfiles=profiles, thresholds=thresholds
    ).iqQaMetrics.iloc[0]
    assert metrics["traceOnly"]
    assert np.isnan(metrics["linesPerFiber"]) and np.isnan(metrics["nLitFibers"])


def testObstructedQuartzFails(runTask, fakes):
    """A quartz whose design lights every fiber, with traces found on a third of them, fails."""
    detectorMap = fakes.DetectorMap(numFibers=21)
    allLit = fakes.PfsConfig(detectorMap.fiberId)
    found = detectorMap.fiberId[::3]
    calexp = fakes.Exposure(fakes.traceImage(detectorMap, sigma=1.3, fibers=found), fakes.header(*TRACE))
    lines = fakes.arcLines(detectorMap, 380, fibers=found, description="Trace")
    thresholds = [lowerIsWorse("linesPerFiber", warn=370, fail=360, seqName="Trace")]
    metrics = runTask(
        lines, detectorMap=detectorMap, calexp=calexp, pfsConfig=allLit, thresholds=thresholds
    ).iqQaMetrics.iloc[0]
    assert metrics["linesPerFiber"] == pytest.approx(380 / 3)
    assert metrics["qaStatus"] == "FAIL"


def _getLamps(md):
    """obs_pfs's ``getLamps`` for the headers used here, with its DATA-TYP shortcut."""
    if (md.get("DATA-TYP") or "").lower().strip() == "flat":
        return {"Quartz"}
    menu = {"W_IISQTH": "Quartz_eng", "W_IISNEO": "Ne_eng", "W_AITQTH": "Quartz", "W_AITNEO": "Ne"}
    return {menu[key] for key in menu if md.get(key, False)}


def testIisQuartzIsNotAFullSlit(iqqa, runTask, fakes, monkeypatch):
    """An IIS quartz, DATA-TYP FLAT, is IIS: sparse, and not judged on the science fibers (PIPE2D-1953)."""
    monkeypatch.setitem(sys.modules, "lsst.obs.pfs.utils", ModuleType("lsst.obs.pfs.utils"))
    monkeypatch.setattr(sys.modules["lsst.obs.pfs.utils"], "getLamps", _getLamps, raising=False)
    iis = {"DATA-TYP": "FLAT", "W_IISQTH": True, **fakes.header("scienceTrace", "ImageQuality (Halogen)")}
    slit = {"DATA-TYP": "FLAT", "W_AITQTH": True, **fakes.header(*TRACE)}
    task = iqqa.ImageQualityQaTask(config=iqqa.ImageQualityQaConfig())
    assert task._classifyVisit(fakes.Exposure(np.zeros((2, 2)), iis), None)[:2] == ("trace", True)
    assert task._classifyVisit(fakes.Exposure(np.zeros((2, 2)), slit), None)[:2] == ("trace", False)
    # Negative control: getLamps on the header as it is takes the IIS quartz for a science-fiber one.
    assert _getLamps(iis) == {"Quartz"}

    detectorMap = fakes.DetectorMap(numFibers=20)
    lines = fakes.arcLines(detectorMap, 380, fibers=detectorMap.fiberId[:2], description="Trace")
    calexp = fakes.Exposure(fakes.noiseImage(detectorMap), iis)
    thresholds = [lowerIsWorse("linesPerFiber", warn=370, fail=360, seqName="ImageQuality (Halogen)")]
    metrics = runTask(
        lines,
        detectorMap=detectorMap,
        calexp=calexp,
        pfsConfig=fakes.PfsConfig(detectorMap.fiberId),
        fiberProfiles=fakes.FiberProfiles(detectorMap, sigma=1.0),
        thresholds=thresholds,
    ).iqQaMetrics.iloc[0]
    assert np.isnan(metrics["linesPerFiber"]) and not metrics["traceOnly"]
    assert metrics["qaStatus"] == "UNKNOWN"
