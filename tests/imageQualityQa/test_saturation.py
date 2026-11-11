"""``imageQualityQa``'s saturation and per-species line flux, from fakes of its inputs."""

import numpy as np
import pandas as pd
import pytest

ARC = ("scienceArc", "Arc: Neon")
SAT, CR = 2, 1


def planes(names) -> int:
    """Return the bit mask of the mask planes ``names``: ``SAT`` is its own bit."""
    return sum({"SAT": SAT}.get(name, CR) for name in names)


def makeInputs(fakes, numPerFiber: int = 5):
    """Return a detectorMap, an arc's lines of two species and its calexp, nothing saturated."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=numPerFiber, description="NeI")
    lines["wavelength"] = lines["lam"]
    lines.loc[lines["fiberId"] > 5, "description"] = "ArI"
    lines.loc[lines["description"] == "ArI", "flux"] = 50.0
    calexp = fakes.Exposure(fakes.traceImage(detectorMap, sigma=1.2), fakes.header(*ARC))
    calexp.mask.getPlaneBitMask = planes
    return detectorMap, lines, calexp


def longValue(frame: pd.DataFrame, description: str, metric: str) -> float:
    """Return the one value of ``metric`` for ``description`` in a long-format table."""
    rows = frame[(frame["description"] == description) & (frame["metric"] == metric)]
    assert len(rows) == 1
    return float(rows["value"].iloc[0])


def testUnsaturatedFrame(runTask, fakes):
    """Nothing saturated: zero, not null, and the species' fluxes are their own."""
    detectorMap, lines, calexp = makeInputs(fakes)
    outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["nSaturatedLines"] == 0
    assert metrics["pctSaturatedPixels"] == 0.0

    species = outputs.iqQaSpeciesMetrics
    assert longValue(species, "NeI", "medLineFlux") == pytest.approx(1000.0)
    assert longValue(species, "ArI", "medLineFlux") == pytest.approx(50.0)
    assert longValue(species, "NeI", "robustRmsLineFlux") == pytest.approx(0.0)
    assert longValue(species, "NeI", "nFluxLines") == 25
    assert longValue(species, "ArI", "nSaturatedLines") == 0


def testSaturatedLinesAreFoundAtTheirExpectedPosition(runTask, fakes):
    """A line with ``SAT`` at the detectorMap's position counts, though its centroid was not measured."""
    detectorMap, lines, calexp = makeInputs(fakes)
    first = lines.index[0]  # fiber 1, NeI
    row, column = round(lines.loc[first, "y"]), round(lines.loc[first, "x"])
    calexp.mask.array[row - 1 : row + 2, column - 1 : column + 2] |= SAT
    # As drp_stella leaves a saturated line: flagged, with no centroid.
    lines.loc[first, ["x", "y", "flux"]] = np.nan
    lines.loc[first, "flag"] = True
    # A cosmic ray on an ArI line is not saturation.
    last = lines.index[-1]
    calexp.mask.array[round(lines.loc[last, "y"]), round(lines.loc[last, "x"])] |= CR

    outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["nSaturatedLines"] == 1
    # 9 SAT pixels, all within fiber 1's 7-column aperture; 10 fibers x 400 rows x 7 columns.
    assert metrics["pctSaturatedPixels"] == pytest.approx(100.0 * 9 / (10 * 400 * 7))
    species = outputs.iqQaSpeciesMetrics
    assert longValue(species, "NeI", "nSaturatedLines") == 1
    assert longValue(species, "ArI", "nSaturatedLines") == 0
    assert longValue(species, "NeI", "nFluxLines") == 24  # the saturated line gives no flux

    # Negative control: read as SAT, the cosmic-ray plane would add the ArI line.
    calexp.mask.getPlaneBitMask = lambda names: SAT | CR
    outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp)
    assert outputs.iqQaMetrics["nSaturatedLines"].iloc[0] == 2


def testSaturationIsOfLitFibers(runTask, fakes):
    """Saturation on a fiber the design hides is in neither count; the denominator is the lit apertures."""
    detectorMap, lines, calexp = makeInputs(fakes)
    lit, pfsConfig = fakes.oneGroup(detectorMap)  # fibers 1, 5, 9
    hidden = lines[lines["fiberId"] == 2].index[0]
    calexp.mask.array[round(lines.loc[hidden, "y"]), round(lines.loc[hidden, "x"])] |= SAT
    litLine = lines[lines["fiberId"] == 5].index[0]
    calexp.mask.array[round(lines.loc[litLine, "y"]), round(lines.loc[litLine, "x"])] |= SAT

    outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp, pfsConfig=pfsConfig)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["nSaturatedLines"] == 1
    assert metrics["pctSaturatedPixels"] == pytest.approx(100.0 * 1 / (len(lit) * 400 * 7))
    # The species rows are of lit fibers too: fibers 1 and 5 are NeI, fiber 9 ArI, 5 lines each.
    species = outputs.iqQaSpeciesMetrics
    assert longValue(species, "NeI", "nSaturatedLines") == 1
    assert longValue(species, "NeI", "nFluxLines") == 10
    assert longValue(species, "ArI", "nFluxLines") == 5

    # Negative control: without the pfsConfig every fiber is taken as lit, and both count.
    outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp)
    assert outputs.iqQaMetrics["nSaturatedLines"].iloc[0] == 2
    assert longValue(outputs.iqQaSpeciesMetrics, "NeI", "nSaturatedLines") == 2
    assert longValue(outputs.iqQaSpeciesMetrics, "NeI", "nFluxLines") == 25


def testSaturatedQuartz(runTask, fakes):
    """A saturated quartz shows in the aperture pixels; its trace centroids are not saturated lines."""
    detectorMap = fakes.DetectorMap()
    # As drp_stella writes a quartz's ``lines``: trace centroids alone, each with its row's wavelength.
    lines = fakes.arcLines(detectorMap, numPerFiber=40, description="Trace")
    lines["wavelength"] = lines["lam"]
    calexp = fakes.Exposure(fakes.traceImage(detectorMap, sigma=1.2), fakes.header("scienceTrace", "Trace"))
    calexp.mask.getPlaneBitMask = planes
    # Fiber 3's trace is saturated over rows 100-299: its central 3 columns.
    column = round(float(detectorMap.getXCenter(3, 0.0)))
    calexp.mask.array[100:300, column - 1 : column + 2] |= SAT

    outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "trace"
    assert metrics["pctSaturatedPixels"] == pytest.approx(100.0 * 200 * 3 / (10 * 400 * 7))
    assert metrics["nSaturatedLines"] == 0
    assert "nSaturatedLines" not in set(outputs.iqQaSpeciesMetrics["metric"])

    # Negative control: the same rows as lamp lines are saturated lines (those of fiber 3 in rows
    # 99-300, within a pixel of the saturated ones), so it is the Trace description that excludes them.
    asLines = lines.assign(description="NeI")
    onTrace = (asLines["fiberId"] == 3) & asLines["y"].round().between(99, 300)
    assert onTrace.sum() > 0
    outputs = runTask(asLines, detectorMap=detectorMap, calexp=calexp)
    assert outputs.iqQaMetrics["nSaturatedLines"].iloc[0] == onTrace.sum()


def testNoCalexpGivesNoSaturation(runTask, fakes):
    """Without a calexp saturation is null, not zero; the line fluxes need only ``lines``."""
    detectorMap, lines, _ = makeInputs(fakes)
    outputs = runTask(lines, detectorMap=detectorMap)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert np.isnan(metrics["nSaturatedLines"]) and np.isnan(metrics["pctSaturatedPixels"])
    species = outputs.iqQaSpeciesMetrics
    assert "nSaturatedLines" not in set(species["metric"])
    assert longValue(species, "NeI", "medLineFlux") == pytest.approx(1000.0)


def testNeitherMetricIsJudged(runTask, fakes):
    """A wholly saturated frame keeps the verdict its other metrics give: no threshold exists yet."""
    detectorMap, lines, calexp = makeInputs(fakes, numPerFiber=30)
    clean = runTask(lines, detectorMap=detectorMap, calexp=calexp).iqQaMetrics.iloc[0]
    calexp.mask.array[:] |= SAT
    saturated = runTask(lines, detectorMap=detectorMap, calexp=calexp).iqQaMetrics.iloc[0]
    assert saturated["pctSaturatedPixels"] == pytest.approx(100.0)
    assert saturated["nSaturatedLines"] == len(lines)
    assert (saturated["qaStatus"], saturated["qaReason"]) == (clean["qaStatus"], clean["qaReason"])
