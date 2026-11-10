"""`pfs.drp.qa.metrics.saturation`: saturated lines and aperture pixels from a boolean plane."""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.saturation import (
    apertureSaturation,
    measureSaturation,
    saturatedAt,
    saturatedLineSummary,
)


def testSaturatedAtSearchesTheBoxAboutThePosition():
    """A saturated pixel next to the expected position counts; one two pixels away doesn't."""
    saturated = np.zeros((20, 30), dtype=bool)
    saturated[10, 15] = True  # row 10, column 15
    x = np.array([15.0, 16.2, 17.0, 15.0, np.nan, 500.0])
    y = np.array([10.0, 9.4, 10.0, 12.0, 10.0, 10.0])
    assert saturatedAt(saturated, x, y).tolist() == [True, True, False, False, False, False]

    # Negative control: the nearest pixel alone misses the neighbour, and x is the column,
    # not the row (the transposed lookup finds nothing at column 10, row 15).
    assert saturatedAt(saturated, x, y, halfSize=0).tolist() == [True, False, False, False, False, False]
    assert not saturatedAt(saturated, np.array([10.0]), np.array([15.0])).any()


def testSaturatedAtIgnoresPixelsOffTheImage():
    """A box hanging over the edge is searched where it is on the image."""
    saturated = np.zeros((5, 5), dtype=bool)
    saturated[0, 0] = True
    assert saturatedAt(saturated, np.array([0.0, -1.0, -2.0]), np.array([0.0, -1.0, 0.0])).tolist() == [
        True,
        True,
        False,
    ]


def testApertureSaturationCountsWithinTheApertures():
    """Saturated pixels are counted within the apertures only, over the apertures' pixels."""
    saturated = np.zeros((10, 40), dtype=bool)
    saturated[2:5, 10] = True  # 3 pixels on the first trace
    saturated[:, 30] = True  # a column between traces
    xCenters = np.vstack([np.full(10, 10.2), np.full(10, 20.0)])
    numSaturated, numPixels = apertureSaturation(saturated, xCenters, halfWidth=3)
    assert (numSaturated, numPixels) == (3, 2 * 10 * 7)

    # Negative control: an aperture reaching the bad column counts it.
    numSaturated, _ = apertureSaturation(saturated, xCenters, halfWidth=10)
    assert numSaturated == 3 + 10


def testApertureSaturationCountsSharedAndMissingPixelsOnce():
    """Overlapping apertures share pixels; rows without a trace and columns off the image have none."""
    saturated = np.ones((4, 12), dtype=bool)
    xCenters = np.array([[5.0, 5.0, np.nan, 0.0], [7.0, 7.0, np.nan, 0.0]])
    # Rows 0-1: columns 4-8 (5 pixels, not 2 x 3); row 2: none; row 3: columns 0-1 (-1 is off the image).
    assert apertureSaturation(saturated, xCenters, halfWidth=1) == (12, 12)
    assert apertureSaturation(np.zeros((4, 12), dtype=bool), xCenters, halfWidth=1) == (0, 12)


class FakeDetectorMap:
    """Vertical traces 10 px apart; wavelength 500 + 0.1 nm per row."""

    fiberId = np.array([1, 2, 3], dtype=np.int32)

    def getXCenter(self, fiberId, y):
        return 10.0 * np.asarray(fiberId, dtype=float) + 0.0 * np.asarray(y)

    def findPoint(self, fiberId, wavelength):
        y = (np.asarray(wavelength, dtype=float) - 500.0) / 0.1
        return np.column_stack([self.getXCenter(fiberId, y), y])


def makeDetector():
    """Return a 50 x 40 plane with fiber 2's line at row 20 saturated, and the three fibers' two lines."""
    saturated = np.zeros((50, 40), dtype=bool)
    saturated[19:22, 19:22] = True
    lines = pd.DataFrame(
        {
            "fiberId": [1, 2, 3, 1, 2, 3, 2],
            "wavelength": [502.0, 502.0, 502.0, 504.0, 504.0, 504.0, np.nan],
            "description": ["NeI"] * 6 + ["Trace"],
        }
    )
    return saturated, lines


def testMeasureSaturationPutsLinesAtTheirWavelength():
    """A line is looked up at the detectorMap's position for its wavelength; a trace centroid nowhere."""
    saturated, lines = makeDetector()
    lineSaturated, x, y, pct = measureSaturation(saturated, FakeDetectorMap(), lines)
    assert lineSaturated.tolist() == [False, True, False, False, False, False, False]
    assert (x[1], y[1]) == (20.0, 20.0) and np.isnan(x[-1])
    assert pct == pytest.approx(100.0 * 9 / (3 * 50 * 7))

    # Lit fibers alone: fiber 2 hidden leaves no saturated aperture pixel.
    assert measureSaturation(saturated, FakeDetectorMap(), lines, litFiberIds={1, 3})[3] == 0.0
    assert np.isnan(measureSaturation(saturated, FakeDetectorMap(), lines, litFiberIds=set())[3])

    # Negative control: the same line 2 nm (20 rows) along is not saturated.
    assert not measureSaturation(saturated, FakeDetectorMap(), lines.assign(wavelength=504.0))[0].any()


def testSaturatedLineSummaryCountsFibersPerLine():
    """Each saturated line is listed with the fibers it is saturated on, the most saturated first."""
    _, lines = makeDetector()
    lines = lines.iloc[:6].assign(saturated=[False, True, False, True, True, False])
    summary = saturatedLineSummary(lines)
    assert summary["wavelength"].tolist() == [504.0, 502.0]
    assert summary["nSaturated"].tolist() == [2, 1]
    assert summary["nFibers"].tolist() == [3, 3]
    assert len(saturatedLineSummary(lines.assign(saturated=False))) == 0
