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

    # A position belongs to the pixel whose centre is nearest: 15.6 is pixel 16, 14.6 pixel 15.
    assert saturatedAt(saturated, np.array([14.6, 15.6]), np.array([10.4, 10.0]), halfSize=0).tolist() == [
        True,
        False,
    ]
    # Negative control: truncated, 15.6 would be pixel 15 and count.
    assert saturated[10, int(15.6)]


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

    # The aperture is about the nearest pixel to the trace centre: at 26.6 that is column 27, whose
    # aperture (24-30) reaches the bad column; truncated to 26 (23-29) it would not.
    numSaturated, _ = apertureSaturation(saturated, np.full((1, 10), 26.6), halfWidth=3)
    assert numSaturated == 10
    assert apertureSaturation(saturated, np.full((1, 10), 26.4), halfWidth=3)[0] == 0


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
    """Return a 50 x 40 plane saturated at fiber 2, row 20; the fibers' two lines; fiber 2's trace centroid there."""
    saturated = np.zeros((50, 40), dtype=bool)
    saturated[19:22, 19:22] = True
    lines = pd.DataFrame(
        {
            "fiberId": [1, 2, 3, 1, 2, 3, 2],
            "wavelength": [502.0, 502.0, 502.0, 504.0, 504.0, 504.0, 502.0],
            "description": ["NeI"] * 6 + ["Trace"],
        }
    )
    return saturated, lines


def testMeasureSaturationPutsLinesAtTheirWavelength():
    """A row is looked up at the detectorMap's position for its wavelength, a trace centroid's too."""
    saturated, lines = makeDetector()
    lineSaturated, x, y, pct = measureSaturation(saturated, FakeDetectorMap(), lines)
    # The last row is fiber 2's trace centroid on the saturated pixels: drp_stella gives trace
    # centroids their row's wavelength, so it is marked, and the caller counts lamp lines alone.
    assert lineSaturated.tolist() == [False, True, False, False, False, False, True]
    assert (x[1], y[1]) == (20.0, 20.0)
    noWavelength = measureSaturation(saturated, FakeDetectorMap(), lines.assign(wavelength=np.nan))
    assert not noWavelength[0].any() and np.isnan(noWavelength[1]).all()
    assert pct == pytest.approx(100.0 * 9 / (3 * 50 * 7))

    # Lit fibers alone: fiber 2 hidden leaves no saturated aperture pixel.
    assert measureSaturation(saturated, FakeDetectorMap(), lines, litFiberIds={1, 3})[3] == 0.0
    assert np.isnan(measureSaturation(saturated, FakeDetectorMap(), lines, litFiberIds=set())[3])

    # Negative control: the same line 2 nm (20 rows) along is not saturated.
    assert not measureSaturation(saturated, FakeDetectorMap(), lines.assign(wavelength=504.0))[0].any()


def testSaturatedLineSummaryCountsFibersPerLine():
    """Each saturated line is listed with the fibers it is saturated on, in order of wavelength."""
    _, lines = makeDetector()
    lines = lines.iloc[:6].assign(saturated=[False, True, False, True, True, False])
    summary = saturatedLineSummary(lines)
    assert summary["wavelength"].tolist() == [502.0, 504.0]
    assert summary["nSaturated"].tolist() == [1, 2]  # by wavelength, not by how saturated
    assert summary["nFibers"].tolist() == [3, 3]
    assert len(saturatedLineSummary(lines.assign(saturated=False))) == 0
    # No lines at all (a quartz): the same columns, typed, so that it sorts like any other.
    empty = saturatedLineSummary(lines.iloc[:0])
    assert list(empty.columns) == list(summary.columns) and len(empty) == 0
    assert len(empty["nSaturated"].nlargest(3)) == 0
