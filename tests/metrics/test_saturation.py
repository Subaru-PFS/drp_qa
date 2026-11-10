"""`pfs.drp.qa.metrics.saturation`: saturated lines and aperture pixels from a boolean plane."""

import numpy as np

from pfs.drp.qa.metrics.saturation import apertureSaturation, saturatedAt


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
