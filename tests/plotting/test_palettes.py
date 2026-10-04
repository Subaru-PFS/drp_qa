"""Tests for `pfs.drp.qa.plotting.palettes`."""

import pytest
from matplotlib.figure import Figure

import pfs.drp.qa.plotting.palettes as palettes
import pfs.drp.qa.utils.plotting as legacy
from pfs.drp.qa.plotting import scatterplot_with_outliers


def testScatterplotWithOutliers(makeArcData):
    ax = Figure().subplots()
    data = makeArcData().assign(isOutlier=False)

    result = scatterplot_with_outliers(data, "fiberId", "xResid", hue="status", ymin=-0.05, ymax=0.05, ax=ax)

    assert result is ax
    assert ax.get_ylim() == (-0.05, 0.05)


@pytest.mark.parametrize("name", legacy.__all__)
def testLegacyModule(name):
    """``pfs.drp.qa.utils.plotting`` re-exports every name."""
    assert getattr(legacy, name) is getattr(palettes, name)
