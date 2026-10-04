"""Smoke tests for `pfs.drp.qa.plotting.dmResiduals`."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from pfs.drp.qa.plotting import DetectorGeometry, plot_detectormap_residuals, plot_residual


def texts(figure: Figure) -> str:
    """Return all the text drawn on a figure's axes, joined."""
    return "\n".join(text.get_text() for ax in figure.axes for text in ax.texts)


@pytest.mark.parametrize(("column", "which"), [("xResid", "spatial"), ("yResid", "wavelength")])
def testPlotResidual(makeArcData, makeStats, column, which):
    figure = plot_residual(makeArcData(), makeStats(), column=column, dataRange=0.1)

    assert isinstance(figure, Figure)
    # Fiber medians, stats block, 2D residuals, residuals by wavelength, colorbar.
    assert len(figure.axes) == 5
    assert figure.axes[0].get_title().startswith(f"Median {which} residual")
    assert "RESERVED:" in texts(figure)
    assert "USED:" in texts(figure)


def testPlotResidualBinsByWavelength(makeArcData, makeStats):
    # The binned aggregation must not warn: pandas deprecates apply() over the
    # grouping columns.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        figure = plot_residual(makeArcData(), makeStats(), column="yResid", dataRange=0.1, binWavelength=50)
    assert "binsize=50" in figure.axes[3].get_title()


def testPlotResidualNeedsReservedData(makeArcData, makeStats):
    data = makeArcData()
    data["isReserved"] = False
    with pytest.raises(ValueError, match="No data"):
        plot_residual(data, makeStats(), column="xResid", dataRange=0.1)


def testPlotDetectorMapResiduals(makeArcData, makeStats, geometry):
    figure = plot_detectormap_residuals(makeArcData(), makeStats(), geometry)

    assert isinstance(figure, Figure)
    assert [subfigure.get_suptitle() for subfigure in figure.subfigs] == ["xResid", "yResid"]
    # The wavelength panel spans the geometry's wavelength range.
    yResidAxes = figure.subfigs[1].axes
    assert yResidAxes[3].get_ylim() == (geometry.wavelengthMin, geometry.wavelengthMax)


def testGeometryFromDetectorMap(makeArcData, makeStats):
    """A DetectorMap is still accepted, and reduced to a `DetectorGeometry`."""
    detectorMap = SimpleNamespace(
        getBBox=lambda: SimpleNamespace(width=2048, height=4176),
        fiberId=np.arange(1, 9),
        metadata={"WAV-MIN": 380.0, "WAV-MAX": 700.0},
    )
    geometry = DetectorGeometry.coerce(detectorMap)

    assert geometry == DetectorGeometry(2048, 4176, 1, 8, 380.0, 700.0)
    assert DetectorGeometry.coerce(geometry) is geometry
    assert isinstance(plot_detectormap_residuals(makeArcData(), makeStats(), detectorMap), Figure)
