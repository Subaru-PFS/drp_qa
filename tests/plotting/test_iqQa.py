"""Smoke tests for `pfs.drp.qa.plotting.iqQa`."""

from matplotlib.figure import Figure

from pfs.drp.qa.plotting import plotIqTimeSeries


def testPlotIqTimeSeries(makeIqMetrics):
    figure = plotIqTimeSeries(makeIqMetrics(), title="Run 99")
    assert isinstance(figure, Figure)
    assert figure.get_suptitle() == "Run 99"


def testLegacyModule():
    """``pfs.drp.qa.iqQaPlots`` re-exports the function."""
    from pfs.drp.qa.iqQaPlots import plotIqTimeSeries as legacy

    assert legacy is plotIqTimeSeries
