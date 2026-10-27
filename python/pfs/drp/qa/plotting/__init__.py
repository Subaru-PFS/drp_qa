"""Plotting for ``drp_qa``: DataFrames in, figures out.

Every function here takes DataFrames (and plain numbers) and returns a
`matplotlib.figure.Figure` (or draws on given axes). Nothing here imports the
Butler, the LSST/PFS stack or a task class, so the plots can be drawn from
stored data in a notebook and tested without the stack. Binding figures to a
Butler storage class is the task's job.

The old locations (``pfs.drp.qa.iqQaPlots`` and ``pfs.drp.qa.utils.plotting``)
re-export these names. The ``dmResiduals`` and ``dmCombined`` plots went with
their tasks (PIPE2D-1956): ``dmLineResidualData`` is plotted after the fact.
"""

from pfs.drp.qa.plotting.iqQa import plotIqTimeSeries
from pfs.drp.qa.plotting.palettes import (
    description_palette,
    detector_palette,
    div_palette,
    opaqueColorbar,
    scatterplot_with_outliers,
    spectrograph_plot_markers,
)
from pfs.drp.qa.plotting.thresholds import plotThresholds, plotTolerances

__all__ = [
    "description_palette",
    "detector_palette",
    "div_palette",
    "opaqueColorbar",
    "plotIqTimeSeries",
    "plotThresholds",
    "plotTolerances",
    "scatterplot_with_outliers",
    "spectrograph_plot_markers",
]
