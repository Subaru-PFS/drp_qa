"""Plotting for ``drp_qa``: DataFrames in, figures out.

Every function here takes DataFrames (and plain numbers) and returns
`matplotlib.figure.Figure` objects. None of them imports the Butler or a task
class. That one constraint is what yields three consumers from a single
implementation -- the dashboard, an on-demand static report, and the notebooks
under ``examples/`` -- and what makes the plotting code unit-testable for the
first time. See ``doc/qa-rebuild-plan.md`` section 1.5.

The pipeline stores data, never a rendering of it (R6). Assembling figures into
a Butler artifact is therefore the task's job, not this package's:
`pfs.drp.qa.plotting.dmCombined.reportFigures` yields the pages and
``DetectorMapCombinedResidualsTask`` binds them to the storage class.
"""

from pfs.drp.qa.plotting.dmCombined import (
    plot_detector_summary,
    plot_detector_summary_per_desc,
    plot_title,
    plot_visits,
    reportFigures,
)
from pfs.drp.qa.plotting.dmResiduals import (
    DetectorGeometry,
    plot_detectormap_residuals,
    plot_residual,
)
from pfs.drp.qa.plotting.iqQa import plotIqTimeSeries
from pfs.drp.qa.plotting.palettes import (
    description_palette,
    detector_palette,
    div_palette,
    scatterplot_with_outliers,
    spectrograph_plot_markers,
)

__all__ = [
    "DetectorGeometry",
    "description_palette",
    "detector_palette",
    "div_palette",
    "plotIqTimeSeries",
    "plot_detector_summary",
    "plot_detector_summary_per_desc",
    "plot_detectormap_residuals",
    "plot_residual",
    "plot_title",
    "plot_visits",
    "reportFigures",
    "scatterplot_with_outliers",
    "spectrograph_plot_markers",
]
