"""Backwards-compatible alias for `pfs.drp.qa.plotting.palettes`.

The plotting code moved to `pfs.drp.qa.plotting` so that it takes DataFrames
and returns figures without importing the Butler or a task class; see
``doc/qa-rebuild-plan.md`` section 1.5. This module re-exports the palettes and
`scatterplot_with_outliers` from their new home. Prefer importing from
``pfs.drp.qa.plotting`` in new code.
"""

from pfs.drp.qa.plotting.palettes import (
    description_palette,
    detector_palette,
    div_palette,
    scatterplot_with_outliers,
    spectrograph_plot_markers,
)

__all__ = [
    "description_palette",
    "detector_palette",
    "div_palette",
    "scatterplot_with_outliers",
    "spectrograph_plot_markers",
]
