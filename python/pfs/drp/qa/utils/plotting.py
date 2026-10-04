"""Re-exports `pfs.drp.qa.plotting.palettes`; import from there in new code."""

from pfs.drp.qa.plotting.palettes import (
    description_palette,
    detector_palette,
    div_palette,
    opaqueColorbar,
    scatterplot_with_outliers,
    spectrograph_plot_markers,
)

__all__ = [
    "description_palette",
    "detector_palette",
    "div_palette",
    "opaqueColorbar",
    "scatterplot_with_outliers",
    "spectrograph_plot_markers",
]
