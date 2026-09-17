"""Fit statistics for detector-map residuals.

`FitStats` is the shape of one row of ``dmQaResidualStats``: per
``(status_type, description)``, a spatial and a wavelength block of error-weighted
and robust statistics. It lives here, away from the task, so that the plotting
library can render a stats block without importing a `PipelineTask` -- and so
that both can be tested without the LSST stack.

``pfs.drp.qa.dmResiduals`` re-exports both classes, so existing imports keep
working.
"""

from dataclasses import dataclass

import pandas as pd

__all__ = ["FitStat", "FitStats"]


@dataclass
class FitStat:
    median: float
    robustRms: float
    weightedRms: float
    softenFit: float
    dof: float
    num_fibers: int
    num_lines: int

    def __str__(self):
        return f"""median  = {self.median:> 7.05f}
rms     = {self.weightedRms:> 7.05f}
soften  = {self.softenFit:> 7.05f}
fibers  = {int(self.num_fibers):>8d}
lines   = {int(self.num_lines):>8d}
"""


@dataclass
class FitStats:
    dof: int
    chi2X: float
    chi2Y: float
    spatial: FitStat
    wavelength: FitStat

    def to_dict(self):
        """Output as dict."""
        return {
            "dof": self.dof,
            "chi2X": self.chi2X,
            "chi2Y": self.chi2Y,
            "spatial": self.spatial.__dict__,
            "wavelength": self.wavelength.__dict__,
        }

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Convert from dataframe to FitStats."""
        try:
            df = df.select_dtypes(include="number").median().to_frame().T

            reserved_wl = df.filter(like="wavelength.").copy()
            reserved_spatial = df.filter(like="spatial.").copy()

            reserved_wl.columns = reserved_wl.columns.str.rsplit(".", n=1).str[-1]
            reserved_spatial.columns = reserved_spatial.columns.str.rsplit(".", n=1).str[-1]

            rec = df.iloc[0]
            fs = cls(
                dof=rec.dof,
                chi2X=rec.chi2X,
                chi2Y=rec.chi2Y,
                spatial=FitStat(*reserved_spatial.iloc[0].to_list()),
                wavelength=FitStat(*reserved_wl.iloc[0].to_list()),
            )
        except Exception as e:
            print(f"Error: {e!r}")
        else:
            return fs
