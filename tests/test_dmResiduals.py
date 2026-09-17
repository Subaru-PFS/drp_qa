"""Tests for ``dmResiduals``' fit statistics.

Stack-dependent: ``pfs.drp.qa.dmResiduals`` imports ``lsst.pipe.base`` and
``pfs.drp.stella`` at module scope, so the module is fetched with
`pytest.importorskip` and the whole file skips where the stack is absent. The
guard is at module level on purpose -- a ``try/except ImportError`` inside a test
body never runs, because the import has already failed during collection.

What is tested is `get_fit_stats`, which is a pure function over a DataFrame.
Each test injects a defect of known size and asserts the statistic recovers it,
rather than asserting that the code merely ran.
"""

import numpy as np
import pandas as pd
import pytest

dmResiduals = pytest.importorskip(
    "pfs.drp.qa.dmResiduals",
    reason="requires the LSST/PFS stack (lsst.pipe.base, pfs.drp.stella)",
)


def makeArcData(numFibers=20, numLines=10, xResid=0.0, yResid=0.0, scatter=0.0, seed=0):
    """Build a synthetic per-line residual frame.

    Parameters
    ----------
    numFibers : `int`, optional
        Number of fibers; each contributes one trace row and ``numLines`` line
        rows.
    numLines : `int`, optional
        Emission lines per fiber.
    xResid : `float`, optional
        Spatial residual to inject into every row, in pixels.
    yResid : `float`, optional
        Wavelength residual to inject into every line row, in pixels.
    scatter : `float`, optional
        Standard deviation of Gaussian noise added to the residuals.
    seed : `int`, optional
        Random seed.

    Returns
    -------
    `pandas.DataFrame`
        Columns as produced by ``get_data_and_stats``, restricted to what
        `get_fit_stats` reads.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fiberId in range(1, numFibers + 1):
        for index in range(numLines + 1):
            isTrace = index == 0
            rows.append(
                {
                    "fiberId": fiberId,
                    "wavelength": 400.0 + 40.0 * index,
                    "xResid": xResid + (rng.normal(0.0, scatter) if scatter else 0.0),
                    "yResid": (
                        np.nan if isTrace else yResid + (rng.normal(0.0, scatter) if scatter else 0.0)
                    ),
                    "xErr": 0.01,
                    "yErr": 0.01,
                    "isTrace": isTrace,
                    "isLine": not isTrace,
                    "xResidOutlier": False,
                    "yResidOutlier": False,
                }
            )
    return pd.DataFrame(rows)


class TestGetFitStats:
    def testZeroResidualsGiveZeroMedians(self):
        stats = dmResiduals.get_fit_stats(makeArcData())
        assert stats.spatial.median == pytest.approx(0.0, abs=1e-12)
        assert stats.wavelength.median == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("shift", [0.05, 0.2, -0.3])
    def testAnInjectedSpatialShiftIsRecovered(self, shift):
        """Shift every centroid by a known amount; the median must report it."""
        stats = dmResiduals.get_fit_stats(makeArcData(xResid=shift))
        assert stats.spatial.median == pytest.approx(shift)

    @pytest.mark.parametrize("shift", [0.05, 0.2, -0.3])
    def testAnInjectedWavelengthShiftIsRecovered(self, shift):
        stats = dmResiduals.get_fit_stats(makeArcData(yResid=shift))
        assert stats.wavelength.median == pytest.approx(shift)

    def testTraceRowsDoNotEnterTheWavelengthStatistics(self):
        """Traces constrain x only; a NaN yResid must not become a zero."""
        data = makeArcData(yResid=0.4)
        stats = dmResiduals.get_fit_stats(data)
        assert stats.wavelength.median == pytest.approx(0.4)
        assert stats.wavelength.num_lines == data.query("isLine").wavelength.nunique()

    def testWeightedRmsTracksTheInjectedScatter(self):
        scatter = 0.05
        stats = dmResiduals.get_fit_stats(makeArcData(scatter=scatter, seed=3))
        assert stats.spatial.weightedRms == pytest.approx(scatter, rel=0.3)
        assert stats.spatial.robustRms == pytest.approx(scatter, rel=0.3)

    def testFiberAndLineCounts(self):
        stats = dmResiduals.get_fit_stats(makeArcData(numFibers=12, numLines=7))
        assert stats.spatial.num_fibers == 12
        assert stats.wavelength.num_fibers == 12
        assert stats.wavelength.num_lines == 7

    def testOutliersAreExcludedWhenAsked(self):
        """``sigmaClipOnly`` drops flagged rows; a wild outlier must not move the median."""
        data = makeArcData()
        data.loc[data.index[:5], ["xResid", "xResidOutlier"]] = [99.0, True]
        clipped = dmResiduals.get_fit_stats(data, sigmaClipOnly=True)
        unclipped = dmResiduals.get_fit_stats(data, sigmaClipOnly=False)
        assert clipped.spatial.median == pytest.approx(0.0, abs=1e-12)
        assert unclipped.spatial.num_lines >= clipped.spatial.num_lines

    def testZeroDofDoesNotCrashTheSofteningSolve(self):
        """Zero dof divides by zero; bisect cannot start from a NaN endpoint.

        Reachable whenever the parameter count eats the whole sample -- a fiber
        left with one surviving line after an S/N cut. The same guard exists in
        drp_stella's calculateSoftening, for the same reason.
        """
        data = makeArcData(numFibers=2, numLines=1)
        stats = dmResiduals.get_fit_stats(data, numParams=4)
        assert stats.wavelength.softenFit == pytest.approx(0.0)

    def testNoLinesLeavesTheWavelengthBlockEmptyRatherThanWrong(self):
        """A trace-only frame must report no lines, not a fabricated statistic."""
        data = makeArcData(numLines=0)
        stats = dmResiduals.get_fit_stats(data)
        assert stats.wavelength.num_lines == 0
        assert np.isnan(stats.wavelength.median)


class TestPlottingIsReExported:
    def testLegacyImportPathsStillResolve(self):
        """Callers and notebooks import these from the task module."""
        from pfs.drp.qa.plotting.dmResiduals import plot_detectormap_residuals, plot_residual

        assert dmResiduals.plot_detectormap_residuals is plot_detectormap_residuals
        assert dmResiduals.plot_residual is plot_residual

    def testFitStatsIsReExported(self):
        from pfs.drp.qa.metrics.fitStats import FitStat, FitStats

        assert dmResiduals.FitStat is FitStat
        assert dmResiduals.FitStats is FitStats
