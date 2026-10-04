"""Tests for `pfs.drp.qa.dmResiduals.get_fit_stats`, which needs the LSST/PFS stack.

``get_fit_stats`` is a function of a DataFrame. Each test injects a defect of
known size and checks that the statistics recover it.
"""

import numpy as np
import pandas as pd
import pytest

import pfs.drp.qa.dmCombinedResiduals as dmCombinedResiduals
import pfs.drp.qa.dmResiduals as dmResiduals
import pfs.drp.qa.metrics.fitStats as fitStats
import pfs.drp.qa.plotting as plotting
from pfs.drp.qa.dmResiduals import get_fit_stats
from pfs.drp.stella.utils.math import robustRms


def makeArcData(
    numFibers: int = 20,
    numLines: int = 10,
    xResid: float = 0.0,
    yResid: float = 0.0,
    scatter: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Make the columns of ``dmQaResidualData`` that `get_fit_stats` reads.

    Parameters
    ----------
    numFibers : `int`, optional
        Number of fibers; each has one trace row and ``numLines`` line rows.
    numLines : `int`, optional
        Emission lines per fiber.
    xResid : `float`, optional
        Spatial residual added to every row (pixels).
    yResid : `float`, optional
        Wavelength residual added to every line row (pixels).
    scatter : `float`, optional
        Standard deviation of Gaussian noise added to the residuals (pixels).
    seed : `int`, optional
        Random seed.

    Returns
    -------
    `pandas.DataFrame`
        The residuals. Trace rows have no ``yResid``.
    """
    rng = np.random.default_rng(seed)
    numRows = numFibers * (numLines + 1)
    isTrace = np.tile(np.arange(numLines + 1) == 0, numFibers)
    return pd.DataFrame(
        {
            "fiberId": np.repeat(np.arange(1, numFibers + 1), numLines + 1),
            "wavelength": np.tile(400.0 + 40.0 * np.arange(numLines + 1), numFibers),
            "xResid": xResid + rng.normal(0.0, scatter, numRows),
            "yResid": np.where(isTrace, np.nan, yResid + rng.normal(0.0, scatter, numRows)),
            "xErr": 0.01,
            "yErr": 0.01,
            "isTrace": isTrace,
            "isLine": ~isTrace,
            "xResidOutlier": False,
            "yResidOutlier": False,
        }
    )


def testZeroResiduals():
    stats = get_fit_stats(makeArcData())
    for block in (stats.spatial, stats.wavelength):
        assert block.median == 0
        assert block.weightedRms == 0
        assert block.softenFit == 0


@pytest.mark.parametrize("shift", [0.05, 0.2, -0.3])
def testShifts(shift):
    """A shift in one direction is recovered there and doesn't leak into the other."""
    stats = get_fit_stats(makeArcData(xResid=shift))
    assert stats.spatial.median == pytest.approx(shift)
    assert stats.wavelength.median == 0

    stats = get_fit_stats(makeArcData(yResid=shift))
    assert stats.wavelength.median == pytest.approx(shift)
    assert stats.spatial.median == 0


def testScatter():
    scatter = 0.05
    data = makeArcData(scatter=scatter, seed=3)
    stats = get_fit_stats(data)

    assert stats.spatial.weightedRms == pytest.approx(scatter, rel=0.2)
    assert stats.spatial.robustRms == pytest.approx(robustRms(data.xResid.to_numpy()))
    # chi2/dof is (0.05 / 0.01)**2 = 25, so a softening near sqrt(24) * 0.01 is needed.
    assert stats.spatial.softenFit == pytest.approx(np.sqrt(24) * 0.01, rel=0.2)


def testSofteningBeyondMaxSoften():
    """A softening larger than ``maxSoften`` is NaN, not ``maxSoften``."""
    stats = get_fit_stats(makeArcData(scatter=0.5), maxSoften=0.1)
    assert np.isnan(stats.spatial.softenFit)


def testTraceRowsAreNotWavelengthData():
    """Trace rows have no yResid; they must not count as wavelength residuals of zero."""
    data = makeArcData(yResid=0.4)
    stats = get_fit_stats(data)

    assert stats.wavelength.median == pytest.approx(0.4)
    assert stats.wavelength.num_lines == 10
    assert stats.spatial.num_lines == 20  # traces


def testCounts():
    stats = get_fit_stats(makeArcData(numFibers=12, numLines=7))

    assert stats.spatial.num_fibers == 12
    assert stats.wavelength.num_fibers == 12
    assert stats.wavelength.num_lines == 7
    assert stats.spatial.dof == 12 * 8
    assert stats.wavelength.dof == 12 * 7


def testOutliers():
    data = makeArcData()
    data.loc[data.index[:5], ["xResid", "xResidOutlier"]] = [99.0, True]

    clipped = get_fit_stats(data, sigmaClipOnly=True)
    assert clipped.spatial.weightedRms == 0

    # Negative control: the same rows, kept.
    unclipped = get_fit_stats(data, sigmaClipOnly=False)
    assert unclipped.spatial.weightedRms > 1


def testNoLines():
    """A trace-only frame reports no lines, not a statistic of nothing."""
    stats = get_fit_stats(makeArcData(numLines=0))

    assert stats.wavelength.num_lines == 0
    assert np.isnan(stats.wavelength.median)
    assert stats.spatial.num_lines == 20


def testZeroDof():
    """With no degrees of freedom the softening can't be solved for, and is NaN."""
    # Two lines (yNum = 2) and four parameters: yDof = 2 - 4 / 2 = 0.
    # Zero residuals make chi2/dof 0/0, which bisect can't start from.
    stats = get_fit_stats(makeArcData(numFibers=2, numLines=1), numParams=4)
    assert stats.wavelength.dof == 0
    assert np.isnan(stats.wavelength.softenFit)
    assert stats.spatial.softenFit == 0  # xDof = 4 - 2

    # chi2/dof is inf.
    with np.errstate(divide="ignore"):
        stats = get_fit_stats(makeArcData(numFibers=2, numLines=1, scatter=0.05), numParams=4)
    assert np.isnan(stats.wavelength.softenFit)


@pytest.mark.parametrize(
    ("module", "name", "home"),
    [
        (dmResiduals, "FitStat", fitStats),
        (dmResiduals, "FitStats", fitStats),
        (dmResiduals, "plot_detectormap_residuals", plotting),
        (dmResiduals, "plot_residual", plotting),
        (dmCombinedResiduals, "plot_detector_summary", plotting),
        (dmCombinedResiduals, "plot_detector_summary_per_desc", plotting),
        (dmCombinedResiduals, "plot_title", plotting),
        (dmCombinedResiduals, "plot_visits", plotting),
    ],
)
def testReExports(module, name, home):
    assert getattr(module, name) is getattr(home, name)
