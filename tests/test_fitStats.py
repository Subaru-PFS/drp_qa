"""Tests for the detector-map fit statistics containers.

Stack-free. `FitStats.from_dataframe` unpacks a row *positionally*, so the order
of `FitStat`'s fields is part of the stored schema of ``dmQaResidualStats``:
reorder them and every stats block silently reports the wrong numbers under the
right labels. That is what these tests pin.
"""

import dataclasses

import pandas as pd
import pytest

from pfs.drp.qa.metrics.fitStats import FitStat, FitStats


def makeFitStat(offset=0.0):
    """Build a `FitStat` with recognisably distinct field values.

    Parameters
    ----------
    offset : `float`, optional
        Added to every value, so the spatial and wavelength blocks differ.

    Returns
    -------
    `FitStat`
        The statistics.
    """
    return FitStat(
        median=0.001 + offset,
        robustRms=0.020 + offset,
        weightedRms=0.025 + offset,
        softenFit=0.003 + offset,
        dof=100.0 + offset,
        num_fibers=8,
        num_lines=48,
    )


class TestFitStat:
    def testFieldOrderIsPartOfTheStoredSchema(self):
        """`from_dataframe` unpacks positionally; this order must not drift."""
        assert tuple(field.name for field in dataclasses.fields(FitStat)) == (
            "median",
            "robustRms",
            "weightedRms",
            "softenFit",
            "dof",
            "num_fibers",
            "num_lines",
        )

    def testStrReportsTheWeightedRms(self):
        """The plots print this block verbatim, so it must stay readable."""
        text = str(makeFitStat())
        assert "median" in text
        assert "rms" in text
        assert "fibers  =        8" in text


class TestFitStats:
    def testRoundTripThroughTheStoredFrame(self):
        """to_dict -> json_normalize -> from_dataframe is how the task stores and the plots read."""
        original = FitStats(
            dof=200.0,
            chi2X=95.0,
            chi2Y=105.0,
            spatial=makeFitStat(),
            wavelength=makeFitStat(offset=0.5),
        )
        frame = pd.json_normalize(original.to_dict())
        restored = FitStats.from_dataframe(frame)

        assert restored.dof == pytest.approx(original.dof)
        assert restored.chi2X == pytest.approx(original.chi2X)
        assert restored.spatial.weightedRms == pytest.approx(original.spatial.weightedRms)
        assert restored.wavelength.weightedRms == pytest.approx(original.wavelength.weightedRms)
        assert restored.spatial.num_lines == pytest.approx(original.spatial.num_lines)

    def testBlocksAreNotSwapped(self):
        """A spatial/wavelength mix-up is silent and catastrophic; assert they differ."""
        original = FitStats(
            dof=200.0,
            chi2X=95.0,
            chi2Y=105.0,
            spatial=makeFitStat(),
            wavelength=makeFitStat(offset=0.5),
        )
        restored = FitStats.from_dataframe(pd.json_normalize(original.to_dict()))
        assert restored.spatial.median == pytest.approx(0.001)
        assert restored.wavelength.median == pytest.approx(0.501)

    def testSeveralRowsAreReducedToTheirMedian(self):
        """The plots pass every row of one status_type; from_dataframe medians them."""
        rows = [
            FitStats(200.0, 95.0, 105.0, makeFitStat(o), makeFitStat(o)).to_dict() for o in (0.0, 1.0, 2.0)
        ]
        restored = FitStats.from_dataframe(pd.json_normalize(rows))
        assert restored.spatial.median == pytest.approx(1.001)

    def testToDictKeepsTheNestedShape(self):
        asDict = FitStats(1.0, 2.0, 3.0, makeFitStat(), makeFitStat()).to_dict()
        assert set(asDict) == {"dof", "chi2X", "chi2Y", "spatial", "wavelength"}
        assert asDict["spatial"]["num_fibers"] == 8
