"""Tests for `pfs.drp.qa.metrics.fitStats`.

`FitStats.from_dataframe` unpacks each block positionally, so the order of
`FitStat`'s fields is part of the stored schema of ``dmQaResidualStats``.
"""

import dataclasses

import pandas as pd
import pytest

from pfs.drp.qa.metrics.fitStats import FitStat, FitStats


def makeFitStat(offset: float = 0.0) -> FitStat:
    """Make a `FitStat` with distinct values, shifted by ``offset``."""
    return FitStat(
        median=0.001 + offset,
        robustRms=0.020 + offset,
        weightedRms=0.025 + offset,
        softenFit=0.003 + offset,
        dof=100.0 + offset,
        num_fibers=8,
        num_lines=48,
    )


def makeFitStats(offset: float = 0.0) -> FitStats:
    """Make a `FitStats` whose wavelength block differs from its spatial one."""
    return FitStats(
        dof=200.0, chi2X=95.0, chi2Y=105.0, spatial=makeFitStat(offset), wavelength=makeFitStat(offset + 0.5)
    )


def testFieldOrder():
    assert [field.name for field in dataclasses.fields(FitStat)] == [
        "median",
        "robustRms",
        "weightedRms",
        "softenFit",
        "dof",
        "num_fibers",
        "num_lines",
    ]


def testStr():
    """The residual plots print this block."""
    assert str(makeFitStat()) == (
        "median  =  0.00100\nrms     =  0.02500\nsoften  =  0.00300\nfibers  =        8\nlines   =       48\n"
    )


def testToDict():
    asDict = makeFitStats().to_dict()
    assert set(asDict) == {"dof", "chi2X", "chi2Y", "spatial", "wavelength"}
    assert asDict["spatial"] == dataclasses.asdict(makeFitStat())


def testRoundTrip():
    """``json_normalize(to_dict())`` is how the task stores a row; the plots read it back."""
    original = makeFitStats()
    restored = FitStats.from_dataframe(pd.json_normalize(original.to_dict()))

    assert restored.dof == original.dof
    assert restored.chi2X == original.chi2X
    assert restored.chi2Y == original.chi2Y
    assert dataclasses.astuple(restored.spatial) == pytest.approx(dataclasses.astuple(original.spatial))
    assert dataclasses.astuple(restored.wavelength) == pytest.approx(dataclasses.astuple(original.wavelength))
    # Negative control: the blocks differ, so a swap would fail the checks above.
    assert restored.spatial.median != pytest.approx(original.wavelength.median)


def testSeveralRowsGiveTheirMedian():
    """The plots pass every row of one status_type."""
    frame = pd.json_normalize([makeFitStats(offset).to_dict() for offset in (0.0, 1.0, 5.0)])
    restored = FitStats.from_dataframe(frame)

    assert restored.spatial.median == pytest.approx(1.001)
    assert restored.wavelength.median == pytest.approx(1.501)


def testIdentifyingColumnsAreIgnored():
    """Stored rows carry text columns (``status_type``, ``ccd``...) beside the numbers."""
    frame = pd.json_normalize(makeFitStats().to_dict()).assign(status_type="RESERVED", ccd="b1")
    assert FitStats.from_dataframe(frame).spatial.median == pytest.approx(0.001)
