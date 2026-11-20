"""Shared fixtures for the ``pfs.drp.qa.plotting`` tests.

These tests need numpy, pandas, matplotlib, seaborn and scipy but not the LSST stack.
CI runs them in the ``deps`` job; the standard-library job ignores this
directory.

Test modules can't import each other (``--import-mode=importlib``), so the
synthetic data builders are handed out by fixtures.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy  # noqa: F401 (pfs.drp.qa.metrics needs it; listed so the directory is skipped without it)
import seaborn  # noqa: F401 (pfs.drp.qa.plotting needs it; listed so the directory is skipped without it)
from matplotlib import pyplot as plt

# Draw without a display.
matplotlib.use("Agg")


def makeArcData(numFibers: int = 8, numLines: int = 6, seed: int = 1) -> pd.DataFrame:
    """Make a synthetic per-line residual frame for one detector, b1, for `scatterplot_with_outliers`.

    Parameters
    ----------
    numFibers : `int`, optional
        Number of fibers.
    numLines : `int`, optional
        Emission lines per fiber; each fiber also gets one trace row.
    seed : `int`, optional
        Random seed.

    Returns
    -------
    `pandas.DataFrame`
        One trace row and ``numLines`` line rows per fiber. Traces are
        RESERVED on even fibers and lines on odd indices, so both the spatial
        and the wavelength panels have RESERVED and USED data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fiberId in range(1, numFibers + 1):
        for index in range(numLines + 1):
            isTrace = index == 0
            isReserved = fiberId % 2 == 0 if isTrace else index % 2 == 1
            rows.append(
                {
                    "fiberId": fiberId,
                    "wavelength": 400.0 + 40.0 * index,
                    "x": 100.0 * fiberId,
                    "xErr": 0.01,
                    "y": 500.0 * (index + 1),
                    "yErr": 0.01,
                    "isTrace": isTrace,
                    "isLine": not isTrace,
                    "xResid": rng.normal(0.0, 0.01),
                    "yResid": rng.normal(0.0, 0.02),
                    "xResidOutlier": False,
                    "yResidOutlier": False,
                    "isUsed": not isReserved,
                    "isReserved": isReserved,
                    "status": "RESERVED" if isReserved else "USED",
                    "status_type": "RESERVED" if isReserved else "USED",
                    "description": "Trace" if isTrace else "HgI",
                    "arm": "b",
                    "spectrograph": 1,
                    "visit": 12345,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def closeFigures():
    """Close any pyplot figures a test leaves open."""
    yield
    plt.close("all")


@pytest.fixture(name="makeArcData")
def makeArcDataFixture():
    """Return `makeArcData`."""
    return makeArcData
