"""Shared fixtures for the ``pfs.drp.qa.plotting`` tests.

These tests need numpy, pandas, matplotlib, seaborn, scipy and pyyaml but not the LSST stack.
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
import yaml  # noqa: F401 (pfs.drp.qa.metrics needs it; listed so the directory is skipped without it)
from matplotlib import pyplot as plt

from pfs.drp.qa.metrics.fitStats import FitStat, FitStats
from pfs.drp.qa.plotting import DetectorGeometry

# Draw without a display.
matplotlib.use("Agg")


def makeArcData(numFibers: int = 8, numLines: int = 6, seed: int = 1) -> pd.DataFrame:
    """Make a synthetic ``dmQaResidualData`` frame for one detector, b1.

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


def makeStats(
    visits: tuple[int, ...] = (12345,),
    ccds: tuple[str, ...] = ("b1",),
    descriptions: tuple[str, ...] = ("Trace", "HgI"),
) -> pd.DataFrame:
    """Make a synthetic ``dmQaResidualStats`` frame.

    Each row is built as `pfs.drp.qa.dmResiduals.get_data_and_stats` builds
    it: ``pd.json_normalize(FitStats.to_dict())`` plus the identifying columns.

    Parameters
    ----------
    visits : `tuple` [`int`], optional
        Visits.
    ccds : `tuple` [`str`], optional
        CCD names, e.g. ``"b1"``.
    descriptions : `tuple` [`str`], optional
        Line descriptions; ``"Trace"`` carries the spatial statistics.

    Returns
    -------
    `pandas.DataFrame`
        One row per (visit, ccd, description, status_type), with ``ccd`` a
        categorical as the combined task makes it.
    """
    fitStat = FitStat(0.001, 0.02, 0.025, 0.003, 100.0, 8, 48)
    rows = []
    for visit in visits:
        for ccd in ccds:
            for description in descriptions:
                for statusType in ("RESERVED", "USED"):
                    row = pd.json_normalize(FitStats(100.0, 95.0, 105.0, fitStat, fitStat).to_dict())
                    row["status_type"] = statusType
                    row["description"] = description
                    row["arm"] = ccd[0]
                    row["spectrograph"] = int(ccd[1])
                    row["visit"] = visit
                    row["ccd"] = ccd
                    row["observationReason"] = "arc"
                    rows.append(row)
    frame = pd.concat(rows, ignore_index=True)
    frame["ccd"] = frame["ccd"].astype("category")
    return frame


def makeIqMetrics(numVisits: int = 4) -> pd.DataFrame:
    """Make a synthetic concatenated ``iqQaMetrics`` frame.

    Parameters
    ----------
    numVisits : `int`, optional
        Number of visits.

    Returns
    -------
    `pandas.DataFrame`
        One row per (visit, arm) on spectrograph 1.
    """
    rng = np.random.default_rng(2)
    rows = []
    for index in range(numVisits):
        for arm in ("b", "r"):
            rows.append(
                {
                    "visit": 140000 + index,
                    "arm": arm,
                    "spectrograph": 1,
                    "medFwhm": rng.normal(2.9, 0.1),
                    "medDxCenter": rng.normal(0.0, 0.2),
                    "dxCenterRms": abs(rng.normal(0.3, 0.05)),
                    "pctFlagged": abs(rng.normal(10.0, 3.0)),
                    "nLines": 500,
                    "traceOnly": False,
                    "obsType": "arc",
                    "seqName": "Arc: HgCd",
                    "qaStatus": "PASS",
                }
            )
    return pd.DataFrame(rows)


class SilentLog:
    """A logger stand-in that records its messages."""

    def __init__(self):
        self.messages = []

    def info(self, message, *args, **kwargs):
        self.messages.append(("info", message))

    def warning(self, message, *args, **kwargs):
        self.messages.append(("warning", message))


@pytest.fixture(autouse=True)
def closeFigures():
    """Close any pyplot figures a test leaves open."""
    yield
    plt.close("all")


@pytest.fixture(name="makeArcData")
def makeArcDataFixture():
    """Return `makeArcData`."""
    return makeArcData


@pytest.fixture(name="makeStats")
def makeStatsFixture():
    """Return `makeStats`."""
    return makeStats


@pytest.fixture(name="makeIqMetrics")
def makeIqMetricsFixture():
    """Return `makeIqMetrics`."""
    return makeIqMetrics


@pytest.fixture
def geometry() -> DetectorGeometry:
    """Return the geometry of the synthetic detector, b1."""
    return DetectorGeometry(
        width=4096,
        height=4176,
        fiberIdMin=1,
        fiberIdMax=8,
        wavelengthMin=380.0,
        wavelengthMax=700.0,
    )


@pytest.fixture
def log() -> SilentLog:
    """Return a logger stand-in that records its messages."""
    return SilentLog()
