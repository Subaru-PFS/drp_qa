"""The slit-offset trajectory heatmap draws from a stored matrix (no Butler, no stack)."""

import pandas as pd
from matplotlib.figure import Figure

from pfs.drp.qa.plotting.slitOffsets import plotTrajectory


def testPlotTrajectory():
    matrix = pd.DataFrame(
        {
            "spectrograph": [2, 2, 3],
            "fiberId": [940, 802, 1358],
            "run21": [4, 0, 4],
            "run31": [4, 3, 0],
        }
    )
    fig = plotTrajectory(matrix)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_yticks().tolist() == [0, 1, 2]  # one row per fiber


def testPlotTrajectoryEmpty():
    fig = plotTrajectory(pd.DataFrame(columns=["spectrograph", "fiberId"]))
    assert isinstance(fig, Figure)
