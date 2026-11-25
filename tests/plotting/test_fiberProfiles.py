"""The wing-excess figure draws from stored per-fiber rows (no Butler, no stack)."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pfs.drp.qa.plotting.fiberProfiles import plotWingExcess


def rows(arm, excess=None, replaced=()):
    """Return per-fiber rows of one arm: ``excess`` by fiber, the rest near zero."""
    fiberId = np.arange(2000, 2040)
    return pd.DataFrame(
        {
            "arm": arm,
            "fiberId": fiberId,
            "wingExcess": [(excess or {}).get(int(ff), 0.0005) for ff in fiberId],
            "replaced": [int(ff) in replaced for ff in fiberId],
        }
    )


def offsets(ax, label):
    """Return the fiber numbers of the points drawn under a legend label."""
    for collection in ax.collections:
        if collection.get_label() == label:
            return sorted(int(x) for x, _ in collection.get_offsets())
    return []


def testOnePanelPerArmWithMeasuredAndCopiesApart():
    fibers = pd.concat([rows("r", {2009: 0.19, 2008: 0.06}, replaced=(2008,)), rows("n", {2009: 0.21})])
    fig = plotWingExcess(fibers)
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    red, infrared = fig.axes
    assert red.get_title(loc="left") == "red: 1 measured and 1 copies over 0.01"
    assert offsets(red, "over the limit, measured") == [2009] and offsets(red, "over the limit, copy") == [
        2008
    ]
    assert offsets(infrared, "over the limit, measured") == [2009]
    # Negative control: a fiber under the limit is drawn as neither.
    assert 2010 not in offsets(red, "over the limit, measured") + offsets(red, "over the limit, copy")


def testTheCalibrationInForceIsDrawnAsRings():
    fibers, previous = rows("r", {2009: 0.19}), rows("r", {2020: 0.03})
    ax = plotWingExcess(fibers, previous, previousLabel="asis").axes[0]
    assert offsets(ax, "over the limit in asis") == [2020]
    assert offsets(plotWingExcess(fibers).axes[0], "over the limit in asis") == []


def testNoProfilesDrawsAnEmptyFigure():
    fig = plotWingExcess(pd.DataFrame(columns=["arm", "fiberId", "wingExcess", "replaced"]))
    assert isinstance(fig, Figure) and len(fig.axes) == 1
