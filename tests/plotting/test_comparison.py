"""Tests for `pfs.drp.qa.plotting.comparison`."""

import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pfs.drp.qa.plotting.comparison import (
    plotArmTimeline,
    plotMetricComparison,
    plotNightlySeries,
    plotVerdictGrid,
)


def _metrics(rng, arms="brn", nights=4, offset=0.0):
    rows = []
    for night in range(nights):
        for arm in arms:
            for spectrograph in (1, 2):
                rows.append(
                    {
                        "night": datetime.date(2026, 9, 1) + datetime.timedelta(days=night),
                        "arm": arm,
                        "spectrograph": spectrograph,
                        "medFwhm": 2.8 + offset + rng.normal(0, 0.05),
                        "status": "FAIL" if (arm, spectrograph, night) == ("b", 1, 2) else "PASS",
                    }
                )
    return pd.DataFrame(rows)


def testMetricComparisonOnePanelPerArm():
    rng = np.random.RandomState(1)
    fig = plotMetricComparison(
        _metrics(rng), _metrics(rng, arms="br", offset=-0.1), "medFwhm", thresholds={"b": (3.2, 3.5)}
    )
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in axes] == [
        "b arm: 8 period, 8 reference",
        "r arm: 8 period, 8 reference",
        "n arm: 8 period, 0 reference",
    ]
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["period", "reference"]
    assert all(ax.get_legend() is None for ax in axes)  # nothing drawn over the curves
    assert len(axes[0].lines) == 4  # two runs, two thresholds
    plt.close(fig)


def testMetricComparisonWithoutReference():
    fig = plotMetricComparison(_metrics(np.random.RandomState(2)), None, "medFwhm")
    assert len(fig.axes) == 3
    plt.close(fig)


def testNightlySeries():
    rng = np.random.RandomState(3)
    fig = plotNightlySeries(_metrics(rng), "medFwhm", reference=_metrics(rng))
    assert [ax.get_title() for ax in fig.axes] == ["b arm", "r arm", "n arm"]
    assert len(fig.axes[0].lines) == 2  # one line per spectrograph
    assert len(fig.axes[0].get_xticks()) == 4  # one tick per night
    plt.close(fig)


def testVerdictGrid():
    fig = plotVerdictGrid(_metrics(np.random.RandomState(4)))
    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["b1", "b2", "r1", "r2", "n1", "n2"]
    letters = [text.get_text() for text in ax.texts]
    assert letters.count("F") == 1 and letters.count("P") == 23
    plt.close(fig)


def testArmTimeline():
    nights = [datetime.date(2026, 9, 1), datetime.date(2026, 9, 2)]
    timeline = pd.DataFrame(
        {
            "night": [nights[0], nights[0], nights[1]],
            "start": pd.to_datetime(["2026-09-02 05:40", "2026-09-02 05:45", "2026-09-02 18:00"]),
            "exptime": [5.0, 300.0, 30.0],
            "kind": ["arc", "dark", "quartz"],
        }
    )
    sequences = pd.DataFrame({"night": [nights[0]], "minutesSince": [1.3], "litKind": ["arc"]})
    fig = plotArmTimeline(timeline, sequences)
    axTime, axGap = fig.axes
    assert [label.get_text() for label in axTime.get_yticklabels()] == ["09-01", "09-02"]
    assert len(axTime.collections) == 3  # one bar group per kind and night
    assert axGap.get_xscale() == "log"
    assert sum(len(c.get_offsets()) for c in axGap.collections) == 1
    plt.close(fig)
