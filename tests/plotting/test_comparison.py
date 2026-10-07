"""Tests for `pfs.drp.qa.plotting.comparison`."""

import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pfs.drp.qa.plotting.comparison import plotMetricComparison, plotNightlySeries, plotVerdictGrid


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
    assert [ax.get_title() for ax in axes] == ["b arm", "r arm", "n arm"]
    labels = [text.get_text() for text in axes[0].get_legend().get_texts()]
    assert labels == ["period (8)", "reference (8)"]
    assert len(axes[0].lines) == 4  # two runs, two thresholds
    assert len(axes[2].get_legend().get_texts()) == 1  # no reference n arm
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
