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


def testNightlySeriesSharedScaleAndZero():
    """A signed drift gets one y scale across arms and a line at zero; the default has neither."""
    rng = np.random.RandomState(5)
    metrics = _metrics(rng)
    metrics["medFwhm"] = np.where(metrics["arm"] == "n", metrics["medFwhm"] + 5, metrics["medFwhm"])
    fig = plotNightlySeries(metrics, "medFwhm", sharey=True, zeroLine=True)
    limits = {ax.get_ylim() for ax in fig.axes}
    assert len(limits) == 1 and limits.pop()[0] < 0  # one scale, and it reaches zero for the line
    assert all(
        any(line.get_ydata()[0] == 0 and len(line.get_xdata()) == 2 for line in ax.lines) for ax in fig.axes
    )
    plt.close(fig)
    # Negative control: without the options the n panel, 5 px up, keeps its own scale.
    fig = plotNightlySeries(metrics, "medFwhm")
    assert len({ax.get_ylim() for ax in fig.axes}) > 1
    plt.close(fig)


def testVerdictGrid():
    fig = plotVerdictGrid(_metrics(np.random.RandomState(4)))
    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["b1", "b2", "r1", "r2", "n1", "n2"]
    letters = [text.get_text() for text in ax.texts]
    assert letters.count("F") == 1 and letters.count("P") == 23
    plt.close(fig)


def testVerdictTimeline():
    from pfs.drp.qa.plotting.comparison import plotVerdictTimeline

    verdicts = _metrics(np.random.RandomState(4))
    nights = sorted(verdicts["night"].unique())
    verdicts["visit"] = [1000 + 10 * nights.index(n) + i % 6 for i, n in enumerate(verdicts["night"])]
    fig = plotVerdictTimeline(verdicts)
    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["b1", "b2", "r1", "r2", "n1", "n2"]
    failing = verdicts[verdicts["status"] == "FAIL"]
    labels = [text.get_text() for text in ax.texts]
    assert all(str(int(v)) in labels for v in failing["visit"])  # the failing visits are named
    assert ax.get_xlim()[1] - ax.get_xlim()[0] == len(verdicts["visit"].unique())
    plt.close(fig)


def testDriftTimeline():
    from pfs.drp.qa.plotting.comparison import plotDriftTimeline

    rng = np.random.RandomState(6)
    rows = []
    for i, visit in enumerate(range(100, 112)):
        for arm in "br":
            for spectrograph in (1, 2):
                rows.append(
                    {
                        "visit": visit,
                        "arm": arm,
                        "spectrograph": spectrograph,
                        "night": datetime.date(2026, 9, 1 + i // 6),
                        "medianDeltaX": 0.3 if visit < 106 else rng.normal(0, 0.01),
                        "medianDeltaY": np.nan if arm == "r" else rng.normal(0, 0.01),
                        "calibRun": "PFS/calib/old/certify" if visit < 106 else "PFS/calib/new/certify",
                        "cameraSet": "bm" if visit == 111 else "brn",
                    }
                )
    fig = plotDriftTimeline(pd.DataFrame(rows))
    shown = [ax for ax in fig.axes if ax.get_visible()]
    # b read with m is hollow; r on the same visit, and b read with r, are filled.
    hollow = [line for line in shown[0].lines if line.get_markerfacecolor() == "none"]
    assert len(hollow) == 2 and all(len(line.get_xdata()) == 1 for line in hollow)  # SM1 and SM2, one visit
    assert not [line for line in shown[1].lines if line.get_markerfacecolor() == "none"]
    assert len(shown) == 4  # b and r, each x above y, in one grid row
    assert [ax.get_ylabel() for ax in shown] == [
        "b arm, x: traces (px)",
        "r arm, x: traces (px)",
        "b arm, y: arc lines (px)",
        "r arm, y: arc lines (px)",
    ]
    assert shown[0].get_ylim() == shown[1].get_ylim() and shown[2].get_ylim() == shown[3].get_ylim()
    assert shown[0].get_ylim() != shown[2].get_ylim()  # x and y scales are each their own
    # The change of calibration is named once per top panel, and drawn as a dashed line in every panel.
    assert sum(text.get_text() == "new" for text in shown[0].texts) == 1
    assert all(
        any(line.get_linestyle() == "--" and len(line.get_xdata()) == 2 for line in ax.lines) for ax in shown
    )
    assert len(shown[3].lines) < len(shown[2].lines)  # the r arm has no y: no data lines there
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


def testSetAgainstReference():
    from pfs.drp.qa.plotting.comparison import plotSetAgainstReference

    rng = np.random.RandomState(7)
    rows = []
    for which, offset in (("current", 0.0), ("reference", 0.3)):
        for arm in "br":
            for spectrograph in (1, 2, 3, 4):
                for _ in range(6):
                    rows.append(
                        {
                            "which": which,
                            "arm": arm,
                            "spectrograph": spectrograph,
                            "medFwhm": 2.8 + offset + rng.normal(0, 0.02),
                        }
                    )
    frame = pd.DataFrame(rows)
    fig = plotSetAgainstReference(
        frame[frame["which"] == "current"],
        frame[frame["which"] == "reference"],
        "medFwhm",
        labels=("run31", "run25"),
        thresholds={"b": (3.0, 3.2)},
    )
    assert [ax.get_title() for ax in fig.axes] == ["b arm", "r arm"]
    hollow = [
        line
        for line in fig.axes[0].lines
        if line.get_markerfacecolor() == "none" and line.get_marker() != "None"
    ]
    assert len(hollow) == 4  # one hollow median per spectrograph for the reference
    filled = [
        line
        for line in fig.axes[0].lines
        if line.get_marker() == "s" and line.get_markerfacecolor() != "none"
    ]
    assert min(h.get_xdata()[0] for h in hollow) < min(
        f.get_xdata()[0] for f in filled
    )  # the reference sits left
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()  # one scale across arms
    dashed = [line for line in fig.axes[0].lines if line.get_linestyle() == "--"]
    assert len(dashed) == 2 and not [line for line in fig.axes[1].lines if line.get_linestyle() == "--"]
    plt.close(fig)
    # Negative control: without a reference nothing is hollow.
    fig = plotSetAgainstReference(frame[frame["which"] == "current"], None, "medFwhm")
    assert not [
        line
        for line in fig.axes[0].lines
        if line.get_markerfacecolor() == "none" and line.get_marker() != "None"
    ]
    plt.close(fig)


def testDriftTimelineReferenceBand():
    from pfs.drp.qa.plotting.comparison import plotDriftTimeline

    rng = np.random.RandomState(8)
    drift = pd.DataFrame(
        {
            "visit": [100 + i for i in range(8)],
            "arm": ["b"] * 8,
            "spectrograph": [1, 2] * 4,
            "night": [datetime.date(2026, 9, 1)] * 8,
            "medianDeltaX": rng.normal(0, 0.01, 8),
            "medianDeltaY": rng.normal(0, 0.01, 8),
        }
    )
    reference = pd.DataFrame(
        {"arm": ["b"] * 20, "medianDeltaX": rng.normal(0, 0.05, 20), "medianDeltaY": rng.normal(0, 0.05, 20)}
    )
    fig = plotDriftTimeline(drift, reference=reference, referenceLabel="run25")
    shown = [ax for ax in fig.axes if ax.get_visible()]
    assert all(ax.patches for ax in shown)  # the band is drawn in every panel
    assert any("run25 5-95%" in text.get_text() for text in fig.legends[0].get_texts())
    plt.close(fig)
    fig = plotDriftTimeline(drift)  # Negative control: no reference, no band
    assert not any(ax.patches for ax in fig.axes if ax.get_visible())
    plt.close(fig)
