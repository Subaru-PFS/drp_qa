"""Smoke tests for `pfs.drp.qa.plotting.dmCombined`."""

import pandas as pd
import pytest
from matplotlib.figure import Figure

import pfs.drp.qa.plotting.dmCombined as dmCombined
from pfs.drp.qa.plotting import (
    plot_detector_summary,
    plot_detector_summary_per_desc,
    plot_title,
    plot_visits,
    reportFigures,
)


def testPlotTitle():
    figure = plot_title("u/someone/run12")
    assert isinstance(figure, Figure)
    assert [text.get_text() for text in figure.axes[0].texts] == [
        "DetectorMap Residuals Summary",
        "u/someone/run12",
    ]


def testPlotDetectorSummary(makeStats):
    stats = makeStats(ccds=("b1", "r1")).query("status_type == 'RESERVED'")
    assert isinstance(plot_detector_summary(stats), Figure)


def testPlotDetectorSummaryPerDescription(makeStats):
    stats = makeStats(ccds=("b1", "r1")).query("status_type == 'RESERVED'")
    assert isinstance(plot_detector_summary_per_desc(stats), Figure)


def testPlotVisits(makeStats):
    stats = makeStats(visits=(12345, 12346, 12347)).query("status_type == 'RESERVED'")
    assert isinstance(plot_visits(stats), Figure)


@pytest.fixture
def report(makeArcData, makeStats):
    """Return the stats and data of two visits on b1."""
    stats = makeStats(visits=(12345, 12346))
    data = pd.concat([makeArcData(), makeArcData(seed=2).assign(visit=12346)], ignore_index=True)
    return stats, data


def testReportFigures(report, geometry, log):
    stats, data = report
    figures = list(reportFigures(stats, data, {"b1": geometry}, "u/someone/run12", log))

    # Title, two detector summaries, then the residuals and per-visit pages of b1.
    assert len(figures) == 5
    assert all(isinstance(figure, Figure) for figure in figures)
    assert figures[3].get_suptitle() == "DetectorMap Residuals - Median of all visits - b1"
    assert figures[4].get_suptitle().endswith(" - b1")
    assert not [message for level, message in log.messages if level == "warning"]


def testReportFiguresSkipsADetectorWithoutAMap(report, log):
    stats, data = report
    figures = list(reportFigures(stats, data, {}, "u/someone/run12", log))

    assert len(figures) == 3
    assert ("warning", "DetectorMap not found for b1. Skipping.") in log.messages


def testReportFiguresKeepsTheResidualPageIfTheNextFails(report, geometry, log, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("cannot draw")

    monkeypatch.setattr(dmCombined, "plot_visits", fail)
    stats, data = report
    figures = list(reportFigures(stats, data, {"b1": geometry}, "u/someone/run12", log))

    assert len(figures) == 4
    assert figures[3].get_suptitle().startswith("DetectorMap Residuals")
    assert ("warning", "Error plotting for b1: cannot draw") in log.messages
