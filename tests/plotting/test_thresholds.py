"""Tests for the threshold plots."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection

from pfs.drp.qa.metrics.calibration import calibrate, labelRows
from pfs.drp.qa.metrics.validationVisits import ValidationVisit, ValidationVisitSet
from pfs.drp.qa.plotting import plotThresholds, plotTolerances


def arcRow(visit: int, arm: str, fwhm: float) -> dict:
    return {
        "visit": visit,
        "arm": arm,
        "spectrograph": 1,
        "obsType": "arc",
        "seqName": "Arc: Neon",
        "medFwhm": fwhm,
    }


@pytest.fixture
def data():
    """Arcs on two arms, a known-bad arc, an unconfirmed one, and an m-arm row with no good data."""
    rng = np.random.default_rng(4)
    rows = [arcRow(visit, arm, rng.normal(2.6, 0.03)) for visit in range(100, 130) for arm in ("b", "r")]
    rows += [arcRow(200, "b", 3.0), arcRow(201, "b", 2.7), arcRow(202, "m", 2.6)]
    metrics = pd.DataFrame(rows)
    visitSet = ValidationVisitSet(
        knownGood=(ValidationVisit(visits=tuple(range(100, 130)), expect="PASS"),),
        knownBad=(
            ValidationVisit(visits=(200, 202), expect="FAIL", metric="medFwhm"),
            ValidationVisit(visits=(201,), expect="FAIL", unconfirmed=True),
        ),
    )
    return labelRows(metrics, visitSet, "medFwhm"), calibrate(metrics, visitSet, ["medFwhm"])


def panel(fig, group: str):
    return next(ax for ax in fig.axes if ax.get_title(loc="left").startswith(group))


def testOnePanelPerPopulation(data):
    labelled, table = data
    assert set(table["method"]) == {"percentile", "tolerance"}
    fig = plotThresholds(labelled, table, "medFwhm", ncols=2)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 3, "percentile populations only"
    assert len(fig.axes) == 4, "the unused fourth slot is hidden"


def testTolerancesShareOneAxisPerArm(data):
    """One panel per arm and obsType with good data; values as percent from their population's median."""
    labelled, table = data
    fig = plotTolerances(labelled, table, "medFwhm", ncols=1, limit=10.0)
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title(loc="left") for ax in panels] == ["b arc", "r arc"], "m has no good data"
    b = panels[0]
    assert [label.get_text() for label in b.get_xticklabels()] == ["1 Neon"]
    row = table[(table["method"] == "tolerance") & (table["group"] == "b/1/arc/Neon")].iloc[0]
    bars = [c for c in b.collections if isinstance(c, LineCollection)]
    heights = sorted(float(segment[0][1]) for bar in bars for segment in bar.get_segments())
    expected = [100 * (row[level] / row["median"] - 1) for level in ("warn", "fail")]
    assert heights == pytest.approx(expected, abs=1e-6)
    points = np.concatenate([c.get_offsets() for c in b.collections if isinstance(c, PathCollection)])
    assert points[:, 1].max() == pytest.approx(10.0), "the known-bad 3.0 px (+15 %) is drawn at the edge"
    assert np.abs(points[:, 1]).max() <= 10.0


def testTolerancesNeedToleranceRows(data):
    labelled, table = data
    with pytest.raises(ValueError, match="No tolerance suggestions"):
        plotTolerances(labelled, table[table["method"] == "percentile"], "medFwhm")


def testPanelShowsThresholdsAndRug(data):
    labelled, table = data
    ax = panel(plotThresholds(labelled, table, "medFwhm"), "b/arc")
    b = table[table["group"] == "b/arc"].iloc[0]
    verticals = {round(line.get_xdata()[0], 6) for line in ax.lines if len(set(line.get_xdata())) == 1}
    assert {round(b["warn"], 6), round(b["fail"], 6)} <= verticals
    rugX = sorted(float(x) for collection in ax.collections for x, _ in collection.get_offsets())
    assert rugX == [2.7, 3.0], "the unconfirmed and known-bad values"
    assert "UNBOUNDED" in ax.get_title(loc="left"), "30 visits cannot bound p99"


def testPopulationWithoutGoodDataSaysSo(data):
    labelled, table = data
    assert "no known-good data" in panel(plotThresholds(labelled, table, "medFwhm"), "m/arc").get_title(
        loc="left"
    )


def testUnknownMetricRaises(data):
    labelled, table = data
    with pytest.raises(ValueError, match="No suggestions"):
        plotThresholds(labelled, table, "pctFlagged")


def testSpeciesGroupedMetricPlots(data):
    """Flag rates are grouped by species; labelRows must carry the column the plot selects on."""
    labelled, _ = data
    metrics = labelled.drop(columns=["validation", "species"]).assign(pctFlagged=5.0)
    visitSet = ValidationVisitSet(knownGood=(ValidationVisit(visits=tuple(range(100, 130)), expect="PASS"),))
    table = calibrate(metrics, visitSet, ["pctFlagged"])
    assert "arc/b/Neon" in set(table["group"])
    fig = plotThresholds(labelRows(metrics, visitSet, "pctFlagged"), table, "pctFlagged")
    assert panel(fig, "arc/b/Neon").lines, "the panel has its CDF"
