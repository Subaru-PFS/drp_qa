"""Tests for `pfs.drp.qa.plotting.extraction.plotExtractionStats`.

Stack-free: a synthetic ``extQaImage_pickle`` (the per-fiber arrays the task
stores) drives the figure. ``targetMask`` is keyed by plain strings standing in
for the task's ``TargetType``; the plot treats the keys as opaque labels, so no
datamodel import is needed.
"""

import numpy as np
from matplotlib.figure import Figure

from pfs.drp.qa.plotting.extraction import ExtractionPlotRanges, plotExtractionStats

_YLABELS = {
    "Chi (median)",
    "Chi (standard deviation)",
    "Chi at profile peak pixels",
    "Residual fraction (percent)",
}

_N = 60


def _qaStats(
    seed: int = 0,
    *,
    chiMed: np.ndarray | None = None,
    targets: tuple[str, ...] = ("SCIENCE", "SKY"),
) -> dict:
    """Return a synthetic one-detector extraction ``QaDict`` for `plotExtractionStats`."""
    rng = np.random.default_rng(seed)
    fiberIds = np.sort(rng.choice(np.arange(1, 651), size=_N, replace=False))
    xa = np.interp(fiberIds, (fiberIds.min(), fiberIds.max()), (50.0, 4050.0)) + rng.normal(0, 2, _N)
    pfsArmAve = rng.uniform(100.0, 5000.0, _N)
    # Spread the fibers across the target types so every legend group has fibers.
    assignment = rng.integers(0, len(targets), _N)
    targetMask = {target: assignment == index for index, target in enumerate(targets)}
    return {
        "dataId": {"visit": 123456, "arm": "r", "spectrograph": 1, "run": "test/run"},
        "fiberIds": fiberIds,
        "xa": xa,
        "pfsArmAve": pfsArmAve,
        "targetMask": targetMask,
        "chiMed": rng.normal(0.0, 0.3, _N) if chiMed is None else np.asarray(chiMed),
        "chiStd": np.abs(rng.normal(1.0, 0.3, _N)),
        "chiAtPeak": rng.normal(0.0, 0.3, _N),
        "resFrac": rng.normal(0.0, 1.0, _N),
    }


def _panel(fig: Figure, ylabel: str):
    """Return the summary panel with the given y-axis label."""
    return next(ax for ax in fig.axes if ax.get_ylabel() == ylabel)


def _scatterY(ax) -> np.ndarray:
    """Return every scattered point's y value on ``ax``."""
    arrays = [coll.get_offsets()[:, 1] for coll in ax.collections if len(coll.get_offsets())]
    return np.concatenate(arrays) if arrays else np.array([])


def testBuildsWithFourPanels():
    fig = plotExtractionStats(_qaStats())
    assert isinstance(fig, Figure)
    assert {ax.get_ylabel() for ax in fig.axes} >= _YLABELS
    assert "Chi distribution" in fig.get_suptitle()
    assert "RUN=test/run" in fig.get_suptitle()


def testTitleFallsBackToDataIdThenOverride():
    # The title is built from qaStats["dataId"] by default, and overridden when given.
    assert "visit=123456" in plotExtractionStats(_qaStats()).get_suptitle()
    assert plotExtractionStats(_qaStats(), title="custom").get_suptitle() == "custom"
    # Negative control: no dataId and no title gives no suptitle, not a crash.
    bare = _qaStats()
    bare["dataId"] = {}
    assert plotExtractionStats(bare).get_suptitle() == ""


def testLegendCountsEachTargetType():
    fig = plotExtractionStats(_qaStats(targets=("SCIENCE", "SKY", "FLUXSTD")))
    labels = {text.get_text().split(":")[0] for text in _panel(fig, "Chi (median)").get_legend().get_texts()}
    assert {"SCIENCE", "SKY", "FLUXSTD"} <= labels


def testOutOfRangePointPinnedToEdgeNotTrueValue():
    ranges = ExtractionPlotRanges()
    ymax = ranges.chiMed[1]
    chiMed = np.zeros(_N)
    chiMed[0] = 5.0  # well above the ceiling
    fig = plotExtractionStats(_qaStats(chiMed=chiMed), ranges=ranges)
    panel = _panel(fig, "Chi (median)")
    ys = _scatterY(panel)
    # The over-range fiber is drawn as a marker pinned to the ceiling...
    assert np.any(np.isclose(ys, ymax))
    # ...and the y-scale is held at the configured range, not stretched to the outlier.
    assert np.allclose(panel.get_ylim(), ranges.chiMed)
    # Negative control: a plot that did not clip would show a point at the true y = 5.0.
    assert not np.any(np.isclose(ys, 5.0))


def testConfigRangesAreApplied():
    # A caller's ranges set each panel's y-limits (the task passes its config here).
    ranges = ExtractionPlotRanges(chiMed=(-2.0, 2.0), chiStd=(0.0, 4.0))
    fig = plotExtractionStats(_qaStats(), ranges=ranges)
    assert np.allclose(_panel(fig, "Chi (median)").get_ylim(), (-2.0, 2.0))
    assert np.allclose(_panel(fig, "Chi (standard deviation)").get_ylim(), (0.0, 4.0))
