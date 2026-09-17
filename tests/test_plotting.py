"""Smoke tests for the plotting library.

Each function is called with a small synthetic frame and must return a
`matplotlib.figure.Figure`. That is a low bar on purpose: the class of breakage
it catches is the one the extraction in section 1.5 was meant to make
impossible -- plotting code left behind with a dangling import, or a signature
that no longer matches what the task passes. Nothing here needed the LSST
stack, which is the whole point of the move.
"""

import ast
import dataclasses
from pathlib import Path
from typing import ClassVar

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

import pfs.drp.qa.plotting
from pfs.drp.qa.metrics.fitStats import FitStat
from pfs.drp.qa.plotting import (
    DetectorGeometry,
    plot_detector_summary,
    plot_detector_summary_per_desc,
    plot_detectormap_residuals,
    plot_residual,
    plot_title,
    plot_visits,
    reportFigures,
    scatterplot_with_outliers,
)
from pfs.drp.qa.plotting.iqQa import plotIqTimeSeries

#: Fields of ``FitStat``, in the order ``FitStats.from_dataframe`` unpacks them.
_FIT_STAT_FIELDS = ("median", "robustRms", "weightedRms", "softenFit", "dof", "num_fibers", "num_lines")


def makeArcData(numFibers=8, numLines=6, seed=1):
    """Build a synthetic ``dmQaResidualData`` frame.

    Parameters
    ----------
    numFibers : `int`, optional
        Number of fibers.
    numLines : `int`, optional
        Number of emission lines per fiber; each fiber also gets one trace row.
    seed : `int`, optional
        Random seed.

    Returns
    -------
    `pandas.DataFrame`
        One row per (fiber, line), plus one trace row per fiber.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fiberId in range(1, numFibers + 1):
        for index in range(numLines + 1):
            isTrace = index == 0
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
                    # Traces alternate by fiber, lines by index, so both the
                    # spatial and the wavelength panel have RESERVED data.
                    "isUsed": not (fiberId % 2 == 0 if isTrace else index % 2 == 1),
                    "isReserved": fiberId % 2 == 0 if isTrace else index % 2 == 1,
                    "status": "RESERVED" if (fiberId % 2 == 0 if isTrace else index % 2) else "USED",
                    "description": "Trace" if isTrace else "HgI",
                    "status_type": "RESERVED" if (fiberId % 2 == 0 if isTrace else index % 2) else "USED",
                    "arm": "b",
                    "spectrograph": 1,
                    "visit": 12345,
                    "ccd": "b1",
                }
            )
    return pd.DataFrame(rows)


def makeStats(visits=(12345,), ccds=("b1",), descriptions=("Trace", "HgI")):
    """Build a synthetic ``dmQaResidualStats`` frame.

    The column layout matches ``pd.json_normalize(FitStats.to_dict())``, which
    is what the task stores and what ``FitStats.from_dataframe`` unpacks
    positionally.

    Parameters
    ----------
    visits : `tuple` [`int`], optional
        Visits to generate rows for.
    ccds : `tuple` [`str`], optional
        CCD names, e.g. ``"b1"``.
    descriptions : `tuple` [`str`], optional
        Line descriptions; ``"Trace"`` carries the spatial statistics.

    Returns
    -------
    `pandas.DataFrame`
        One row per (visit, ccd, description, status_type).
    """
    rows = []
    for visit in visits:
        for ccd in ccds:
            for description in descriptions:
                for statusType in ("RESERVED", "USED"):
                    row = {"dof": 100.0, "chi2X": 95.0, "chi2Y": 105.0}
                    for block in ("spatial", "wavelength"):
                        values = (0.001, 0.02, 0.025, 0.003, 100.0, 8.0, 48.0)
                        row.update(
                            {
                                f"{block}.{name}": value
                                for name, value in zip(_FIT_STAT_FIELDS, values, strict=True)
                            }
                        )
                    row.update(
                        {
                            "status_type": statusType,
                            "description": description,
                            "arm": ccd[0],
                            "spectrograph": int(ccd[1]),
                            "visit": visit,
                            "ccd": ccd,
                            "observationReason": "science",
                        }
                    )
                    rows.append(row)
    frame = pd.DataFrame(rows)
    frame["ccd"] = frame["ccd"].astype("category")
    return frame


def makeIqMetrics(numVisits=4):
    """Build a synthetic concatenated ``iqQaMetrics`` frame.

    Parameters
    ----------
    numVisits : `int`, optional
        Number of visits.

    Returns
    -------
    `pandas.DataFrame`
        One row per (visit, arm, spectrograph).
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


#: Modules the plotting library must never reach for. ``lsst.*`` is the stack;
#: the task modules are where the Butler connections live.
_FORBIDDEN_PREFIXES = (
    "lsst",
    "pfs.drp.stella",
    "pfs.drp.qa.dmResiduals",
    "pfs.drp.qa.dmCombinedResiduals",
    "pfs.drp.qa.imageQualityQa",
    "pfs.drp.qa.extractionQa",
    "pfs.drp.qa.storageClasses",
)


def _importedNames(path):
    """Return every module name a source file imports, at any scope.

    Parameters
    ----------
    path : `pathlib.Path`
        Python source file.

    Returns
    -------
    `set` [`str`]
        Absolute module names; relative imports are skipped.
    """
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


class TestNoStackNoButler:
    """The constraint the whole extraction rests on (section 1.5)."""

    def testPlottingImportsNoStackAndNoTask(self):
        """Checked statically, so the result does not depend on what is installed."""
        package = Path(pfs.drp.qa.plotting.__file__).parent
        offenders = {}
        for source in sorted(package.glob("*.py")):
            bad = sorted(
                name
                for name in _importedNames(source)
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_PREFIXES)
            )
            if bad:
                offenders[source.name] = bad
        assert offenders == {}, f"plotting reached for the stack or a task: {offenders}"

    def testFitStatsIsAvailableWithoutTheTaskModule(self):
        """The stats block a residual plot renders must not drag in a PipelineTask."""
        assert tuple(field.name for field in dataclasses.fields(FitStat)) == _FIT_STAT_FIELDS


class TestDmResidualPlots:
    def testPlotResidualReturnsAFigure(self):
        figure = plot_residual(makeArcData(), makeStats(), column="xResid", dataRange=0.1)
        assert isinstance(figure, Figure)

    def testPlotResidualHandlesTheWavelengthColumn(self):
        figure = plot_residual(makeArcData(), makeStats(), column="yResid", dataRange=0.1)
        assert isinstance(figure, Figure)

    def testPlotResidualRaisesWhenThereIsNoReservedData(self):
        data = makeArcData()
        data["isReserved"] = False
        with pytest.raises(ValueError, match="No data"):
            plot_residual(data, makeStats(), column="xResid", dataRange=0.1)

    def testDetectorMapResidualsFromGeometry(self):
        """A `DetectorGeometry` stands in for the stack's DetectorMap."""
        geometry = DetectorGeometry(
            width=4096,
            height=4176,
            fiberIdMin=1,
            fiberIdMax=8,
            wavelengthMin=380.0,
            wavelengthMax=700.0,
        )
        figure = plot_detectormap_residuals(makeArcData(), makeStats(), geometry)
        assert isinstance(figure, Figure)

    def testGeometryIsReadOffADuckTypedDetectorMap(self):
        """Existing callers pass a DetectorMap; it must still work."""

        class FakeBBox:
            width = 2048
            height = 4176

        class FakeDetectorMap:
            fiberId = np.arange(1, 9)
            metadata: ClassVar[dict] = {"WAV-MIN": 380.0, "WAV-MAX": 700.0}

            @staticmethod
            def getBBox():
                return FakeBBox()

        geometry = DetectorGeometry.coerce(FakeDetectorMap())
        assert geometry.width == 2048
        assert geometry.fiberIdMax == 8
        assert geometry.wavelengthMin == 380.0
        assert isinstance(plot_detectormap_residuals(makeArcData(), makeStats(), FakeDetectorMap()), Figure)


class TestDmCombinedPlots:
    def testPlotTitle(self):
        assert isinstance(plot_title("u/someone/run12"), Figure)

    def testPlotDetectorSummary(self):
        assert isinstance(plot_detector_summary(makeStats().query("status_type == 'RESERVED'")), Figure)

    def testPlotDetectorSummaryPerDescription(self):
        stats = makeStats().query("status_type == 'RESERVED'")
        assert isinstance(plot_detector_summary_per_desc(stats), Figure)

    def testPlotVisits(self):
        stats = makeStats(visits=(12345, 12346, 12347)).query("status_type == 'RESERVED'")
        assert isinstance(plot_visits(stats), Figure)

    def testReportFiguresYieldsEveryPage(self):
        stats = makeStats(visits=(12345, 12346))
        data = pd.concat([makeArcData(), makeArcData(seed=2)], ignore_index=True)
        detectorMaps = {
            "b1": DetectorGeometry(
                width=4096,
                height=4176,
                fiberIdMin=1,
                fiberIdMax=8,
                wavelengthMin=380.0,
                wavelengthMax=700.0,
            )
        }
        figures = list(reportFigures(stats, data, detectorMaps, "u/someone/run12", _SilentLog()))
        # Title, two summaries, then a residual and a per-visit page for b1.
        assert len(figures) == 5
        assert all(isinstance(figure, Figure) for figure in figures)

    def testReportFiguresSkipsADetectorWithNoMap(self):
        """A missing detector map is logged and skipped, not raised."""
        stats = makeStats()
        figures = list(reportFigures(stats, makeArcData(), {}, "u/someone/run12", _SilentLog()))
        assert len(figures) == 3, "title and the two summaries survive"


class TestIqQaPlots:
    def testTimeSeriesReturnsAFigure(self):
        assert isinstance(plotIqTimeSeries(makeIqMetrics()), Figure)

    def testLegacyModulePathStillWorks(self):
        """``bin.src/plotIqQaTimeSeries.py`` and the notebooks import this path."""
        from pfs.drp.qa.iqQaPlots import plotIqTimeSeries as legacy

        assert legacy is plotIqTimeSeries


class TestPalettes:
    def testScatterplotWithOutliers(self):
        figure = Figure()
        axes = figure.add_subplot(111)
        data = makeArcData()
        data["isOutlier"] = False
        result = scatterplot_with_outliers(data, "fiberId", "xResid", hue="status", ax=axes)
        assert result is axes

    def testLegacyPaletteModulePathStillWorks(self):
        from pfs.drp.qa.plotting.palettes import detector_palette as current
        from pfs.drp.qa.utils.plotting import detector_palette as legacy

        assert legacy is current


class _SilentLog:
    """A logger stand-in, so a smoke test does not depend on logging setup."""

    def info(self, *args, **kwargs):
        """Swallow an info message."""

    def warning(self, *args, **kwargs):
        """Swallow a warning message."""
