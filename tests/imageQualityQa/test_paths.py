"""Which measurement path ``imageQualityQa`` takes, and what it then writes: numbers, no verdict."""

import logging
import re

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import measurements

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
TRACE = ("scienceTrace", "Trace")
ARC = ("scienceArc", "Arc: Neon")


def widthsRead(metrics: pd.DataFrame) -> list[float]:
    """Return the widths the gate reads from ``iqQaMetrics`` rows (`pfs.drp.qa.metrics.gate.measurements`)."""
    measured = measurements(iq=metrics)
    return measured.loc[measured["check"] == "width", "value"].tolist()


#: Columns of ``iqQaMetrics`` on the arc-line path, in order. ``runQuantum`` appends ``calexpRun``.
ARC_LINE_COLUMNS = [
    "medFwhm",
    "medDxCenter",
    "dxCenterRms",
    "pctFlagged",
    "nLines",
    "linesPerFiber",
    "nLitFibers",
    "nFibers",
    "nTraceRows",
    "nSaturatedLines",
    "pctSaturatedPixels",
    "traceOnly",
    "obsType",
    "illumination",
    "seqName",
    "isrBadPixels",
    "isrTime",
    "cosmicRayCount",
    "cosmicRayPixels",
    "cosmicRayTime",
    "reduceExposureTime",
    "fitChi2",
    "fitDof",
    "fitXRms",
    "fitYRms",
    "fitXSoften",
    "fitYSoften",
    "fitNLines",
    "fitTotalLines",
    "fitReservedChi2",
    "fitReservedXRms",
    "fitReservedYRms",
    "fitReservedXSoften",
    "fitReservedYSoften",
    "fitReservedNLines",
    "fitTraceXRms",
    "fitTraceYRms",
    "fiberIds",
    "fiberXRms",
    "fiberYRms",
    "fiberNLines",
    "pctNotVisible",
    "pctBlend",
    "pctSuspect",
    "pctRejected",
    "pctBroad",
    "pctLowSN",
    "pctMeasFail",
    "visit",
    "arm",
    "spectrograph",
]

#: What a verdict column (``qaStatus``) or a limit (``thresholdsFile``, ``fwhmWarnThreshold``) is called.
JUDGING = re.compile(r"^qa[A-Z]|[Tt]hreshold")


def judging(names) -> list[str]:
    """Return the ``names`` that are a verdict's or a limit's: none may be a column or a config field."""
    return [name for name in names if JUDGING.search(name)]


def testTaskWritesNumbersOnly(iqqa, runTask, fakes):
    """``iqQaMetrics`` holds measurements and what identifies the frame; the config holds no limit."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)
    arc = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*ARC))
    metrics = runTask(lines, detectorMap=detectorMap, calexp=arc).iqQaMetrics
    assert list(metrics.columns) == ARC_LINE_COLUMNS
    assert judging(metrics.columns) == []
    assert judging(vars(iqqa.ImageQualityQaConfig())) == []
    assert not hasattr(iqqa.ImageQualityQaTask(config=iqqa.ImageQualityQaConfig()), "thresholds")
    # With nothing measured the row is still written, and still without a verdict.
    dark = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header("scienceDark", "Dark"))
    sparse = runTask(lines, detectorMap=detectorMap, calexp=dark).iqQaMetrics
    assert len(sparse) == 1 and judging(sparse.columns) == []
    # Negative control: the check finds the columns and fields the task used to have.
    assert judging([*ARC_LINE_COLUMNS, "qaStatus", "thresholdsSha256", "fwhmWarnThreshold"]) == [
        "qaStatus",
        "thresholdsSha256",
        "fwhmWarnThreshold",
    ]


def testTraceFallsBackWhenCalexpUnusable(runTask, fakes, caplog):
    """A calexp that measures nothing hands over to fiberProfiles, marked ``traceOnly``."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)
    calexp = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*TRACE))
    profiles = fakes.FiberProfiles(detectorMap, sigma=1.2)
    with caplog.at_level(logging.INFO):
        outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp, fiberProfiles=profiles)

    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["traceOnly"]
    assert metrics["medFwhm"] == pytest.approx(FWHM_FACTOR * 1.2, rel=0.005)  # read at half maximum
    # The profile path flags nothing, so its flag rate would be zero by construction.
    assert np.isnan(metrics["pctFlagged"])
    # nLines counts the rows of ``lines``, not the 50 profile swaths.
    assert metrics["nLines"] == len(lines)
    assert len(outputs.lineData) == 50 and outputs.lineData["traceOnly"].all()

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    # 10 fibers on 4 rows: 400 rows sampled every profileYStride (100).
    assert any("calexp unusable" in message and "of 40 cross-dispersion" in message for message in warnings)
    assert any("fiber profile calibration widths" in message for message in warnings)

    # Negative control: with no fiberProfiles the same quantum is sparse.
    outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert not metrics["traceOnly"] and np.isnan(metrics["medFwhm"])


def testProfileFwhmIsNotReadAsAMeasurement(runTask, fakes):
    """The gate reads no width from a row whose FWHM is the fiberProfiles'; it reads a calexp one."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)

    unusable = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*TRACE))
    profiles = fakes.FiberProfiles(detectorMap, sigma=1.2)
    outputs = runTask(lines, detectorMap=detectorMap, calexp=unusable, fiberProfiles=profiles)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["traceOnly"] and metrics["medFwhm"] == pytest.approx(FWHM_FACTOR * 1.2, rel=0.005)
    assert widthsRead(outputs.iqQaMetrics) == []

    # Negative control: the same width measured from the calexp is read.
    usable = fakes.Exposure(fakes.traceImage(detectorMap, sigma=1.2), fakes.header(*TRACE))
    outputs = runTask(lines, detectorMap=detectorMap, calexp=usable, fiberProfiles=profiles)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "trace" and not metrics["traceOnly"]
    assert widthsRead(outputs.iqQaMetrics) == [pytest.approx(FWHM_FACTOR * 1.2, rel=0.02)]


def testQuartzAtThePfsPitchIsMeasuredFromTheCalexp(runTask, fakes):
    """At 6.17 px, where an aperture of half-width 7 holds two neighbours, the calexp is still measured.

    The aperture measurement this replaced flagged essentially every sample at
    this pitch, so every quartz fell back to the fiberProfiles calibration.
    """
    detectorMap = fakes.DetectorMap(numFibers=60, pitch=fakes.PFS_PITCH)
    lines = fakes.arcLines(detectorMap, numPerFiber=30)
    calexp = fakes.Exposure(fakes.traceImage(detectorMap, sigma=1.3), fakes.header(*TRACE))
    profiles = fakes.FiberProfiles(detectorMap, sigma=1.0)  # must not be used
    outputs = runTask(lines, detectorMap=detectorMap, calexp=calexp, fiberProfiles=profiles)

    metrics = outputs.iqQaMetrics.iloc[0]
    assert not metrics["traceOnly"]
    assert metrics["medFwhm"] == pytest.approx(FWHM_FACTOR * 1.3, rel=0.02)
    assert (~outputs.lineData["flag"]).mean() > 0.9


def testDarkMeasuresNothing(runTask, fakes):
    """A dark's metrics are null, though the same lines on an arc are measured."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)

    dark = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header("scienceDark", "Dark"))
    outputs = runTask(lines, detectorMap=detectorMap, calexp=dark)
    metrics = outputs.iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "dark"
    assert np.isnan(metrics["medFwhm"]) and np.isnan(metrics["pctFlagged"])
    assert widthsRead(outputs.iqQaMetrics) == []

    # Negative control: on an arc the same lines give a width and a flag rate.
    arc = fakes.Exposure(fakes.noiseImage(detectorMap), fakes.header(*ARC))
    metrics = runTask(lines, detectorMap=detectorMap, calexp=arc).iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "arc" and metrics["medFwhm"] == 2.5
    assert metrics["nLines"] == len(lines) and metrics["pctFlagged"] == 0.0


def testTwilightIsNotMeasured(runTask, fakes):
    """Twilight is its own class and, until sky lines are measured, has no width."""
    detectorMap = fakes.DetectorMap()
    lines = fakes.arcLines(detectorMap, numPerFiber=30)
    calexp = fakes.Exposure(
        fakes.traceImage(detectorMap, sigma=1.2), fakes.header("scienceObject", "Twilight sky")
    )
    metrics = runTask(lines, detectorMap=detectorMap, calexp=calexp).iqQaMetrics.iloc[0]
    assert metrics["obsType"] == "twilight"
    assert np.isnan(metrics["medFwhm"])
    assert metrics["nLines"] == len(lines)
    # Negative control: the same image as a quartz is measured.
    quartz = fakes.Exposure(calexp.image.array, fakes.header(*TRACE))
    metrics = runTask(lines, detectorMap=detectorMap, calexp=quartz).iqQaMetrics.iloc[0]
    assert metrics["medFwhm"] == pytest.approx(FWHM_FACTOR * 1.2, rel=0.02)


@pytest.mark.parametrize(
    ("seqType", "seqName", "obsType"),
    [
        ("scienceObject", "Twilight sky", "twilight"),
        ("scienceObject_windowed", "twilight", "twilight"),
        ("scienceObject", "sky flat", "allsky"),
        ("scienceObject", "Field 3", "science"),
        ("scienceDark", "Twilight sky", "dark"),
        ("scienceTrace", "Trace", "trace"),
        ("scienceArc", "Arc: HgCd", "arc"),
        ("biases", "Bias", "unknown"),
    ],
)
def testClassifyVisit(iqqa, fakes, seqType, seqName, obsType):
    """``W_SEQTYP`` sets the class; ``W_SEQNAM`` splits scienceObject."""
    task = iqqa.ImageQualityQaTask(config=iqqa.ImageQualityQaConfig())
    calexp = fakes.Exposure(np.zeros((2, 2)), fakes.header(seqType, seqName))
    assert task._classifyVisit(calexp, None)[0] == obsType


def testRunQuantumRecordsTheCalexpRun(iqqa, monkeypatch):
    """IqQaMetrics names the RUN of the calexp measured, whose pfsArm records the pipeline."""
    from types import SimpleNamespace

    task = iqqa.ImageQualityQaTask(config=iqqa.ImageQualityQaConfig())
    outputs = SimpleNamespace(lineData=pd.DataFrame(), iqQaSpeciesMetrics=pd.DataFrame())
    monkeypatch.setattr(
        task,
        "run",
        lambda **inputs: SimpleNamespace(**vars(outputs), iqQaMetrics=pd.DataFrame({"medFwhm": [2.0]})),
    )
    put = {}
    butlerQC = SimpleNamespace(get=lambda refs: {}, put=lambda value, ref: put.__setitem__(ref, value))
    dataId = SimpleNamespace(mapping={"visit": 1, "arm": "r", "spectrograph": 2})
    outputRefs = SimpleNamespace(iqQaMetrics="metrics", iqQaSpeciesMetrics="species")
    withCalexp = SimpleNamespace(
        arcLines=SimpleNamespace(dataId=dataId), calexp=SimpleNamespace(run="PFS/r/1")
    )
    task.runQuantum(butlerQC, withCalexp, outputRefs)
    assert put["metrics"]["calexpRun"].tolist() == ["PFS/r/1"]
    # The per-line data is not stored by default.
    assert set(put) == {"metrics", "species"}
    task.runQuantum(
        butlerQC, SimpleNamespace(arcLines=SimpleNamespace(dataId=dataId), calexp=None), outputRefs
    )
    assert put["metrics"]["calexpRun"].isna().all()


def testLineDataIsWrittenOnlyWhenAskedFor(iqqa, monkeypatch):
    """``iqQaData`` is an output, and is written, only with ``doWriteLineData``.

    The connection itself stays: stored configs set ``config.connections.iqQaData``.
    """
    from types import SimpleNamespace

    assert hasattr(iqqa.ImageQualityQaConnections, "iqQaData")
    config = iqqa.ImageQualityQaConfig()
    assert config.doWriteLineData is False
    assert "iqQaData" not in iqqa.ImageQualityQaConnections(config=config).outputs

    config.doWriteLineData = True
    assert "iqQaData" in iqqa.ImageQualityQaConnections(config=config).outputs
    task = iqqa.ImageQualityQaTask(config=config)
    lineData = pd.DataFrame({"fwhm": [2.5]})
    monkeypatch.setattr(
        task,
        "run",
        lambda **inputs: SimpleNamespace(
            lineData=lineData, iqQaMetrics=pd.DataFrame({"medFwhm": [2.0]}), iqQaSpeciesMetrics=pd.DataFrame()
        ),
    )
    put = {}
    butlerQC = SimpleNamespace(get=lambda refs: {}, put=lambda value, ref: put.__setitem__(ref, value))
    dataId = SimpleNamespace(mapping={"visit": 1, "arm": "r", "spectrograph": 2})
    outputRefs = SimpleNamespace(iqQaData="data", iqQaMetrics="metrics", iqQaSpeciesMetrics="species")
    task.runQuantum(
        butlerQC, SimpleNamespace(arcLines=SimpleNamespace(dataId=dataId), calexp=None), outputRefs
    )
    assert set(put) == {"data", "metrics", "species"} and put["data"] is lineData
