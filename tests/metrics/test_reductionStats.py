"""The detector-health statistics read from synthetic reduction metadata.

No reduction runs here: the ISR and reduceExposure task metadata and the ISR header
are built as dictionaries of the shape drp_stella and obs_pfs record, so the reader is
exercised without the stack. The real keys are verified on a reduced night separately.
"""

import math

import pandas as pd

from pfs.drp.qa.metrics.reductionStats import (
    FIT_METRICS,
    ISR_COUNT_METRICS,
    REDUCTION_COLUMNS,
    reductionStatsFrame,
    speciesFitStats,
)

DATA_ID = {"visit": 155486, "arm": "r", "spectrograph": 1}
N_DATA_ID = {"visit": 155527, "arm": "n", "spectrograph": 3}


def _value(frame: pd.DataFrame, metric: str, amplifier: str = "") -> float:
    """Return the single value of one metric (and amplifier) in a frame."""
    rows = frame[(frame["metric"] == metric) & (frame["amplifier"] == amplifier)]
    assert len(rows) == 1, f"{metric!r} amplifier={amplifier!r}: {len(rows)} rows"
    return float(rows["value"].iloc[0])


def _adjustMetadata(applied: bool = True) -> dict:
    """Build reduceExposure metadata with the adjustDetectorMap fit scalars."""
    section = {name: float(index + 1) for index, name in enumerate(FIT_METRICS)}
    return {
        "detectorMapAdjustment": "applied" if applied else "failed",
        "reduceExposure:adjustDetectorMap": section,
        "reduceExposure:scatteredLight": {
            "top": 1.0,
            "bottom": 2.0,
            "modelMean": 3.0,
            "modelMax": 4.0,
        },
        "reduceExposure:pfiCorrection": {
            "correctionMean": 0.5,
            "correctionStd": 0.1,
            "skyNormsMean": float("nan"),
            "skyNormsStd": float("nan"),
            "numBad": 7,
        },
    }


def testSchema():
    """The frame has exactly the detector-health columns, in order, with the dtypes."""
    frame = reductionStatsFrame(DATA_ID, reduceExposureMetadata=_adjustMetadata())
    assert list(frame.columns) == list(REDUCTION_COLUMNS)
    assert {str(dtype) for dtype in frame.dtypes} == {"int64", "string", "float64"}
    assert (frame["visit"] == DATA_ID["visit"]).all()
    assert (frame["arm"] == DATA_ID["arm"]).all()


def testEmpty():
    """No metadata at all gives a well-formed empty frame, not an error."""
    frame = reductionStatsFrame(DATA_ID)
    assert len(frame) == 0
    assert list(frame.columns) == list(REDUCTION_COLUMNS)


def testFitStatsWhenApplied():
    """Every fit scalar is a whole-detector row when the adjustment applied."""
    frame = reductionStatsFrame(DATA_ID, reduceExposureMetadata=_adjustMetadata(applied=True))
    for name in FIT_METRICS:
        assert _value(frame, name) == float(FIT_METRICS.index(name) + 1)
    # Whole-detector, so no amplifier.
    fitRows = frame[frame["metric"].isin(FIT_METRICS)]
    assert (fitRows["amplifier"] == "").all()


def testFitStatsSkippedWhenNotApplied():
    """fit* and reserved* are dropped when the adjustment did not apply; final* remain.

    Negative control: a fit that failed can leave stale fit*/reserved* in the metadata,
    so they must not be read; the convention is to gate them on ``detectorMapAdjustment``.
    """
    frame = reductionStatsFrame(DATA_ID, reduceExposureMetadata=_adjustMetadata(applied=False))
    present = set(frame["metric"])
    assert "finalXRms" in present
    # Negative control: gated scalars are absent, not merely zero.
    assert "fitXRms" not in present
    assert "reservedChi2" not in present
    assert not any(name.startswith(("fit", "reserved")) for name in present if name in FIT_METRICS)


def testBareSubtaskKey():
    """The subtask is found by its bare name, as reduceExposure_metadata records it."""
    metadata = {
        "detectorMapAdjustment": "applied",
        "adjustDetectorMap": {"fitXRms": 0.021, "fitYRms": 0.034},
    }
    frame = reductionStatsFrame(DATA_ID, reduceExposureMetadata=metadata)
    assert _value(frame, "fitXRms") == 0.021
    assert _value(frame, "fitYRms") == 0.034


def testFlattenedSubtaskKeys():
    """A flattened metadata mapping (dotted keys) is navigated too."""
    metadata = {
        "detectorMapAdjustment": "applied",
        "adjustDetectorMap.fitXRms": 0.021,
        "adjustDetectorMap.fitYRms": 0.034,
    }
    frame = reductionStatsFrame(DATA_ID, reduceExposureMetadata=metadata)
    assert _value(frame, "fitXRms") == 0.021
    assert _value(frame, "fitYRms") == 0.034


def testScatteredAndPfiNamespaced():
    """Scattered-light and PFI scalars are namespaced so generic keys don't collide."""
    frame = reductionStatsFrame(DATA_ID, reduceExposureMetadata=_adjustMetadata())
    assert _value(frame, "scatteredLight.top") == 1.0
    assert _value(frame, "pfiCorrection.correctionMean") == 0.5
    # Negative control: the bare key is not used.
    assert "top" not in set(frame["metric"])
    # A NaN scalar still gets a row (it is a recorded value).
    assert math.isnan(_value(frame, "pfiCorrection.skyNormsMean"))


def testIsrCountsOnlyWhenPresent():
    """An n-arm count is a row only when recorded; a step that did not run leaves none.

    Negative control: a CDS arc records no CR keys, so ``nCR`` must be absent rather
    than defaulted to zero.
    """
    recorded = {"nSat": 0, "nLow": 977, "nDefects": 201178, "nLinDefects": 194254}
    frame = reductionStatsFrame(N_DATA_ID, isrMetadata=recorded)
    assert _value(frame, "nLow") == 977.0
    assert _value(frame, "nSat") == 0.0  # present and zero: a row, with value 0
    present = set(frame["metric"])
    assert "nCR" not in present  # not recorded on this frame
    assert present <= set(ISR_COUNT_METRICS)
    assert (frame["amplifier"] == "").all()  # n arm: whole-detector


def testPerAmplifierHeader():
    """Per-amplifier gain and read noise come from the ISR header keys."""
    header = {
        "LSST GAIN c1": 1.24,
        "LSST READNOISE c1": 3.91,
        "LSST GAIN c2": 1.19,
        "LSST READNOISE c2": 3.88,
        "W_SEQTYP": "object",  # unrelated header key: ignored
    }
    frame = reductionStatsFrame(DATA_ID, header=header)
    assert _value(frame, "gain", amplifier="c1") == 1.24
    assert _value(frame, "readNoise", amplifier="c2") == 3.88
    assert set(frame["amplifier"]) == {"c1", "c2"}
    # Negative control: a non-ISR header key does not become a metric.
    assert "W_SEQTYP" not in set(frame["metric"])


def testPerAmplifierHeaderCcdOnly():
    """Per-amp header keys are read on a CCD arm, not on the n arm (an H4, not CCD amps)."""
    header = {"LSST GAIN c1": 1.24, "LSST READNOISE c1": 3.91}
    ccd = reductionStatsFrame({"visit": 155486, "arm": "r", "spectrograph": 1}, header=header)
    assert _value(ccd, "gain", amplifier="c1") == 1.24
    # Negative control: the same keys on the n arm produce no rows.
    narm = reductionStatsFrame({"visit": 155486, "arm": "n", "spectrograph": 3}, header=header)
    assert len(narm) == 0


def testDarkScale():
    """Dark statistics are recorded on a CCD that has a dark."""
    frame = reductionStatsFrame(
        {"visit": 155527, "arm": "r", "spectrograph": 3},
        isrMetadata={"darkScale": 355.41, "darkScalePerSecond": 1.185},
    )
    assert _value(frame, "darkScale") == 355.41
    assert _value(frame, "darkScalePerSecond") == 1.185


def testAllSourcesCombine():
    """ISR, reduceExposure and header rows appear together, keyed to the one image."""
    frame = reductionStatsFrame(
        DATA_ID,
        isrMetadata={"nDefects": 201178},
        reduceExposureMetadata=_adjustMetadata(),
        header={"LSST READNOISE c1": 3.91},
    )
    assert _value(frame, "nDefects") == 201178.0
    assert _value(frame, "fitXRms") == float(FIT_METRICS.index("fitXRms") + 1)
    assert _value(frame, "readNoise", amplifier="c1") == 3.91
    assert (frame["visit"] == DATA_ID["visit"]).all()


def _speciesMetadata(applied: bool = True) -> dict:
    """Build reduceExposure metadata with the adjustDetectorMap per-species arrays.

    The arrays run parallel to ``speciesName``; ``fitDetectorMap`` records them for every
    fitted species, ``Trace`` among them, and leaves an undefined value NaN.
    """
    return {
        "detectorMapAdjustment": "applied" if applied else "failed",
        "reduceExposure:adjustDetectorMap": {
            "speciesName": ["HgI", "NeI", "Trace"],
            "speciesXRms": [0.021, 0.018, 0.040],
            "speciesYRms": [0.034, 0.030, float("nan")],
            "speciesXSoften": [0.011, 0.009, 0.020],
            "speciesYSoften": [0.0, 0.0, 0.0],
        },
    }


def testSpeciesFitStats():
    """Each species' (xRms, yRms, xSoften, ySoften) comes from the parallel arrays, Trace included."""
    stats = speciesFitStats(_speciesMetadata())
    assert set(stats) == {"HgI", "NeI", "Trace"}
    assert stats["HgI"] == (0.021, 0.034, 0.011, 0.0)
    assert stats["NeI"] == (0.018, 0.030, 0.009, 0.0)
    # Trace is kept here; the lamp-species mapping (longFormat.speciesStats) is what drops it.
    assert stats["Trace"][0] == 0.040 and stats["Trace"][2] == 0.020
    assert math.isnan(stats["Trace"][1])


def testSpeciesFitStatsGatedOnApplied():
    """Nothing is read when the adjustment did not apply.

    Negative control: a fit that later failed can leave stale species arrays in the metadata,
    so an unapplied adjustment must yield no stats rather than those arrays.
    """
    assert speciesFitStats(_speciesMetadata(applied=False)) == {}


def testSpeciesFitStatsAbsent():
    """No metadata, or no species recorded, gives an empty mapping, not an error."""
    assert speciesFitStats(None) == {}
    assert speciesFitStats({"detectorMapAdjustment": "applied"}) == {}


def testSpeciesFitStatsBareSubtaskKey():
    """The subtask is found by its bare name, as reduceExposure_metadata records it."""
    metadata = {
        "detectorMapAdjustment": "applied",
        "adjustDetectorMap": {
            "speciesName": ["ArI"],
            "speciesXRms": [0.025],
            "speciesYRms": [0.035],
            "speciesXSoften": [0.01],
            "speciesYSoften": [0.0],
        },
    }
    assert speciesFitStats(metadata) == {"ArI": (0.025, 0.035, 0.01, 0.0)}
