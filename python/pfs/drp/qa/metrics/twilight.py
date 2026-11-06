"""Twilight sky flats: is each exposure of a twilight sequence a good flat? Measurement and verdict.

A twilight is the sky flat behind the screen response: each fiber's flux over
its normalisation should be the same across the focal plane, but for a smooth
sky gradient. The reference (rule R1 of ``docs/qa-principles.md``) is the
sequence's other exposures, minutes apart, and nothing here depends on a
calibration (PIPE2D-1973). The check says whether an exposure is a good flat
and what differs in it; it does not say why.

`fiberFlux` reduces a detector's spectra to one value per fiber: the median of
flux / norm over the arm's window (`WINDOWS`), with its S/N. Any static
normalisation (fiberNorms, the PFI correction) cancels within a sequence.

`sequenceConsistency` compares the exposures of each sequence and arm. Each
exposure's fiber values are divided by their median over the arm's fibers (the
sky's level changes through twilight), then:

``twilightLevelOffset``
    Each detector's median ratio to the per-fiber median of all the sequence's
    exposures, over the field's median ratio, minus one (%). With at least
    `MIN_EXPOSURES` exposures, the median resists one bad exposure, so a
    detector whose level differs in one exposure is named in that exposure
    alone. Every exposure is judged: a detector's ~600 fibers are spread over
    the field, so its median is not moved by structure in a few of them. No
    row is dropped: a detector whose fibers read zero is at -100 %, and one
    with no measured fiber at all is infinite, so both fail; a fiber needs
    `MIN_EXPOSURES` measured exposures for a reference, and a detector with
    none is NaN, unjudged.
``twilightResidRms``
    The robust RMS, beyond photon noise (%), of the ratio of each fiber to the
    per-fiber median of the sequence's bright exposures, this one included,
    after each detector's median and a robust quadratic over the focal plane
    (the sky's gradient) are divided out. Including the judged exposure keeps
    one bad exposure from moving the reference: with two others, their median
    would be their mean, and a patch in one would show, mirrored at half its
    depth, in both of the others. It needs `MIN_EXPOSURES` bright exposures.
    Only exposures at or above ``minSnr`` in the brightness arm are judged or
    used in the reference: below it (``tooFaint``), exposures carry structure
    beyond their noise that brighter ones of the same sequence do not, on clear
    nights too, and are not good flats. The noise is simulated: each bright
    exposure's median per-fiber error (that of a flat spectrum), drawn for
    every exposure of the reference, through the same median. Real spectra have
    structure in the window, so a few tenths of a per cent of noise can remain
    in the excess.
``twilightPatch``
    The deepest coherent deviation on the focal plane (%): the largest
    absolute median of that residual over each fiber's `PATCH_NEIGHBOURS`
    nearest fibers. The RMS is robust, so it ignores a patch on a few per cent
    of the fibers, and the quadratic absorbs much of one spanning the field;
    this sees what lies between. Measured on the same exposures as the RMS.
``twilightArmMean``, ``twilightArmHalfDiff``
    Diagnostics, not judged: the robust RMS (%) of the mean and of the
    half-difference of each fiber's b and r residuals, telling a difference
    common to the arms from one that is not.

``satFraction`` is stored, not judged: saturated pixels are left out of the
medians, and no saturated twilight has been seen to set a limit by.

Blind spot: anything the same through the sequence cancels, whether a pattern
on the focal plane or a change of colour uniform across it. Comparing a
twilight with good twilights of other nights covers it (PIPE2D-1981), and is
not done here.

The provisional thresholds (`twilightThresholds`) are from a prototype over 16
sequences of 2025-03 to 2026-09; the n arm is not judged, its residuals sit
3-9 % above the noise model even on clear nights.

Nothing here imports the LSST stack.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pfs.drp.qa.metrics.gate import THRESHOLDS_COLUMNS, gate, judge, thresholdsPath, thresholdsRecord
from pfs.drp.qa.metrics.gate import VERDICT_COLUMNS as VERDICT_COLUMNS_GATE
from pfs.drp.qa.metrics.lineResiduals import observationType

__all__ = [
    "FIBER_COLUMNS",
    "IMAGE_VERDICT_COLUMNS",
    "METRIC_COLUMNS",
    "MIN_EXPOSURES",
    "PATCH_NEIGHBOURS",
    "TWILIGHT_METRICS",
    "TWILIGHT_THRESHOLDS_FILE",
    "VERDICT_COLUMNS",
    "WINDOWS",
    "addPositions",
    "fiberFlux",
    "isTwilightSequence",
    "sequenceConsistency",
    "sequenceReadiness",
    "twilightImageVerdict",
    "twilightThresholds",
    "twilightVerdict",
]

#: Wavelength window per arm (nm), fiberNormsQa's.
WINDOWS = {"b": (450.0, 650.0), "r": (650.0, 950.0), "m": (650.0, 950.0), "n": (950.0, 1250.0)}

#: Exposures a sequence needs in an arm for its medians to resist one bad exposure.
MIN_EXPOSURES = 3

#: Provisional patch thresholds (%): normal exposures reached 1.6 % in r and 2.0 % in b; a 6 % patch
#: 30-60 mm wide gives 2.5-3.4 %.
PATCH_WARN, PATCH_FAIL = 2.5, 4.0

#: Fibers a patch is averaged over: about 1 % of the focal plane, 45 mm across.
PATCH_NEIGHBOURS = 30

#: The judged metrics.
TWILIGHT_METRICS = ("twilightLevelOffset", "twilightResidRms", "twilightPatch")

#: Columns of `twilightVerdict`'s table, one row per (visit, arm).
VERDICT_COLUMNS = ("visit", "arm", "sequence", "status", "reason", "twilightArmMean", "twilightArmHalfDiff")

#: The thresholds file shipped with the package (``metrics/data``, made by ``makeTwilightThresholds.py`` from
#: `twilightThresholds`): what the judge judges with and records, so qadb's gate can cite it.
TWILIGHT_THRESHOLDS_FILE = "twilightThresholds.yaml"

#: Columns of `fiberFlux`'s table, the ``twilightFiberFlux`` dataset with the task's identity columns.
FIBER_COLUMNS = {
    "fiberId": "Fiber identifier.",
    "x": "Fiber position on the focal plane (PfsConfig.pfiCenter), mm.",
    "y": "Fiber position on the focal plane (PfsConfig.pfiCenter), mm.",
    "flux": "Median of flux / norm over the arm's window, good pixels only.",
    "snr": "Median of flux / sqrt(variance) over the same pixels.",
    "nGood": "Number of good pixels in the window.",
    "satFraction": "Fraction of the window's pixels with the SAT mask bit.",
}

#: Columns of `sequenceConsistency`'s table, one row per (visit, arm, spectrograph).
METRIC_COLUMNS = {
    "visit": "Visit.",
    "arm": "Arm.",
    "spectrograph": "Spectrograph.",
    "sequence": "IIC sequence id (W_SEQID).",
    "seqName": "Sequence name (W_SEQNAM).",
    "nExposures": "Exposures of the sequence in this arm.",
    "snr": "Median S/N per pixel of the exposure in this arm.",
    "brightness": "Median S/N per pixel of the exposure in the brightness arm.",
    "tooFaint": "Below the brightness floor: not a good flat; the RMS is not measured.",
    "nReference": "Bright exposures in the reference median, this one included; 0 when the RMS is not measured.",
    "noise": "Photon noise expected in the RMS (%).",
    "satFraction": "Largest saturated fraction of a fiber's window on the detector.",
    "nMeasured": "Fibers of the detector with a measured flux in this exposure.",
    "twilightLevelOffset": (
        "The detector's level against the sequence's, over the field's (%); inf with no measured fiber,"
        " NaN with no reference."
    ),
    "twilightResidRms": "Field RMS beyond photon noise against the sequence's other bright exposures (%).",
    "twilightPatch": "Deepest coherent deviation: largest |median| over a fiber's nearest neighbours (%).",
    "twilightArmMean": "RMS of the mean of each fiber's b and r residuals (%).",
    "twilightArmHalfDiff": "RMS of half the difference of each fiber's b and r residuals (%).",
}

#: Columns of `twilightImageVerdict`'s table, the ``twilightQaMetrics`` dataset: one row per detector image.
IMAGE_VERDICT_COLUMNS = (*METRIC_COLUMNS, *VERDICT_COLUMNS_GATE, *THRESHOLDS_COLUMNS)

# The expected error of a median is sqrt(pi/2) that of a mean.
_MEDIAN_ERROR = np.sqrt(np.pi / 2)

#: Fibers drawn to simulate the noise of the residual against the reference median.
_NOISE_DRAWS = 4000


def _robustRms(values: Any) -> float:
    """Robust RMS from the interquartile range, as drp_stella's ``robustRms``."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    lower, upper = np.percentile(values, [25.0, 75.0])
    return float(0.741 * (upper - lower))


def fiberFlux(
    fiberId: np.ndarray,
    wavelength: np.ndarray,
    flux: np.ndarray,
    norm: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    window: tuple[float, float],
    satBit: int = 0,
) -> pd.DataFrame:
    """Reduce a detector's spectra to one value per fiber.

    Parameters
    ----------
    fiberId : `numpy.ndarray` [`int`], shape ``(numFibers,)``
        Fiber identifiers.
    wavelength, flux, norm, variance, mask : `numpy.ndarray`, shape ``(numFibers, numPixels)``
        The arm's spectra, as ``PfsArm`` holds them.
    window : `tuple` [`float`, `float`]
        Wavelength range (nm).
    satBit : `int`, optional
        The mask value of saturated pixels (``pfsArm.flags["SAT"]``); 0 when
        the mask has none.

    Returns
    -------
    `pandas.DataFrame`
        One row per fiber: ``fiberId``, ``flux``, ``snr``, ``nGood``,
        ``satFraction`` (`FIBER_COLUMNS`). ``flux`` and ``snr`` are NaN for a
        fiber with no good pixel in the window.
    """
    inWindow = (wavelength >= window[0]) & (wavelength <= window[1])
    good = inWindow & (mask == 0) & (norm > 0) & np.isfinite(flux) & (variance > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(good, flux / norm, np.nan)
        snr = np.where(good, flux / np.sqrt(variance), np.nan)
    nGood = good.sum(axis=1)
    nWindow = np.maximum(inWindow.sum(axis=1), 1)
    saturated = (inWindow & ((mask & satBit) != 0)).sum(axis=1) if satBit else np.zeros(len(fiberId), int)
    hasGood = nGood > 0
    medianFlux = np.full(len(fiberId), np.nan)
    medianSnr = np.full(len(fiberId), np.nan)
    medianFlux[hasGood] = np.nanmedian(ratio[hasGood], axis=1)
    medianSnr[hasGood] = np.nanmedian(snr[hasGood], axis=1)
    return pd.DataFrame(
        {
            "fiberId": np.asarray(fiberId, dtype=int),
            "flux": medianFlux,
            "snr": medianSnr,
            "nGood": nGood.astype(int),
            "satFraction": saturated / nWindow,
        }
    )


def addPositions(table: pd.DataFrame, fiberId: np.ndarray, pfiCenter: np.ndarray) -> pd.DataFrame:
    """Return `fiberFlux`'s table with each fiber's focal-plane position, ``x`` and ``y``.

    Parameters
    ----------
    table : `pandas.DataFrame`
        Rows with ``fiberId``.
    fiberId : `numpy.ndarray` [`int`], shape ``(numFibers,)``
        The pfsConfig's fibers.
    pfiCenter : `numpy.ndarray` [`float`], shape ``(numFibers, 2)``
        Their positions (mm), ``PfsConfig.pfiCenter``. Read from FITS, these
        are big-endian, which pandas cannot join on: they are converted.

    Returns
    -------
    `pandas.DataFrame`
        ``table`` with ``x`` and ``y``; NaN for a fiber the pfsConfig lacks.
    """
    center = np.asarray(pfiCenter, dtype=np.float64)
    positions = pd.DataFrame(
        {"fiberId": np.asarray(fiberId, dtype=np.int64), "x": center[:, 0], "y": center[:, 1]}
    )
    return table.merge(positions, on="fiberId", how="left")


def _quadraticFlat(x: np.ndarray, y: np.ndarray, values: np.ndarray, nIter: int = 3, clip: float = 4.0):
    """Return ``values`` over a robust quadratic fit in (x, y)."""
    design = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    use = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    if use.sum() <= design.shape[1]:
        return np.full_like(values, np.nan)
    for _ in range(nIter):
        coeffs = np.linalg.lstsq(design[use], values[use], rcond=None)[0]
        resid = values - design @ coeffs
        rms = _robustRms(resid[use])
        if not np.isfinite(rms) or rms == 0:
            break
        use = np.isfinite(resid) & (np.abs(resid) < clip * rms)
    return values / (design @ coeffs)


def _perDetectorMedian(values: np.ndarray, spectrograph: np.ndarray) -> np.ndarray:
    """Return each value's detector median, aligned with ``values``."""
    return pd.Series(values).groupby(spectrograph).transform("median").to_numpy()


def sequenceConsistency(
    fibers: pd.DataFrame,
    minSnr: float = 15.0,
    brightnessArm: str = "b",
    minExposures: int = MIN_EXPOSURES,
) -> pd.DataFrame:
    """Compare the exposures of each twilight sequence.

    Parameters
    ----------
    fibers : `pandas.DataFrame`
        `fiberFlux` rows of the fibers to use (lit, good, on the sky), with
        ``visit``, ``arm``, ``spectrograph``, ``sequence``, ``x``, ``y`` and
        optionally ``seqName``: the ``twilightFiberFlux`` datasets of the
        sequences, concatenated.
    minSnr : `float`
        Brightness floor: the median S/N per pixel, in ``brightnessArm``, an
        exposure needs for its RMS to be measured and to serve as a reference.
    brightnessArm : `str`
        The arm whose S/N sets an exposure's brightness; an exposure without it
        is judged by each arm's own S/N.
    minExposures : `int`
        Exposures a sequence needs in an arm: fewer, and nothing is measured.

    Returns
    -------
    `pandas.DataFrame`
        One row per (visit, arm, spectrograph) of a measured sequence:
        `METRIC_COLUMNS`. ``twilightResidRms`` and the diagnostics are NaN for
        an exposure below the floor or with fewer than two bright references.
    """
    # No row is dropped: a fiber with no measured flux stays NaN, and one reading zero stays zero, so a
    # detector that failed in an exposure is judged there rather than missing.
    fibers = fibers.copy()
    if "seqName" not in fibers:
        fibers["seqName"] = ""
    positive = fibers["flux"].where(np.isfinite(fibers["flux"]) & (fibers["flux"] > 0))
    fibers["rel"] = fibers["flux"] / positive.groupby([fibers["visit"], fibers["arm"]]).transform("median")
    fibers["err"] = _MEDIAN_ERROR / (fibers["snr"] * np.sqrt(fibers["nGood"].clip(lower=1)))
    armSnr = fibers.groupby(["visit", "arm"])["snr"].median()
    bright = (
        armSnr.xs(brightnessArm, level="arm")
        if brightnessArm in armSnr.index.get_level_values("arm")
        else None
    )

    rows, residuals = [], []
    for (sequence, arm), seq in fibers.groupby(["sequence", "arm"], sort=False):
        visits = sorted(seq["visit"].unique())
        if len(visits) < minExposures:
            continue
        wide = seq.pivot_table(index="fiberId", columns="visit", values="rel", dropna=False)
        reference = _supportedMedian(wide, visits, minExposures)
        brightness = {
            visit: float(bright[visit])
            if bright is not None and visit in bright.index
            else float(armSnr[(visit, arm)])
            for visit in visits
        }
        brightVisits = [visit for visit in visits if brightness[visit] >= minSnr]
        brightReference = _supportedMedian(wide, brightVisits, minExposures) if brightVisits else None
        noiseOf = _referenceNoise(seq, brightVisits) if len(brightVisits) >= minExposures else {}
        for visit, frame in seq.groupby("visit"):
            frame = frame.reset_index(drop=True)
            fiberIds = frame["fiberId"].to_numpy()
            spectrograph = frame["spectrograph"].to_numpy()
            ratio = frame["rel"].to_numpy() / reference.reindex(fiberIds).to_numpy()
            levels = pd.Series(ratio).groupby(spectrograph).median() / np.nanmedian(ratio)
            measuredFibers = pd.Series(np.isfinite(frame["flux"].to_numpy())).groupby(spectrograph).sum()
            levels[measuredFibers == 0] = np.inf

            tooFaint = brightness[visit] < minSnr
            measured = not tooFaint and len(brightVisits) >= minExposures
            residRms = noise = patch = np.nan
            if measured:
                ratioOut = frame["rel"].to_numpy() / brightReference.reindex(fiberIds).to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):  # a dead detector's 0/0 stays NaN
                    ratioOut = ratioOut / _perDetectorMedian(ratioOut, spectrograph)
                flat = _quadraticFlat(frame["x"].to_numpy(), frame["y"].to_numpy(), ratioOut)
                noise = noiseOf[visit]
                observed = 100 * _robustRms(flat - 1)
                residRms = float(np.sqrt(max(observed**2 - noise**2, 0.0)))
                patch = _deepestPatch(frame["x"].to_numpy(), frame["y"].to_numpy(), flat)
                residuals.append(
                    pd.DataFrame({"visit": visit, "arm": arm, "fiberId": fiberIds, "flat": flat})
                )
            saturation = frame.groupby("spectrograph")["satFraction"].max()
            for spec, level in levels.items():
                rows.append(
                    {
                        "visit": int(visit),
                        "arm": arm,
                        "spectrograph": int(spec),
                        "sequence": sequence,
                        "seqName": frame["seqName"].iloc[0],
                        "nExposures": len(visits),
                        "snr": float(armSnr[(visit, arm)]),
                        "brightness": brightness[visit],
                        "tooFaint": bool(tooFaint),
                        "nReference": len(brightVisits) if measured else 0,
                        "noise": noise,
                        "satFraction": float(saturation.get(spec, np.nan)),
                        "nMeasured": int(measuredFibers.get(spec, 0)),
                        "twilightLevelOffset": 100 * (float(level) - 1),
                        "twilightResidRms": residRms,
                        "twilightPatch": patch,
                    }
                )
    metrics = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(METRIC_COLUMNS))
    split = _armMeanAndHalfDiff(residuals)
    if not metrics.empty:
        metrics = metrics.merge(split, on="visit", how="left")
    return metrics.reindex(columns=list(METRIC_COLUMNS))


def _referenceNoise(seq: pd.DataFrame, visits: list[int]) -> dict[int, float]:
    """Return, per exposure, the robust RMS (%) photon noise leaves against the median of ``visits``.

    Each exposure's median per-fiber error is drawn for `_NOISE_DRAWS` fibers, and the residuals against
    the per-fiber median of all of them measured as the data are. With N odd an exposure is often the
    median itself, and the errors differ through the sequence, so no closed form fits.
    """
    errors = np.array([float(np.nanmedian(seq.loc[seq["visit"] == visit, "err"])) for visit in visits])
    draws = np.random.default_rng(0).normal(0.0, 1.0, (_NOISE_DRAWS, len(visits))) * errors
    resid = draws - np.median(draws, axis=1, keepdims=True)
    return {visit: 100 * _robustRms(resid[:, ii]) for ii, visit in enumerate(visits)}


def _supportedMedian(wide: pd.DataFrame, visits: list[int], minExposures: int) -> pd.Series:
    """Return the per-fiber median over ``visits``, NaN for a fiber measured in fewer than ``minExposures``."""
    columns = wide[visits]
    return columns.median(axis=1).where(columns.notna().sum(axis=1) >= minExposures)


def _deepestPatch(
    x: np.ndarray, y: np.ndarray, flat: np.ndarray, neighbours: int = PATCH_NEIGHBOURS
) -> float:
    """Return the largest absolute median (%) of ``flat - 1`` over each fiber's nearest neighbours."""
    use = np.isfinite(flat) & np.isfinite(x) & np.isfinite(y)
    if use.sum() < neighbours:
        return np.nan
    points = np.column_stack([x[use], y[use]])
    _, index = cKDTree(points).query(points, k=neighbours)
    return float(100 * np.nanmax(np.abs(np.median(flat[use][index], axis=1) - 1)))


def _armMeanAndHalfDiff(residuals: list[pd.DataFrame]) -> pd.DataFrame:
    """Return, per visit, the RMS (%) of the mean and half-difference of its b and r residuals."""
    columns = ["visit", "twilightArmMean", "twilightArmHalfDiff"]
    if not residuals:
        return pd.DataFrame(columns=columns)
    wide = pd.concat(residuals).pivot_table(index=["visit", "fiberId"], columns="arm", values="flat")
    if not {"b", "r"} <= set(wide.columns):
        return pd.DataFrame(columns=columns)
    wide = wide[["b", "r"]].dropna()
    armMean = ((wide["b"] + wide["r"]) / 2 - 1).groupby(level="visit").agg(lambda vv: 100 * _robustRms(vv))
    halfDiff = ((wide["b"] - wide["r"]) / 2).groupby(level="visit").agg(lambda vv: 100 * _robustRms(vv))
    return pd.DataFrame({"twilightArmMean": armMean, "twilightArmHalfDiff": halfDiff}).reset_index()


def isTwilightSequence(sequence: pd.DataFrame) -> bool:
    """Return whether an IIC sequence, as `pfs.drp.qa.comparison.queries.readSequenceOfVisit` returns it, is a twilight.

    False for no rows (the visit is in no sequence). The test is `observationType`'s, on the
    sequence's type and name, so the opdb and the raw header agree on what a twilight is.
    """
    if sequence.empty:
        return False
    first = sequence.iloc[0]
    header = {"W_SEQTYP": first["sequence_type"], "W_SEQNAM": first["sequence_name"]}
    return observationType(header) == "twilight"


def sequenceReadiness(
    sequence: pd.DataFrame, reduced: dict[int, set[str]], stored: dict[int, set[str]]
) -> tuple[bool, str]:
    """Return whether a twilight sequence can be judged: finished, and every visit's tables stored.

    Parameters
    ----------
    sequence : `pandas.DataFrame`
        The sequence's visits, as `pfs.drp.qa.comparison.queries.readSequenceOfVisit`
        returns them: ``pfs_visit_id``, ``sequence_type``, ``sequence_name``,
        ``finished_at``.
    reduced : `dict` [`int`, `set` [`str`]]
        Per visit, the cameras with a pfsArm.
    stored : `dict` [`int`, `set` [`str`]]
        Per visit, the cameras with a ``twilightFiberFlux``.

    Returns
    -------
    ready : `bool`
        True when the sequence is a twilight, has finished, and every visit has
        a ``twilightFiberFlux`` for every camera it has a pfsArm for. A visit
        with neither has not been reduced yet, so the sequence is not ready.
    why : `str`
        What is ready, or what is missing.
    """
    if sequence.empty:
        return False, "the visit is in no IIC sequence"
    first = sequence.iloc[0]
    if not isTwilightSequence(sequence):
        return False, f"sequence {first['iic_sequence_id']} ({first['sequence_name']!r}) is not a twilight"
    if pd.isna(first["finished_at"]):
        return False, f"sequence {first['iic_sequence_id']} has not finished"
    missing = []
    for visit in sequence["pfs_visit_id"]:
        have = stored.get(int(visit), set())
        if not have:
            missing.append(f"{visit} (no tables yet)")
        elif reduced.get(int(visit), set()) - have:
            missing.append(f"{visit} ({', '.join(sorted(reduced[int(visit)] - have))})")
    if missing:
        return False, f"sequence {first['iic_sequence_id']} is waiting for " + "; ".join(missing)
    return True, f"sequence {first['iic_sequence_id']}: {len(sequence)} visits, all stored"


def twilightThresholds(
    levelWarn: float = 2.0,
    levelFail: float = 3.0,
    rmsWarn: float = 1.5,
    rmsFail: float = 2.5,
    patchWarn: float = PATCH_WARN,
    patchFail: float = PATCH_FAIL,
    unjudgedArms: tuple[str, ...] = ("n",),
    source: str = (
        "Provisional (PIPE2D-1973): from a prototype over 16 twilight sequences, 2025-03 to 2026-09."
        " Normal detectors stepped at most 1.02 % and the Run21 faults 8-14 %; normal bright exposures"
        " reached 1.2 % RMS, and a patch of 1.6 % (r) and 2.0 % (b). To be replaced by the flat precision"
        " the screen response needs."
    ),
) -> pd.DataFrame:
    """Return the twilight thresholds as a `gate` table.

    ``twilightLevelOffset`` is judged on its absolute value. The arms in
    ``unjudgedArms`` get entries with no thresholds, which stop the search:
    deliberately not judged.
    """
    rows = []
    for metric, warn, fail, absolute in (
        ("twilightLevelOffset", levelWarn, levelFail, True),
        ("twilightResidRms", rmsWarn, rmsFail, False),
        ("twilightPatch", patchWarn, patchFail, False),
    ):
        rows.append(
            {
                "metric": metric,
                "arm": None,
                "warn": float(warn),
                "fail": float(fail),
                "higherIsWorse": True,
                "absolute": absolute,
                "provenance": source,
            }
        )
        for arm in unjudgedArms:
            rows.append(
                {
                    "metric": metric,
                    "arm": arm,
                    "warn": np.nan,
                    "fail": np.nan,
                    "higherIsWorse": True,
                    "absolute": absolute,
                    "provenance": f"{arm} arm not judged: its residuals exceed the noise model on clear nights.",
                }
            )
    return pd.DataFrame(rows).astype({"arm": object})


def twilightVerdict(metrics: pd.DataFrame, thresholds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one verdict per exposure and arm: whether it is a good flat.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        `sequenceConsistency` rows.
    thresholds : `pandas.DataFrame` or path, optional
        A `gate` table or thresholds file; `TWILIGHT_THRESHOLDS_FILE` by default.

    Returns
    -------
    `pandas.DataFrame`
        One row per (visit, arm), `VERDICT_COLUMNS`: ``status`` is the worst
        of ``PASS``, ``WARN``, ``FAIL`` over the arm's detector levels, RMS and
        patch; ``TOO_FAINT`` in place of a ``PASS`` for an exposure below the
        floor (its levels are judged, and a step still fails it); empty when
        nothing is judged, as in an arm the thresholds leave out. ``reason`` holds the reasons,
        ``;``-joined. No rows when ``metrics`` has none.
    """
    if metrics.empty:
        return pd.DataFrame(columns=list(VERDICT_COLUMNS))
    thresholds = thresholdsPath(TWILIGHT_THRESHOLDS_FILE) if thresholds is None else thresholds
    judged = judge(metrics.reset_index(drop=True), thresholds)
    judged = judged.merge(
        metrics.reset_index(drop=True)[["sequence", "tooFaint", "twilightArmMean", "twilightArmHalfDiff"]],
        left_on="row",
        right_index=True,
    )
    order = {"": 0, "PASS": 1, "WARN": 2, "FAIL": 3}
    rows = []
    for (visit, arm), frame in judged.groupby(["visit", "arm"], sort=False):
        worst = max(frame["status"], key=lambda status: order.get(status, 0))
        reasons = sorted(set(frame.loc[frame["status"].isin(["WARN", "FAIL"]), "reason"]))
        tooFaint = bool(frame["tooFaint"].iloc[0])
        rows.append(
            {
                "visit": visit,
                "arm": arm,
                "sequence": frame["sequence"].iloc[0],
                "status": "TOO_FAINT" if tooFaint and worst == "PASS" else worst,
                "reason": "; ".join(reasons),
                "twilightArmMean": frame["twilightArmMean"].iloc[0],
                "twilightArmHalfDiff": frame["twilightArmHalfDiff"].iloc[0],
            }
        )
    return pd.DataFrame(rows, columns=list(VERDICT_COLUMNS))


def _thresholdsName(path: Path | str) -> str:
    """Return how a thresholds file is recorded: its name when shipped with the package, else its path."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(thresholdsPath("").resolve()))
    except ValueError:
        return str(path)


def twilightImageVerdict(
    metrics: pd.DataFrame, thresholds: pd.DataFrame | Path | str | None = None
) -> pd.DataFrame:
    """Return one verdict per detector image, for the per-image gate: the ``twilightQaMetrics`` rows.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        `sequenceConsistency` rows.
    thresholds : `pandas.DataFrame` or path, optional
        A `gate` table or thresholds file; `TWILIGHT_THRESHOLDS_FILE` by default.
        A file is recorded in `pfs.drp.qa.metrics.gate.THRESHOLDS_COLUMNS` (by
        its name when it is a shipped file, else by its path) so the qadb loader
        can cite it in ``qa_gate``; a table has no file and records none.

    Returns
    -------
    `pandas.DataFrame`
        `IMAGE_VERDICT_COLUMNS`: each metrics row with ``qaStatus`` (the worst
        of the detector's level offset and its arm's RMS and patch: ``PASS``,
        ``WARN`` or ``FAIL``; ``UNKNOWN`` for an arm the thresholds leave out),
        ``qaDecidedBy``, ``qaReason``, and the thresholds record. An exposure
        below the brightness floor keeps ``tooFaint``; the gate's vocabulary has
        no ``TOO_FAINT``, which `twilightVerdict` gives the arm instead. No rows
        when ``metrics`` has none.

    Notes
    -----
    `twilightVerdict` is the sequence's view, one row per exposure and arm;
    this is the image's, so an arm's worst image verdict is its arm verdict
    (``TOO_FAINT`` read as ``PASS``).
    """
    if metrics.empty:
        return pd.DataFrame(columns=list(IMAGE_VERDICT_COLUMNS))
    if thresholds is None:
        thresholds = thresholdsPath(TWILIGHT_THRESHOLDS_FILE)
    if isinstance(thresholds, str | Path):
        record = thresholdsRecord([_thresholdsName(thresholds)])
    else:
        record = dict.fromkeys(THRESHOLDS_COLUMNS, "")
    rows = metrics.reset_index(drop=True)
    verdicts = gate(rows, thresholds, default="UNKNOWN")
    image = pd.concat([rows, verdicts], axis=1)
    for column, value in record.items():
        image[column] = value
    return image.reindex(columns=list(IMAGE_VERDICT_COLUMNS))
