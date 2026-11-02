"""Twilight sky flats: is each exposure of a twilight sequence a good flat? Measurement and verdict.

A twilight is the sky flat behind the screen response: each fiber's flux over
the instrument's response should be the same across the focal plane, but for a
smooth sky gradient. Whatever else spoils it (passing cloud, moonlight, a
detector's exposure or readout, faint sources) shows as a difference between
the exposures of one sequence, which are minutes apart with the same
instrument. So the reference (rule R1 of ``docs/qa-principles.md``) is the
sequence's other exposures, and nothing here depends on a calibration
(PIPE2D-1973).

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
    detector that steps in one exposure is named in that exposure alone. Every
    exposure is judged: the median of a detector's ~600 fibers, spread over the
    field, is not moved by faint sources.
``twilightResidRms``
    The robust RMS, beyond photon noise (%), of the ratio of each fiber to the
    per-fiber median of the sequence's bright exposures, this one included,
    after each detector's median and a robust quadratic over the focal plane
    (the sky's gradient) are divided out. Including the judged exposure keeps
    one bad exposure from moving the reference: with two others, their median
    would be their mean, and a patch in one would show, mirrored at half its
    depth, in both of the others. It needs `MIN_EXPOSURES` bright exposures.
    Only exposures at or above ``minSnr`` in the brightness arm are judged or
    used in the reference: below it (``tooFaint``), starlight and airglow in
    the fibers, shared by the arms, are not yet swamped by the twilight, and the
    exposure is not a good flat. The noise is simulated: each bright
    exposure's median per-fiber error (that of a flat spectrum), drawn for
    every exposure of the reference, through the same median. Real spectra have
    structure in the window, so a few tenths of a per cent of noise can remain
    in the excess.
``twilightPatch``
    The deepest coherent deviation on the focal plane (%): the largest
    absolute median of that residual over each fiber's `PATCH_NEIGHBOURS`
    nearest fibers. The RMS is robust, so it ignores a patch on a few per cent
    of the fibers, and the quadratic absorbs much of one spanning the field; a
    cloud between those is what this sees. Measured on the same exposures as
    the RMS.
``twilightShared``, ``twilightDiffering``
    Diagnostics, for the reason: the robust RMS (%) of the mean and the
    half-difference of the b and r residuals per fiber. The sky lights a fiber
    in every arm alike; a detector or spectrograph does not.

``satFraction`` is stored, not judged: saturated pixels are left out of the
medians, and no saturated twilight has been seen to set a limit by.

Blind spot: anything constant over the sequence (a stationary cloud bank, a
steady moon gradient beyond a quadratic, the instrument's state that night)
cancels. Comparing a twilight with good twilights of other nights covers it,
and is not done here.

The provisional thresholds (`twilightThresholds`) are from a prototype over 16
sequences of 2025-03 to 2026-09; the n arm is not judged, its residuals sit
3-9 % above the noise model even on clear nights.

Nothing here imports the LSST stack.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pfs.drp.qa.metrics.gate import judge

__all__ = [
    "FIBER_COLUMNS",
    "METRIC_COLUMNS",
    "MIN_EXPOSURES",
    "PATCH_NEIGHBOURS",
    "TWILIGHT_METRICS",
    "VERDICT_COLUMNS",
    "WINDOWS",
    "addPositions",
    "fiberFlux",
    "sequenceConsistency",
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
VERDICT_COLUMNS = ("visit", "arm", "sequence", "status", "reason", "twilightShared", "twilightDiffering")

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
    "twilightLevelOffset": "The detector's level against the sequence's, over the field's (%).",
    "twilightResidRms": "Field RMS beyond photon noise against the sequence's other bright exposures (%).",
    "twilightPatch": "Deepest coherent deviation: largest |median| over a fiber's nearest neighbours (%).",
    "twilightShared": "RMS of the residual shared by b and r (%): the sky.",
    "twilightDiffering": "RMS of half the b-r difference of the residuals (%): the instrument.",
}

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
    fibers = fibers[np.isfinite(fibers["flux"]) & (fibers["flux"] > 0)].copy()
    if "seqName" not in fibers:
        fibers["seqName"] = ""
    fibers["rel"] = fibers["flux"] / fibers.groupby(["visit", "arm"])["flux"].transform("median")
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
        wide = seq.pivot_table(index="fiberId", columns="visit", values="rel")
        reference = wide.median(axis=1)
        brightness = {
            visit: float(bright[visit])
            if bright is not None and visit in bright.index
            else float(armSnr[(visit, arm)])
            for visit in visits
        }
        brightVisits = [visit for visit in visits if brightness[visit] >= minSnr]
        brightReference = wide[brightVisits].median(axis=1) if brightVisits else None
        noiseOf = _referenceNoise(seq, brightVisits) if len(brightVisits) >= minExposures else {}
        for visit, frame in seq.groupby("visit"):
            frame = frame.reset_index(drop=True)
            fiberIds = frame["fiberId"].to_numpy()
            spectrograph = frame["spectrograph"].to_numpy()
            ratio = frame["rel"].to_numpy() / reference.reindex(fiberIds).to_numpy()
            levels = pd.Series(ratio).groupby(spectrograph).median() / np.nanmedian(ratio)

            tooFaint = brightness[visit] < minSnr
            measured = not tooFaint and len(brightVisits) >= minExposures
            residRms = noise = patch = np.nan
            if measured:
                ratioOut = frame["rel"].to_numpy() / brightReference.reindex(fiberIds).to_numpy()
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
                        "twilightLevelOffset": 100 * (float(level) - 1),
                        "twilightResidRms": residRms,
                        "twilightPatch": patch,
                    }
                )
    metrics = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(METRIC_COLUMNS))
    split = _sharedAndDiffering(residuals)
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


def _sharedAndDiffering(residuals: list[pd.DataFrame]) -> pd.DataFrame:
    """Return, per visit, the RMS (%) of the mean and half-difference of its b and r residuals."""
    columns = ["visit", "twilightShared", "twilightDiffering"]
    if not residuals:
        return pd.DataFrame(columns=columns)
    wide = pd.concat(residuals).pivot_table(index=["visit", "fiberId"], columns="arm", values="flat")
    if not {"b", "r"} <= set(wide.columns):
        return pd.DataFrame(columns=columns)
    wide = wide[["b", "r"]].dropna()
    shared = ((wide["b"] + wide["r"]) / 2 - 1).groupby(level="visit").agg(lambda vv: 100 * _robustRms(vv))
    differing = ((wide["b"] - wide["r"]) / 2).groupby(level="visit").agg(lambda vv: 100 * _robustRms(vv))
    return pd.DataFrame({"twilightShared": shared, "twilightDiffering": differing}).reset_index()


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
    thresholds : `pandas.DataFrame`, optional
        A `gate` table (`twilightThresholds` by default).

    Returns
    -------
    `pandas.DataFrame`
        One row per (visit, arm), `VERDICT_COLUMNS`: ``status`` is the worst
        of ``PASS``, ``WARN``, ``FAIL`` over the arm's detector levels, RMS and
        patch; ``TOO_FAINT`` for an exposure below the floor that nothing
        failed or warned on (its levels are judged, and a step still fails it);
        empty when nothing is judged. ``reason`` holds the reasons,
        ``;``-joined. No rows when ``metrics`` has none.
    """
    if metrics.empty:
        return pd.DataFrame(columns=list(VERDICT_COLUMNS))
    thresholds = twilightThresholds() if thresholds is None else thresholds
    judged = judge(metrics.reset_index(drop=True), thresholds)
    judged = judged.merge(
        metrics.reset_index(drop=True)[["sequence", "tooFaint", "twilightShared", "twilightDiffering"]],
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
                "status": "TOO_FAINT" if tooFaint and worst in ("", "PASS") else worst,
                "reason": "; ".join(reasons),
                "twilightShared": frame["twilightShared"].iloc[0],
                "twilightDiffering": frame["twilightDiffering"].iloc[0],
            }
        )
    return pd.DataFrame(rows, columns=list(VERDICT_COLUMNS))
