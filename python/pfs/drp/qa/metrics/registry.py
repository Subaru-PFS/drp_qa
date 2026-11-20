"""What each QA metric is, and the limits it is judged by, declared once.

A `MetricSpec` says what a measured number is: its units, direction, the
reference it is measured against and how to phrase it. A `Check` is a rule
that turns numbers into a verdict: the frames it applies to, a WARN and a FAIL
limit, and whether the limits are fixed or a percentage from the camera's
normal value. `CHECKS` holds every rule there is, 13 of them; nothing else in
the package holds a limit. `pfs.drp.qa.metrics.gate` reads them to judge and
`pfs.drp.qa.metrics.normalValues` to make the normal values, so the two cannot
disagree on what a check covers.

A fixed check needs only the number measured. A relative check compares it
with the camera's normal value for that frame type and lamp or species
(`Check.by`), and gives no verdict without one. A value at a limit has crossed
it.

`EXCLUSIONS` names what is measured and stored but not judged, with the reason.

What a frame is comes from its IIC sequence, here and nowhere else:
`SEQUENCE_TYPES` gives each sequence type its frame type, and `frameType` adds
the sequence name and the illumination. A frame type either has checks or is
in `NOT_JUDGED` with the reason it has none. A sequence type the table lacks
gives no frame type, so a new one shows up as untyped instead of being judged
as something it is not.

Nothing here imports the LSST stack.
"""

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ALLSKY",
    "ARC",
    "BIAS",
    "CHECKS",
    "DARK",
    "DITHERED",
    "DOME_FLAT",
    "DOT_ROACH",
    "ENGINEERING",
    "EXCLUSIONS",
    "FIBER_PROFILE",
    "FIXED",
    "FRAME_TYPES",
    "IIS_ARC",
    "IIS_QUARTZ",
    "METRIC_SPECS",
    "MIN_LINES_PER_FIBER",
    "MIN_NORMAL_FRAMES",
    "M_ARM_OFFSETS",
    "NOT_JUDGED",
    "QUARTZ",
    "RELATIVE",
    "SCIENCE",
    "SEQUENCE_TYPES",
    "THROUGH_FOCUS",
    "TWILIGHT",
    "UNJUDGED_SPECIES",
    "WINDOWED",
    "Check",
    "MetricSpec",
    "addSpecies",
    "checkFor",
    "frameType",
    "lampName",
    "lampOf",
    "observationType",
    "sequenceOf",
    "specFor",
    "whyNotJudged",
]

#: Frame types a `Check` names: the judged ones. ``IIS_ARC`` and ``IIS_QUARTZ`` are lit through the 16
#: engineering fibers of each spectrograph by the IIS lamps; the others through the science fibers.
SCIENCE, ALLSKY, ARC, QUARTZ, TWILIGHT, IIS_ARC, IIS_QUARTZ = (
    "science",
    "allsky",
    "arc",
    "quartz",
    "twilight",
    "iisArc",
    "iisQuartz",
)
FRAME_TYPES = (SCIENCE, ALLSKY, ARC, QUARTZ, TWILIGHT, IIS_ARC, IIS_QUARTZ)

#: Frame types that are named and not judged, each with the reason no check applies to it.
DARK, BIAS, FIBER_PROFILE, DOT_ROACH, THROUGH_FOCUS, DITHERED, ENGINEERING, DOME_FLAT, WINDOWED = (
    "dark",
    "bias",
    "fiberProfile",
    "dotRoach",
    "throughFocus",
    "dithered",
    "engineering",
    "domeFlat",
    "windowed",
)
NOT_JUDGED = {
    DARK: "No lamp or sky is exposed.",
    BIAS: "No lamp or sky is exposed.",
    FIBER_PROFILE: (
        "The slit is dithered in x between the exposures of the sequence, and one group of fibers is lit at a "
        "time."
    ),
    DOT_ROACH: "Cobras are moved between the exposures of the sequence, so the fibers lit change within it.",
    THROUGH_FOCUS: "The focus is stepped between the exposures of the sequence.",
    DITHERED: "The slit is dithered between the exposures of the sequence.",
    ENGINEERING: (
        "Taken with the IIC's plain expose command, with any lamp, exposure time, cameras and row binning: no "
        "normal value is kept for it."
    ),
    DOME_FLAT: "No normal value is kept for a dome flat.",
    WINDOWED: "Only part of the detector is read.",
}

#: The frame type each IIC sequence type gives (``W_SEQTYP``, the opdb's ``iic_sequence.sequence_type``),
#: before the sequence name and the illumination are read (`frameType`). A type missing here has no frame type.
SEQUENCE_TYPES = {
    "scienceObject": SCIENCE,
    "scienceObject_windowed": SCIENCE,
    "scienceArc": ARC,
    "scienceTrace": QUARTZ,
    "darks": DARK,
    "masterDarks": DARK,
    "scienceDark": DARK,
    "biases": BIAS,
    "masterBiases": BIAS,
    "fiberProfiles": FIBER_PROFILE,
    "dotRoach": DOT_ROACH,
    "dotRoachInit": DOT_ROACH,
    "dotRoachNew": DOT_ROACH,
    "dotScan": DOT_ROACH,
    "detThroughFocus": THROUGH_FOCUS,
    "fpaThroughFocus": THROUGH_FOCUS,
    "slitThroughFocus": THROUGH_FOCUS,
    "defocusedArcs": THROUGH_FOCUS,
    "ditheredArcs": DITHERED,
    "ditheredFlats": DITHERED,
    "arcs": ENGINEERING,
    "arcs_binned": ENGINEERING,
    "flats": ENGINEERING,
    "flats_binned": ENGINEERING,
    "domeFlat": DOME_FLAT,
    "scienceTrace_windowed": WINDOWED,
}

#: The suffix the IIC adds to the type of a sequence that resumes another: the same kind of frame.
_CONTINUED = "_continued"

#: ``obsType``, the measurement the tasks make on a frame, for the frame types that have one of their own.
#: Any other frame is ``unknown``: the tasks measure what its lines allow.
_OBS_TYPES = {
    SCIENCE: "science",
    ALLSKY: "allsky",
    TWILIGHT: "twilight",
    ARC: "arc",
    QUARTZ: "trace",
    DARK: "dark",
    BIAS: "bias",
}

#: Frame types lit through the science fibers unless the illumination says the engineering fibers.
_LIT = (SCIENCE, ALLSKY, TWILIGHT, ARC, QUARTZ)

#: Spellings of a lamp, in sequence names and lamp headers, by the one name a normal value is kept under.
_LAMPS = {
    "ne": "Ne",
    "neon": "Ne",
    "ar": "Ar",
    "argon": "Ar",
    "kr": "Kr",
    "krypton": "Kr",
    "xe": "Xe",
    "xenon": "Xe",
    "hgcd": "HgCd",
    "hgar": "HgAr",
    "he": "He",
    "helium": "He",
    "h": "H",
    "hydrogen": "H",
    "d": "D",
    "deuterium": "D",
    "quartz": "Quartz",
    "halogen": "Quartz",
    "qth": "Quartz",
}

#: Kinds of `Check`: limits in the metric's own units, or in per cent from the normal value.
FIXED, RELATIVE = "fixed", "relative"

#: A normal value made from fewer exposures than this judges nothing, however many detectors each gave.
MIN_NORMAL_FRAMES = 3

#: A width or a line count resting on fewer lines per lit fiber than this is not judged.
MIN_LINES_PER_FIBER = 5.0

#: Line species whose scatter is not judged.
UNJUDGED_SPECIES = ("HgI", "CdI")

#: The normal-value rows for visits read with the m arm: the b arm's standing offset from the calibration in
#: x and the n arm's in y, by metric name, with the arm each applies to.
M_ARM_OFFSETS = {"mArmOffsetX": "b", "mArmOffsetY": "n"}


def _text(value: object) -> str:
    """Return a header or table value as stripped text; empty for `None`, NaN or anything not a string."""
    return value.strip() if isinstance(value, str) else ""


def sequenceOf(*headers: Any) -> tuple[str, str]:
    """Return the sequence type and name of a frame from its headers.

    Parameters
    ----------
    *headers : mapping or `None`
        Headers in order of trust, the raw exposure's (or the calexp's, which
        copies it) first and the pfsConfig's after: Run25 and Run26 pfsConfigs
        lack ``W_SEQTYP``, and so do raws from before the IIC wrote it.

    Returns
    -------
    seqType, seqName : `str`
        ``W_SEQTYP`` and ``W_SEQNAM`` of the first header that has a
        ``W_SEQTYP``, stripped; two empty strings when none has.
    """
    for header in headers:
        if header is None:
            continue
        seqType = _text(header.get("W_SEQTYP"))
        if seqType:
            return seqType, _text(header.get("W_SEQNAM"))
    return "", ""


def _sequenceKind(seqType: object, seqName: object = None) -> str | None:
    """Return the frame type a sequence gives before its illumination is read, or `None`."""
    name = _text(seqType)
    while name.endswith(_CONTINUED):
        name = name[: -len(_CONTINUED)]
    kind = SEQUENCE_TYPES.get(name)
    if kind == SCIENCE:
        # Twilights and all-sky frames are taken as science sequences; only the name tells them apart.
        named = _text(seqName).lower()
        if named.startswith("twilight"):
            return TWILIGHT
        if named.startswith("sky"):
            return ALLSKY
    return kind


def observationType(seqType: object, seqName: object = None) -> str:
    """Return ``obsType``: which measurement the tasks make on a frame of this sequence.

    Parameters
    ----------
    seqType, seqName : `str`
        The sequence type and name (`sequenceOf`).

    Returns
    -------
    `str`
        ``arc``, ``trace`` (a quartz), ``science``, ``allsky``, ``twilight``,
        ``dark`` or ``bias`` (nothing is measured on either), else
        ``unknown``: the sequence type is missing, not in `SEQUENCE_TYPES`,
        or of a frame type the tasks have no measurement of their own for.
        It says how a frame was measured, not what it is judged as: that is
        `frameType`.
    """
    return _OBS_TYPES.get(_sequenceKind(seqType, seqName), "unknown")


def frameType(
    seqType: object = None, seqName: object = None, illumination: object = None, obsType: object = None
) -> str | None:
    """Return the frame type of an image, or `None` when it has none.

    Parameters
    ----------
    seqType : `str`, optional
        The IIC sequence type, as the tasks store it (``seqType``).
    seqName : `str`, optional
        The sequence name: it tells a twilight and an all-sky frame from
        science. A table without it (the line residuals) gives ``obsType``,
        which the tasks made from the name, and that tells them apart.
    illumination : `str`, optional
        ``full`` or ``iis``; anything else is taken as ``full``. Read only for
        the frame types a lamp or the sky lights: a dark has none, whatever
        its pfsConfig says of the engineering fibers.
    obsType : `str`, optional
        The tasks' ``obsType``, read when there is no ``seqType`` (rows stored
        before it was a column), and for a science sequence with no
        ``seqName``.

    Returns
    -------
    `str` or `None`
        One of `FRAME_TYPES` or of `NOT_JUDGED`. `None` when the sequence type
        is not in `SEQUENCE_TYPES`, when there is neither a sequence type nor
        a known ``obsType``, and for a science, all-sky or twilight frame with
        the engineering fibers lit.
    """
    stored = {name: kind for kind, name in _OBS_TYPES.items()}.get(_text(obsType))
    if _text(seqType):
        kind = _sequenceKind(seqType, seqName)
        if kind == SCIENCE and not _text(seqName) and stored in (ALLSKY, TWILIGHT):
            kind = stored
    else:
        kind = stored
    if illumination == "iis" and kind in _LIT:
        return {ARC: IIS_ARC, QUARTZ: IIS_QUARTZ}.get(kind)
    return kind


def whyNotJudged(kind: object, seqType: object = None) -> str:
    """Return why an image with no judged check has none: what it is, and the reason.

    Parameters
    ----------
    kind : `str` or `None`
        The image's `frameType`.
    seqType : `str`, optional
        Its sequence type, named when the image has no frame type.

    Returns
    -------
    `str`
        For a frame type of `NOT_JUDGED`, its name and reason; for no frame
        type, that, with the sequence type when there is one; empty for a
        frame type that has checks.
    """
    if isinstance(kind, str) and kind in NOT_JUDGED:
        return f"{kind} frames are not judged: {NOT_JUDGED[kind]}"
    if isinstance(kind, str) and kind:
        return ""
    named = _text(seqType)
    return f"no frame type for sequence type {named}" if named else "no frame type"


def lampOf(seqName: object, lamps: object = None) -> str:
    """Return the lamp of a frame, under the name a normal value is kept by, or an empty string.

    Parameters
    ----------
    seqName : `str`
        The sequence name, read when ``lamps`` is empty: the part of
        ``W_SEQNAM`` after the colon (``"Arc: Neon"`` gives ``"Ne"``), else
        the part in trailing parentheses (``"ImageQuality (Kr)"`` gives
        ``"Kr"``). A sequence can be named anything, so this is the fallback.
    lamps : `str`, optional
        The lamps the image's header says were lit, as the tasks store them
        (``lamps``): obs_pfs's names joined by ``+``, e.g. ``"Kr_eng"`` or
        ``"HgAr+Ne"``.

    Returns
    -------
    `str`
        `lampName` of the header's lamps, else of the part of the sequence
        name.
    """
    header = lampName(lamps)
    if header:
        return header
    name = _text(seqName)
    if ":" in name:
        return lampName(name.split(":", 1)[-1])
    match = re.search(r"\(([^()]+)\)\s*$", name)
    return lampName(match.group(1)) if match else ""


def lampName(lamps: object) -> str:
    """Return lamps under the names a normal value is kept by, whichever way they are spelled.

    Parameters
    ----------
    lamps : `str`
        One lamp or several joined by ``+``, as a header, a sequence name or
        a stored normal value spells them.

    Returns
    -------
    `str`
        One spelling per lamp whatever the source (``Ne`` for ``Neon``,
        ``Ne_eng`` and ``Ne``), several sorted and joined by ``+``. The
        engineering-fiber suffix is dropped: the frame type says it. A part
        that is no known lamp is returned as it is, stripped. Empty for
        `None`, NaN or an empty string.
    """
    parts = (part.strip().removesuffix("_eng") for part in _text(lamps).split("+"))
    return "+".join(sorted({_LAMPS.get(part.lower(), part) for part in parts if part}))


def addSpecies(metrics):
    """Return ``metrics`` with a ``species`` column, the `lampOf` each row.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Metrics with a ``seqName`` column, and ``lamps`` when stored.

    Returns
    -------
    `pandas.DataFrame`
        A copy with ``species`` added; unchanged if ``seqName`` is absent.
    """
    if "seqName" not in metrics.columns:
        return metrics
    lamps = metrics["lamps"] if "lamps" in metrics.columns else [None] * len(metrics)
    return metrics.assign(species=[lampOf(n, lit) for n, lit in zip(metrics["seqName"], lamps, strict=True)])


@dataclass(frozen=True)
class MetricSpec:
    """One metric's declaration.

    Attributes
    ----------
    name : `str`
        Column name in a wide table, or the ``metric`` value in a long one.
    higherIsWorse : `bool`
        Direction of the metric.
    absolute : `bool`
        Judge the absolute value: an offset is as bad in either direction.
    units : `str`
        Physical units, for axis labels.
    reference : `str`
        What the metric is measured against.
    label : `str`
        Name in reason strings; defaults to ``name``.
    valueFormat : `str`
        Format spec for the value in reason strings.
    unitSuffix : `str`
        Suffix for values in reason strings, e.g. ``" px"``.
    notMeasuredWhen : `tuple` [`str`]
        Boolean columns that, when true, mark the value as not measured from
        the exposure (a width read back from ``fiberProfiles``, a twilight
        too faint to measure). Such a value is not judged.
    """

    name: str
    higherIsWorse: bool = True
    absolute: bool = False
    units: str = ""
    reference: str = ""
    label: str = ""
    valueFormat: str = ".4g"
    unitSuffix: str = ""
    notMeasuredWhen: tuple[str, ...] = ()

    def notMeasured(self, frame) -> np.ndarray:
        """Return, per row of ``frame``, whether the value was not measured.

        Parameters
        ----------
        frame : `pandas.DataFrame`
            Metrics rows; a `notMeasuredWhen` column it lacks marks nothing.

        Returns
        -------
        `numpy.ndarray` [`bool`]
            True where any `notMeasuredWhen` column is true.
        """
        mask = np.zeros(len(frame), dtype=bool)
        for column in self.notMeasuredWhen:
            if column in frame.columns:
                values = frame[column].to_numpy(dtype=object)
                mask |= np.array([isinstance(v, (bool, np.bool_)) and bool(v) for v in values], dtype=bool)
        return mask

    def show(self, value: float) -> str:
        """Return ``value`` as reason strings show it, with its unit."""
        shown = format(value, self.valueFormat) if math.isfinite(value) else str(value)
        return f"{shown}{self.unitSuffix}"


@dataclass(frozen=True)
class Check:
    """One rule: the frames it judges, and its WARN and FAIL.

    Attributes
    ----------
    name : `str`
        The check's name, as ``qa_gate.qa_decided_by`` records it.
    title : `str`
        What it judges, for reason strings and tables.
    metrics : `tuple` [`str`]
        The `MetricSpec` names it reads. A check on several judges each and
        takes the worst.
    frames : `tuple` [`str`]
        The `FRAME_TYPES` it applies to.
    warn, fail : `float`
        The limits: in the metric's units for a `FIXED` check, in per cent
        from the normal value, in the metric's bad direction, for a
        `RELATIVE` one.
    kind : `str`
        `FIXED` or `RELATIVE`.
    arms : `tuple` [`str`]
        The arms it applies to; every arm when empty.
    by : `tuple` [`str`]
        `RELATIVE` only: what a normal value is kept per, besides the frame
        type. ``camera`` or ``arm``, then ``lamp`` (the lamp of the sequence)
        or ``species`` (the line species), or neither.
    """

    name: str
    title: str
    metrics: tuple[str, ...]
    frames: tuple[str, ...]
    warn: float
    fail: float
    kind: str = FIXED
    arms: tuple[str, ...] = ()
    by: tuple[str, ...] = ()

    def limits(self, normal: float | None = None, higherIsWorse: bool = True) -> tuple[float, float]:
        """Return WARN and FAIL in the metric's units.

        Parameters
        ----------
        normal : `float`, optional
            The normal value, for a `RELATIVE` check.
        higherIsWorse : `bool`, optional
            The metric's direction, for a `RELATIVE` check.
        """
        if self.kind == FIXED:
            return self.warn, self.fail
        sign = 1.0 if higherIsWorse else -1.0
        return normal * (1 + sign * self.warn / 100), normal * (1 + sign * self.fail / 100)


_CALIB = (
    "detectorMap_calib, the calibration the exposure is reduced against: an offset from it is how far "
    "the instrument has moved since the calibration was made."
)
_VISIT = (
    "The visit's own detectorMap, adjusted to the visit's lines: the lines the adjustment held back "
    "(RESERVED) were not fitted, so their residuals say how well the adjusted map predicts a line."
)
_TWILIGHT = (
    "The other exposures of the same twilight sequence, minutes apart: each fiber's flux / norm against their "
    "median."
)
_SATURATION = (
    "The detector's saturation level, as ISR marks it in the calexp's SAT mask plane (on an H4RG, "
    "a ramp leaving the valid range)."
)
_LINE_FLUX = (
    "The same line species on the same detector at another epoch: within one image the lines of a "
    "species span orders of magnitude, so it is compared across visits, not judged per image."
)
_NORMAL = (
    " Judged against the camera's normal value for the frame type: cameras differ by more than the tolerance."
)

#: The metrics, by name.
METRIC_SPECS = {
    spec.name: spec
    for spec in (
        MetricSpec(
            "medFwhm",
            units="pixels",
            reference=(
                "A Gaussian-equivalent FWHM from arc-line second moments, or a cross-dispersion fit to "
                "calexp pixels on a quartz." + _NORMAL
            ),
            label="width",
            valueFormat=".2f",
            unitSuffix=" px",
            notMeasuredWhen=("traceOnly",),
        ),
        MetricSpec(
            "medDxCenter",
            absolute=True,
            units="pixels",
            reference=_CALIB,
            label="trace centre offset",
            valueFormat="+.2f",
            unitSuffix=" px",
        ),
        MetricSpec("dxCenterRms", units="pixels", reference=_CALIB, valueFormat=".3f", unitSuffix=" px"),
        MetricSpec(
            "pctFlagged",
            units="percent",
            reference="The line list and fitDetectorMap's flag bits on the arc lines.",
            valueFormat=".1f",
            unitSuffix=" %",
        ),
        MetricSpec(
            "nLines",
            higherIsWorse=False,
            units="lines",
            reference="The line list of the lamp.",
            valueFormat=".0f",
        ),
        MetricSpec(
            "linesPerFiber",
            higherIsWorse=False,
            units="lines per lit fiber",
            reference=(
                "The line list of the lamp, on the fibers the pfsConfig lights; on a quartz, the "
                "detector rows a trace is found on." + _NORMAL
            ),
            label="lines per lit fiber",
            valueFormat=".1f",
        ),
        # dmLineResidualMetrics: per line species and line set, against each reference.
        MetricSpec(
            "medianDeltaX",
            absolute=True,
            units="pixels",
            reference=_CALIB,
            label="x offset",
            valueFormat="+.2f",
            unitSuffix=" px",
        ),
        MetricSpec(
            "medianDeltaY",
            absolute=True,
            units="pixels",
            reference=_CALIB,
            label="y offset",
            valueFormat="+.2f",
            unitSuffix=" px",
        ),
        MetricSpec(
            "robustRmsDeltaX",
            units="pixels",
            reference=_VISIT + _NORMAL,
            label="x scatter",
            valueFormat=".3f",
            unitSuffix=" px",
        ),
        MetricSpec(
            "robustRmsDeltaY",
            units="pixels",
            reference=_VISIT + _NORMAL,
            label="y scatter",
            valueFormat=".3f",
            unitSuffix=" px",
        ),
        MetricSpec(
            "medianDeltaYKms",
            absolute=True,
            units="km/s",
            reference=_VISIT
            + " Each line's residual is converted with the detectorMap's dispersion at that line.",
            label="residual",
            valueFormat="+.2f",
            unitSuffix=" km/s",
        ),
        MetricSpec(
            "fitXRms",
            units="pixels",
            reference="The fitDetectorMap solution, per line species, from the reduceExposure log.",
            unitSuffix=" px",
        ),
        MetricSpec(
            "fitYRms",
            units="pixels",
            reference="The fitDetectorMap solution, per line species, from the reduceExposure log.",
            unitSuffix=" px",
        ),
        MetricSpec(
            "pctSaturatedPixels", units="percent", reference=_SATURATION, valueFormat=".3f", unitSuffix=" %"
        ),
        MetricSpec("nSaturatedLines", units="lines", reference=_SATURATION, valueFormat=".0f"),
        MetricSpec("medLineFlux", higherIsWorse=False, units="counts", reference=_LINE_FLUX),
        MetricSpec("robustRmsLineFlux", units="counts", reference=_LINE_FLUX),
        # Twilight sky flats: each exposure against the other exposures of its sequence
        # (pfs.drp.qa.metrics.twilight).
        MetricSpec(
            "twilightLevelOffset",
            absolute=True,
            units="percent",
            reference=_TWILIGHT,
            label="level offset",
            valueFormat="+.2f",
            unitSuffix=" %",
        ),
        MetricSpec(
            "twilightResidRms",
            units="percent",
            reference=_TWILIGHT,
            label="residual scatter",
            valueFormat=".2f",
            unitSuffix=" %",
            notMeasuredWhen=("tooFaint",),
        ),
        MetricSpec(
            "twilightPatch",
            units="percent",
            reference=_TWILIGHT,
            label="patch",
            valueFormat=".2f",
            unitSuffix=" %",
            notMeasuredWhen=("tooFaint",),
        ),
    )
}

_SKY = (SCIENCE, ALLSKY)
_LAMP = (ARC, QUARTZ)
_ALL_LAMP = (ARC, QUARTZ, IIS_ARC, IIS_QUARTZ)

#: Every rule, by name: the 13 pairs of limits. `pfs.drp.qa.metrics.gate` says which rows of which table each
#: reads.
CHECKS = {
    check.name: check
    for check in (
        # Science and all-sky frames: two fixed checks, the same for short and long exposures.
        Check(
            "skyOffset",
            "offset from the calibration",
            ("medianDeltaX", "medianDeltaY"),
            _SKY,
            warn=0.5,
            fail=1.0,
        ),
        Check(
            "skyLineResidual",
            "OH residual after the visit's adjustment",
            ("medianDeltaYKms",),
            _SKY,
            warn=0.5,
            fail=1.0,
            arms=("r", "n", "m"),
        ),
        Check(
            "skyLineResidualBlue",
            "[O I] residual after the visit's adjustment",
            ("medianDeltaYKms",),
            _SKY,
            warn=1.5,
            fail=2.5,
            arms=("b",),
        ),
        # Calibration frames.
        Check(
            "calibOffset",
            "offset from the calibration",
            ("medianDeltaX", "medianDeltaY"),
            _LAMP,
            warn=1.0,
            fail=2.0,
        ),
        Check(
            "traceCentreOffset",
            "trace centre offset",
            ("medDxCenter",),
            (IIS_ARC, IIS_QUARTZ),
            warn=1.0,
            fail=2.0,
        ),
        # Twilights: each exposure against the sequence's others.
        Check(
            "twilightLevel",
            "twilight level offset",
            ("twilightLevelOffset",),
            (TWILIGHT,),
            warn=2.0,
            fail=3.0,
            arms=("b", "r", "m"),
        ),
        Check(
            "twilightScatter",
            "twilight residual scatter",
            ("twilightResidRms",),
            (TWILIGHT,),
            warn=1.5,
            fail=2.5,
            arms=("b", "r", "m"),
        ),
        Check(
            "twilightPatch",
            "twilight patch",
            ("twilightPatch",),
            (TWILIGHT,),
            warn=2.5,
            fail=4.0,
            arms=("b", "r", "m"),
        ),
        # Against each camera's normal value.
        Check(
            "width",
            "line and trace width",
            ("medFwhm",),
            _ALL_LAMP,
            warn=3.0,
            fail=5.0,
            kind=RELATIVE,
            by=("camera", "lamp"),
        ),
        Check(
            "scatterX",
            "line scatter in x",
            ("robustRmsDeltaX",),
            _LAMP,
            warn=20.0,
            fail=50.0,
            kind=RELATIVE,
            by=("camera", "species"),
        ),
        Check(
            "scatterY",
            "line scatter in y",
            ("robustRmsDeltaY",),
            (ARC,),
            warn=50.0,
            fail=150.0,
            kind=RELATIVE,
            by=("camera", "species"),
        ),
        Check(
            "linesPerFiberArc",
            "lines per lit fiber",
            ("linesPerFiber",),
            (ARC, IIS_ARC),
            warn=25.0,
            fail=50.0,
            kind=RELATIVE,
            by=("arm", "lamp"),
        ),
        Check(
            "linesPerFiberQuartz",
            "lines per lit fiber",
            ("linesPerFiber",),
            (QUARTZ, IIS_QUARTZ),
            warn=2.5,
            fail=5.0,
            kind=RELATIVE,
            by=("arm", "lamp"),
        ),
    )
}

#: What is measured and stored but not judged, and why.
EXCLUSIONS = (
    (
        "Width and line or trace scatter on science and all-sky frames",
        "They follow fiber signal and exposure time: the width is 15 to 30 % wider on faint fibers, and trace "
        "scatter 15 to 20 % higher on 60 s exposures.",
    ),
    (
        "Scatter of mercury and cadmium lines, in x and y",
        "It varies by itself from frame to frame of a healthy run.",
    ),
    (
        "Scatter of trace centroids on arcs and twilights",
        "On a faint continuum they scatter two to three times more than on a quartz.",
    ),
    (
        "Sky species on twilights, and the n arm on twilights",
        "Its residuals exceed the noise model on clear nights.",
    ),
    (
        f"A width resting on fewer than {MIN_LINES_PER_FIBER:g} lines per lit fiber, and a line count whose "
        "normal value is below that",
        "Too few lines for a median: this covers b-arm Argon.",
    ),
    (
        "Line residuals on frames lit through the engineering fibers",
        "No normal values are kept for them. Their width and lines per lit fiber are judged, against their own.",
    ),
    (
        "A relative check whose normal value rests on fewer than "
        f"{MIN_NORMAL_FRAMES} exposures, or that has none",
        "A median of one or two exposures is not a normal value, however many detectors each gave.",
    ),
    (
        "Flagged-line rate (pctFlagged)",
        "It is a property of the line list and the fit's flags, not of the frame.",
    ),
    (
        "Saturation (nSaturatedLines, pctSaturatedPixels) and line flux",
        "Stored for cross-visit comparison; no limit has been set.",
    ),
    *((f"Every number measured on {kind} frames", reason) for kind, reason in NOT_JUDGED.items()),
    (
        "Frames with no frame type",
        "Their sequence type is missing or not in SEQUENCE_TYPES, so nobody has said what they are. No check "
        "applies; the image is UNKNOWN, never PASS.",
    ),
)


def specFor(name: str) -> MetricSpec:
    """Return the spec for ``name``, or a default higher-is-worse one."""
    return METRIC_SPECS.get(name, MetricSpec(name))


def checkFor(name: str) -> Check:
    """Return the check named ``name``."""
    return CHECKS[name]
