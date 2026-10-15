"""What each QA metric is, declared once.

A `MetricSpec` holds everything about a metric other than its value and its
thresholds: its units, the external reference it is measured against (rule R1
of ``docs/qa-principles.md``), its direction, the populations it is judged in
(R7) and how to phrase it in a reason string. Threshold derivation
(`pfs.drp.qa.metrics.calibration`) and gating (`pfs.drp.qa.metrics.gate`) both
read it, so the two cannot disagree on direction, populations, or which values
were measured at all.

The thresholds themselves are not here: they live in a thresholds file
(`~pfs.drp.qa.metrics.calibration.writeThresholds`), with the provenance of
each (R2).
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["METRIC_SPECS", "MetricSpec", "specFor"]


@dataclass(frozen=True)
class MetricSpec:
    """One metric's declaration.

    Attributes
    ----------
    name : `str`
        Column name in ``iqQaMetrics``, or the ``metric`` value in a
        long-format table.
    higherIsWorse : `bool`
        Direction of the metric.
    absolute : `bool`
        Judge, and derive thresholds from, the absolute value: an offset is as
        bad in either direction.
    groupBy : `tuple` [`str`]
        Columns separating the populations thresholds are derived for. Columns
        absent from the data are dropped from the grouping.
    physicalLimit : `float` or `None`
        A physical limit to use for FAIL.
    tolerance : `tuple` [`float`, `float`] or `None`
        WARN and FAIL as fractions beyond the median of each `toleranceBy`
        population's known-good values, in the bad direction. The reference
        is then the same detector, observed the same way, in the reference
        run (R1), and a median needs far fewer values than a tail percentile.
    toleranceBy : `tuple` [`str`]
        Columns separating the populations of `tolerance`. An image whose
        population has no known-good values is judged by the `groupBy`
        percentiles instead.
    units : `str`
        Physical units, for axis labels.
    reference : `str`
        The external reference the metric is measured against (R1).
    label : `str`
        Name in reason strings; defaults to ``name``.
    valueFormat : `str`
        Format spec for the value in reason strings.
    unitSuffix : `str`
        Suffix for values in reason strings, e.g. ``"px"``.
    notMeasuredWhen : `tuple` [`str`]
        Boolean columns that, when true, mark the value as read back from a
        calibration rather than measured from the exposure. Such a value is
        identical for every exposure reduced against that calibration, so it
        is neither judged nor used to derive thresholds.
    """

    name: str
    higherIsWorse: bool = True
    absolute: bool = False
    groupBy: tuple[str, ...] = ("arm", "obsType")
    physicalLimit: float | None = None
    tolerance: tuple[float, float] | None = None
    toleranceBy: tuple[str, ...] = ()
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

    def crossed(self, value: float, limit: float) -> bool:
        """Return whether ``value`` is at or beyond ``limit`` in the bad direction."""
        return value >= limit if self.higherIsWorse else value <= limit

    def describe(self, value: float, status: str, limit: float) -> str:
        """Return the reason string for a crossed threshold.

        Parameters
        ----------
        value : `float`
            The value judged (absolute, for an ``absolute`` metric).
        status : `str`
            ``"WARN"`` or ``"FAIL"``.
        limit : `float`
            The threshold crossed.

        Returns
        -------
        `str`
            E.g. ``"medFWHM=3.60px >= fail threshold 3.5px"``.
        """
        comparison = ">=" if self.higherIsWorse else "<="
        shown = format(value, self.valueFormat) if math.isfinite(value) else str(value)
        return (
            f"{self.label or self.name}={shown}{self.unitSuffix} {comparison} "
            f"{status.lower()} threshold {limit:g}{self.unitSuffix}"
        )


_CALIB = (
    "detectorMap_calib, the calibration the exposure is reduced against: an offset against an "
    "out-of-date calibration is a calibration problem, not an instrument one."
)

_LINE_REFERENCES = (
    "Per the ``reference`` column: detectorMap_calib (``calib``), as above; or the visit's own "
    "detectorMap (``visit``), adjusted to these lines, whose RESERVED lines it was not fitted to."
)

#: The metrics with a known treatment, by name. ``pctFlagged`` is split by lamp
#: (``species``, from ``seqName``) because the line lists differ per lamp;
#: ``nLines`` and ``linesPerFiber`` by sequence because the line count is a
#: property of the lamp. ``linesPerFiber`` is the one judged: per lit fiber, so a
#: design lighting one fiber group is judged like a full slit, and without the
#: trace centroids an arc also records (PIPE2D-1935).
#: ``medFwhm`` is judged against each detector's own width per lamp
#: (PIPE2D-1942): the spectrographs of one arm differ from each other by more
#: than any one of them varies, and the lamps of one arm differ even more (b:
#: Argon 2.5 px, Krypton 2.85 px), so a per-arm percentile sits at the tail of
#: the widest spectrograph or lamp. Run25's known-good values
#: sit at most 1.4 % above their population's median (3.3 % for Argon), the
#: SM1 defocus 24-41 %.
METRIC_SPECS = {
    spec.name: spec
    for spec in (
        MetricSpec(
            "medFwhm",
            units="pixels",
            reference=(
                "The optical design's spot size: a Gaussian-equivalent FWHM from arc-line second "
                "moments, a cross-dispersion fit to calexp pixels, or the fiberProfiles widths."
            ),
            tolerance=(0.03, 0.05),
            toleranceBy=("arm", "spectrograph", "obsType", "species"),
            label="medFWHM",
            valueFormat=".2f",
            unitSuffix="px",
            notMeasuredWhen=("traceOnly",),
        ),
        MetricSpec(
            "medDxCenter",
            absolute=True,
            units="pixels",
            reference=_CALIB,
            label="|dxCenter|",
            valueFormat=".3f",
            unitSuffix="px",
        ),
        MetricSpec("dxCenterRms", units="pixels", reference=_CALIB, valueFormat=".3f", unitSuffix="px"),
        MetricSpec(
            "pctFlagged",
            groupBy=("obsType", "arm", "species"),
            units="percent",
            reference="The line list and fitDetectorMap's flag bits on the arc lines.",
            valueFormat=".1f",
            unitSuffix="%",
        ),
        MetricSpec(
            "nLines",
            higherIsWorse=False,
            groupBy=("arm", "seqName"),
            units="lines",
            reference="The line list of the lamp.",
            valueFormat=".0f",
        ),
        MetricSpec(
            "linesPerFiber",
            higherIsWorse=False,
            groupBy=("arm", "seqName"),
            units="lines per lit fiber",
            reference=(
                "The line list of the lamp, on the fibers the pfsConfig lights; on a quartz, the "
                "detector rows a trace is found on."
            ),
            valueFormat=".1f",
        ),
        # dmLineResidualMetrics (PIPE2D-1921, PIPE2D-1956): per line species and line set, against
        # each reference (the ``reference`` column). x from every species, y from lamp lines; b-arm
        # offsets from the calibration depend on the camera set (PIPE2D-1927).
        *(
            MetricSpec(
                name,
                absolute=name.startswith("median"),
                groupBy=("reference", "statusType", "arm", "description", "cameraSet"),
                units="pixels",
                reference=_LINE_REFERENCES,
                label=f"|{name}|" if name.startswith("median") else "",
                valueFormat=".3f",
                unitSuffix="px",
            )
            for name in ("medianDeltaX", "robustRmsDeltaX", "medianDeltaY", "robustRmsDeltaY")
        ),
        MetricSpec(
            "fitXRms",
            groupBy=("arm", "description"),
            units="pixels",
            reference="The fitDetectorMap solution, per line species, from the reduceExposure log.",
            unitSuffix="px",
        ),
        MetricSpec(
            "fitYRms",
            groupBy=("arm", "description"),
            units="pixels",
            reference="The fitDetectorMap solution, per line species, from the reduceExposure log.",
            unitSuffix="px",
        ),
    )
}


def specFor(name: str) -> MetricSpec:
    """Return the spec for ``name``, or a default higher-is-worse one."""
    return METRIC_SPECS.get(name, MetricSpec(name))
