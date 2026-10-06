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

#: The metrics with a known treatment, by name. ``pctFlagged`` is split by lamp
#: (``species``, from ``seqName``) because the line lists differ per lamp;
#: ``nLines`` by sequence because the line count is a property of the lamp.
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
