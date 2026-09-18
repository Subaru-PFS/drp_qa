"""Declarative metric definitions and the single gating path built on them.

Rule R3 of ``doc/qa-rebuild-plan.md`` separates measurement from judgement:
tasks emit numbers, and a separate layer maps numbers to PASS/WARN/FAIL. A
hand-written if/elif ladder in each task cannot deliver that -- the thresholds,
their direction, their units and the reference they are measured against end up
spread across the task, the config docstrings and the dashboard, and drift
apart.

So a metric declares itself once, as a `MetricDef`, and everything else is
derived:

* gating is `MetricRegistry.gate`, the same code for every metric;
* the reason string is generated, not written per metric;
* ``reference`` records what external reference the metric is measured against
  (R1) and ``provenance`` records where its thresholds came from (R2), in a
  place that cannot be separated from the numbers;
* the dashboard reads ``units``, ``higherIsWorse`` and the thresholds from the
  same definition the pipeline gated with, rather than growing its own copy
  (section 5.5).

This module imports neither the LSST stack nor the Butler. Tasks build their
registry from their own config so that command-line overrides still take effect;
see `pfs.drp.qa.metrics.definitions`.
"""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

__all__ = [
    "STATUS_ORDER",
    "UNKNOWN",
    "GateResult",
    "MetricDef",
    "MetricRegistry",
    "Thresholds",
    "worstStatus",
]

#: Gate verdicts, from best to worst.
#:
#: ``UNKNOWN`` is deliberately absent from the *ordering*. "We could not
#: measure" is a statement about a whole quantum, not about one value, and only
#: the task knows whether *every* measurement was missing or just one.
#: `MetricRegistry.gate` therefore returns ``None`` for a value it cannot judge,
#: and the task decides what a quantum of nothing but ``None`` means -- normally
#: by passing ``default=UNKNOWN`` to `worstStatus`.
STATUS_ORDER = ("PASS", "WARN", "FAIL")

#: The verdict for a quantum where nothing could be measured.
#:
#: It sits outside `STATUS_ORDER` because it is not a severity: a detector that
#: could not be measured is not "worse than PASS but better than WARN", it is
#: unassessed. Reporting PASS instead would say the detector is fine on the
#: strength of having looked at nothing (plan section 2.3).
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Thresholds:
    """A WARN/FAIL pair.

    Attributes
    ----------
    warn : `float` or `None`
        Value at which the metric warns; ``None`` disables the WARN check.
    fail : `float` or `None`
        Value at which the metric fails; ``None`` disables the FAIL check.
    """

    warn: float | None = None
    fail: float | None = None


@dataclass(frozen=True)
class GateResult:
    """The verdict on one value of one metric.

    Attributes
    ----------
    metric : `str`
        The metric's name.
    status : `str`
        One of `STATUS_ORDER`.
    value : `float`
        The value that was gated, after any ``useAbsolute`` transform.
    threshold : `float` or `None`
        The threshold that was crossed, or ``None`` when the status is PASS.
    key : `str` or `None`
        The override key whose thresholds were used, or ``None`` when the
        metric's base thresholds applied.
    reason : `str` or `None`
        A human-readable explanation, or ``None`` when the status is PASS.
    """

    metric: str
    status: str
    value: float
    threshold: float | None = None
    key: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class MetricDef:
    """Everything that is true about a metric other than its measured value.

    Attributes
    ----------
    name : `str`
        The metric's name, matching its column in the metrics table.
    units : `str`
        Physical units, e.g. ``"pixels"`` or ``"percent"``. For axis labels.
    reference : `str`
        What external reference the metric is measured against (R1): a
        calibration product, a detector constant, a physical constant, or the
        same measurement from a different epoch. A metric whose reference is its
        own sample is self-referential and measures nothing; stating the
        reference here is what makes that visible at review time.
    higherIsWorse : `bool`, optional
        True when large values indicate a problem. Default is True.
    thresholds : `Thresholds`, optional
        The base WARN/FAIL pair.
    overrides : `dict` [`str`, `Thresholds`], optional
        Per-key thresholds, tried before the base pair. Keys are matched in the
        order the caller supplies them, most specific first: the image-quality
        flag rate uses ``arm:species`` then ``arm``, because several lamp
        species have almost no usable blue-arm lines and a blended threshold
        would be dominated by the species mix rather than by the instrument
        (R7).
    provenance : `str`, optional
        Where the thresholds came from: the visit range and the date they were
        derived (R2). `pfs.drp.qa.metrics.thresholds.formatProvenance` writes
        this sentence for you.
    label : `str`, optional
        Display name used in reason strings. Defaults to ``name``.
    unitSuffix : `str`, optional
        Short suffix appended to values in reason strings, e.g. ``"px"``.
    valueFormat : `str`, optional
        Format spec for the value in reason strings, e.g. ``".2f"``.
    useAbsolute : `bool`, optional
        Gate on ``abs(value)``. A bulk spatial offset is equally bad in either
        direction. Default is False.
    description : `str`, optional
        One line on what the metric means, for the dashboard.
    """

    name: str
    units: str
    reference: str
    higherIsWorse: bool = True
    thresholds: Thresholds = field(default_factory=Thresholds)
    overrides: Mapping[str, Thresholds] = field(default_factory=dict)
    provenance: str = ""
    label: str = ""
    unitSuffix: str = ""
    valueFormat: str = ".3g"
    useAbsolute: bool = False
    description: str = ""

    @property
    def displayName(self) -> str:
        """The name to use in reason strings."""
        return self.label or self.name

    def thresholdsFor(self, keys: Sequence[str] = ()) -> tuple[Thresholds, str | None]:
        """Resolve the thresholds that apply to a given lookup key.

        Parameters
        ----------
        keys : `Sequence` [`str`], optional
            Candidate override keys, most specific first. Empty or unmatched
            keys fall through to the base thresholds.

        Returns
        -------
        thresholds : `Thresholds`
            The thresholds to apply.
        key : `str` or `None`
            The override key that matched, or ``None`` for the base pair.
        """
        for key in keys:
            if key and key in self.overrides:
                return self.overrides[key], key
        return self.thresholds, None

    def gate(self, value: float, keys: Sequence[str] = ()) -> GateResult | None:
        """Judge one value.

        Parameters
        ----------
        value : `float`
            The measured value.
        keys : `Sequence` [`str`], optional
            Candidate override keys, most specific first.

        Returns
        -------
        `GateResult` or `None`
            The verdict, or ``None`` when there is no verdict to give: the
            value is missing (``None`` or NaN), or the metric has no thresholds
            and so is reported but not gated. ``None`` is not a PASS -- an
            unmeasured or ungated metric must not be able to turn a bad quantum
            green, nor claim to have been checked when it was not.

        Notes
        -----
        An infinite value *is* judged, and against a "higher is worse" metric it
        fails. A measurement that blew up is a fault, not a missing measurement,
        and treating it as unmeasured would let it pass silently.
        """
        if value is None:
            return None
        value = float(value)
        if math.isnan(value):
            return None
        if self.useAbsolute:
            value = abs(value)

        thresholds, key = self.thresholdsFor(keys)
        if thresholds.warn is None and thresholds.fail is None:
            return None
        for status, limit in (("FAIL", thresholds.fail), ("WARN", thresholds.warn)):
            if limit is not None and self._crossed(value, limit):
                return GateResult(
                    metric=self.name,
                    status=status,
                    value=value,
                    threshold=limit,
                    key=key,
                    reason=self._reason(value, status, limit),
                )
        return GateResult(metric=self.name, status="PASS", value=value, key=key)

    def _crossed(self, value: float, limit: float) -> bool:
        """Return True when ``value`` is at or beyond ``limit`` in the bad direction.

        Parameters
        ----------
        value : `float`
            The measured value.
        limit : `float`
            The threshold.

        Returns
        -------
        `bool`
            Whether the threshold is crossed.
        """
        return value >= limit if self.higherIsWorse else value <= limit

    def _reason(self, value: float, status: str, limit: float) -> str:
        """Build the human-readable reason for a crossed threshold.

        Parameters
        ----------
        value : `float`
            The measured value.
        status : `str`
            ``"WARN"`` or ``"FAIL"``.
        limit : `float`
            The threshold that was crossed.

        Returns
        -------
        `str`
            e.g. ``"medFWHM=3.60px >= fail threshold 3.5px"``.
        """
        comparison = ">=" if self.higherIsWorse else "<="
        formatted = format(value, self.valueFormat)
        return (
            f"{self.displayName}={formatted}{self.unitSuffix} {comparison} "
            f"{status.lower()} threshold {limit}{self.unitSuffix}"
        )

    def withThresholds(
        self,
        warn: float | None = None,
        fail: float | None = None,
        overrides: Mapping[str, Thresholds] | None = None,
        provenance: str | None = None,
    ) -> "MetricDef":
        """Return a copy carrying different thresholds.

        Tasks use this to fold their config -- which operators can override on
        the command line -- into the static definition, so that the registry
        always gates with the numbers actually in force.

        Parameters
        ----------
        warn : `float`, optional
            New base WARN threshold.
        fail : `float`, optional
            New base FAIL threshold.
        overrides : `Mapping` [`str`, `Thresholds`], optional
            New per-key thresholds, replacing any existing ones.
        provenance : `str`, optional
            New provenance sentence.

        Returns
        -------
        `MetricDef`
            The copy.
        """
        changes: dict = {"thresholds": Thresholds(warn=warn, fail=fail)}
        if overrides is not None:
            changes["overrides"] = dict(overrides)
        if provenance is not None:
            changes["provenance"] = provenance
        return replace(self, **changes)


class MetricRegistry:
    """An ordered collection of `MetricDef`, and the one gating entry point.

    Parameters
    ----------
    metrics : `Iterable` [`MetricDef`], optional
        Definitions to register up front.
    """

    def __init__(self, metrics: Iterable[MetricDef] = ()) -> None:
        self._metrics: dict[str, MetricDef] = {}
        for metric in metrics:
            self.register(metric)

    def register(self, metric: MetricDef) -> MetricDef:
        """Add a definition.

        Parameters
        ----------
        metric : `MetricDef`
            The definition to add.

        Returns
        -------
        `MetricDef`
            The definition, for convenience.

        Raises
        ------
        ValueError
            If a metric of the same name is already registered. Two definitions
            of one metric is exactly the drift this module exists to prevent.
        """
        if metric.name in self._metrics:
            raise ValueError(f"Metric {metric.name!r} is already registered")
        self._metrics[metric.name] = metric
        return metric

    def __contains__(self, name: object) -> bool:
        """Return whether a metric of this name is registered."""
        return name in self._metrics

    def __getitem__(self, name: str) -> MetricDef:
        """Return the named definition, raising `KeyError` when absent."""
        return self._metrics[name]

    def __iter__(self) -> Iterator[MetricDef]:
        """Iterate over the definitions in registration order."""
        return iter(self._metrics.values())

    def __len__(self) -> int:
        """Return the number of registered metrics."""
        return len(self._metrics)

    @property
    def names(self) -> tuple[str, ...]:
        """The registered metric names, in registration order."""
        return tuple(self._metrics)

    def get(self, name: str, default: MetricDef | None = None) -> MetricDef | None:
        """Return the named definition, or ``default`` when it is absent.

        Parameters
        ----------
        name : `str`
            Metric name.
        default : `MetricDef`, optional
            Value to return when the metric is not registered.

        Returns
        -------
        `MetricDef` or `None`
            The definition, or ``default``.
        """
        return self._metrics.get(name, default)

    def gate(self, name: str, value: float, keys: Sequence[str] = ()) -> GateResult | None:
        """Judge one value of one metric.

        Parameters
        ----------
        name : `str`
            Metric name.
        value : `float`
            The measured value.
        keys : `Sequence` [`str`], optional
            Candidate override keys, most specific first.

        Returns
        -------
        `GateResult` or `None`
            The verdict, or ``None`` when the value could not be judged.

        Raises
        ------
        KeyError
            If the metric is not registered. Gating an unregistered metric is a
            programming error, not a missing measurement, and silently
            returning PASS for it would hide a whole metric.
        """
        return self._metrics[name].gate(value, keys)


def worstStatus(statuses: Iterable[str | GateResult | None], default: str = "PASS") -> str:
    """Reduce several verdicts to the worst one.

    Parameters
    ----------
    statuses : `Iterable`
        Status strings, `GateResult` objects, or ``None`` for metrics that were
        not judged. ``None`` entries are skipped.
    default : `str`, optional
        Result when nothing was judged -- every entry was ``None``. Defaults to
        ``"PASS"`` for callers that want the old behaviour, but a task whose
        metrics were all unmeasurable should pass `UNKNOWN`: PASS would claim
        the detector is fine on the strength of having measured nothing. The
        default is returned as given and is not required to be in
        `STATUS_ORDER`.

    Returns
    -------
    `str`
        The worst status present, or ``default``.

    Raises
    ------
    ValueError
        If a status is not one of `STATUS_ORDER`.
    """
    worst = None
    for entry in statuses:
        if entry is None:
            continue
        status = entry.status if isinstance(entry, GateResult) else entry
        if status not in STATUS_ORDER:
            raise ValueError(f"Unknown status {status!r}; expected one of {', '.join(STATUS_ORDER)}")
        if worst is None or STATUS_ORDER.index(status) > STATUS_ORDER.index(worst):
            worst = status
    return default if worst is None else worst
