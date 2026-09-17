"""Loader for the golden visit set.

The golden visit set is a small, fixed list of visits with known verdicts,
checked into the repository as ``tests/data/goldenVisits.yaml``. It is the
external reference that makes a threshold defensible: a metric that flags a
``known_good`` visit, or passes a ``known_bad`` one, does not merge.

See ``doc/qa-rebuild-plan.md`` section 1.1 for the rationale and the file's own
header comment for the entry schema.

This module imports neither the LSST stack nor the Butler, so it runs in CI.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "GoldenVisit",
    "GoldenVisitSet",
    "defaultGoldenVisitsPath",
    "loadGoldenVisits",
]

#: Verdicts an entry may expect. Mirrors ``qaStatus`` in the QA tasks.
VALID_EXPECTATIONS = ("PASS", "WARN", "FAIL")

#: The schema version this loader understands.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GoldenVisit:
    """One entry in the golden visit set.

    An entry covers either a single visit or an inclusive range of visits, and
    optionally narrows to particular arms, spectrographs or a sequence type. An
    omitted selector means "every value": an entry with no ``arms`` applies to
    every arm.

    Attributes
    ----------
    visits : `tuple` [`int`]
        The visits this entry covers, expanded from ``visit``/``visitRange``.
    expect : `str`
        The expected verdict, one of ``PASS``, ``WARN`` or ``FAIL``.
    arms : `tuple` [`str`]
        Arms the entry applies to; empty means all arms.
    spectrographs : `tuple` [`int`]
        Spectrographs the entry applies to; empty means all spectrographs.
    seqType : `str` or `None`
        The ``W_SEQNAM`` value this entry applies to, if restricted.
    metric : `str` or `None`
        Name of the metric expected to identify the fault. Only meaningful for
        ``known_bad`` entries, where failing for the *right* reason is part of
        the acceptance criterion.
    reason : `str` or `None`
        Why this verdict is expected.
    note : `str` or `None`
        Free-form context.
    placeholder : `bool`
        True when the entry is a template awaiting a real visit number.
    """

    visits: tuple[int, ...]
    expect: str
    arms: tuple[str, ...] = ()
    spectrographs: tuple[int, ...] = ()
    seqType: str | None = None
    metric: str | None = None
    reason: str | None = None
    note: str | None = None
    placeholder: bool = False

    def matches(
        self,
        visit: int,
        arm: str | None = None,
        spectrograph: int | None = None,
        seqType: str | None = None,
    ) -> bool:
        """Return True when this entry covers the given detector-visit.

        Parameters
        ----------
        visit : `int`
            Visit number.
        arm : `str`, optional
            Arm name. When ``None`` the arm selector is not applied.
        spectrograph : `int`, optional
            Spectrograph number. When ``None`` the selector is not applied.
        seqType : `str`, optional
            ``W_SEQNAM`` value. When ``None`` the selector is not applied.

        Returns
        -------
        `bool`
            True if the entry applies.
        """
        if visit not in self.visits:
            return False
        if arm is not None and self.arms and arm not in self.arms:
            return False
        if spectrograph is not None and self.spectrographs and spectrograph not in self.spectrographs:
            return False
        return not (seqType is not None and self.seqType is not None and seqType != self.seqType)


@dataclass(frozen=True)
class GoldenVisitSet:
    """The parsed contents of ``goldenVisits.yaml``.

    Attributes
    ----------
    knownGood : `tuple` [`GoldenVisit`]
        Entries expected to pass every metric.
    knownBad : `tuple` [`GoldenVisit`]
        Entries expected to WARN or FAIL, for a stated reason.
    path : `pathlib.Path` or `None`
        Where the set was loaded from, for error messages and provenance.
    """

    knownGood: tuple[GoldenVisit, ...] = ()
    knownBad: tuple[GoldenVisit, ...] = ()
    path: Path | None = field(default=None, compare=False)

    def __iter__(self) -> Iterator[GoldenVisit]:
        """Iterate over every entry, good then bad."""
        yield from self.knownGood
        yield from self.knownBad

    def __len__(self) -> int:
        """Return the total number of entries."""
        return len(self.knownGood) + len(self.knownBad)

    @property
    def visits(self) -> tuple[int, ...]:
        """Every visit number mentioned by the set, sorted and deduplicated."""
        return tuple(sorted({visit for entry in self for visit in entry.visits}))

    @property
    def goodVisits(self) -> tuple[int, ...]:
        """Every ``known_good`` visit number, sorted and deduplicated."""
        return tuple(sorted({visit for entry in self.knownGood for visit in entry.visits}))

    def find(
        self,
        visit: int,
        arm: str | None = None,
        spectrograph: int | None = None,
        seqType: str | None = None,
    ) -> tuple[GoldenVisit, ...]:
        """Return every entry covering the given detector-visit.

        Parameters
        ----------
        visit : `int`
            Visit number.
        arm : `str`, optional
            Arm name.
        spectrograph : `int`, optional
            Spectrograph number.
        seqType : `str`, optional
            ``W_SEQNAM`` value.

        Returns
        -------
        `tuple` [`GoldenVisit`]
            Matching entries, ``known_good`` first. Empty when the visit is not
            in the set, which means "no expectation recorded", never "PASS".
        """
        return tuple(entry for entry in self if entry.matches(visit, arm, spectrograph, seqType))

    def expectationFor(
        self,
        visit: int,
        arm: str | None = None,
        spectrograph: int | None = None,
        seqType: str | None = None,
    ) -> str | None:
        """Return the worst expected verdict for a detector-visit.

        Parameters
        ----------
        visit : `int`
            Visit number.
        arm : `str`, optional
            Arm name.
        spectrograph : `int`, optional
            Spectrograph number.
        seqType : `str`, optional
            ``W_SEQNAM`` value.

        Returns
        -------
        `str` or `None`
            ``PASS``, ``WARN`` or ``FAIL``, or ``None`` when the set records no
            expectation for this detector-visit.
        """
        matches = self.find(visit, arm, spectrograph, seqType)
        if not matches:
            return None
        return max((entry.expect for entry in matches), key=VALID_EXPECTATIONS.index)


def defaultGoldenVisitsPath() -> Path:
    """Return the in-repository path of the golden visit set.

    Returns
    -------
    `pathlib.Path`
        ``tests/data/goldenVisits.yaml`` relative to the repository root. The
        file ships with the source checkout rather than with an installed
        wheel, so callers working from an installed package must pass an
        explicit path.
    """
    return Path(__file__).resolve().parents[5] / "tests" / "data" / "goldenVisits.yaml"


def loadGoldenVisits(
    path: Path | str | None = None,
    includePlaceholders: bool = False,
) -> GoldenVisitSet:
    """Load and validate the golden visit set.

    Parameters
    ----------
    path : `pathlib.Path` or `str`, optional
        The YAML file to read. Defaults to `defaultGoldenVisitsPath`.
    includePlaceholders : `bool`, optional
        Include entries marked ``placeholder: true``. Default is False, so that
        an entry that has not yet been filled in with a real visit number can
        never silently validate a threshold.

    Returns
    -------
    `GoldenVisitSet`
        The parsed set.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the schema version is unrecognised, a section is not a list, or an
        entry is malformed. Validation is strict on purpose: a golden set that
        silently drops a malformed entry stops being an external reference.
    """
    path = Path(path) if path is not None else defaultGoldenVisitsPath()
    if not path.exists():
        raise FileNotFoundError(f"Golden visit set not found: {path}")

    with path.open() as fd:
        doc = yaml.safe_load(fd)

    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(doc).__name__}")

    version = doc.get("version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema version {version!r}, expected {SCHEMA_VERSION}")

    sections = {}
    for section, defaultExpect in (("known_good", "PASS"), ("known_bad", None)):
        raw = doc.get(section) or []
        if not isinstance(raw, list):
            raise ValueError(f"{path}: '{section}' must be a list, got {type(raw).__name__}")
        entries = []
        for index, item in enumerate(raw):
            entry = _parseEntry(item, defaultExpect, f"{path}: {section}[{index}]")
            if entry.placeholder and not includePlaceholders:
                continue
            entries.append(entry)
        sections[section] = tuple(entries)

    return GoldenVisitSet(
        knownGood=sections["known_good"],
        knownBad=sections["known_bad"],
        path=path,
    )


def _parseEntry(item: Any, defaultExpect: str | None, where: str) -> GoldenVisit:
    """Parse and validate one entry.

    Parameters
    ----------
    item : `Any`
        The raw YAML value for the entry.
    defaultExpect : `str` or `None`
        Verdict to assume when the entry does not state one. ``None`` makes
        ``expect`` mandatory, which is the case for ``known_bad``.
    where : `str`
        Human-readable location, used in error messages.

    Returns
    -------
    `GoldenVisit`
        The parsed entry.

    Raises
    ------
    ValueError
        If the entry is malformed.
    """
    if not isinstance(item, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(item).__name__}")

    placeholder = bool(item.get("placeholder", False))
    visits = _parseVisits(item, where, placeholder)

    expect = item.get("expect", defaultExpect)
    if expect is None:
        raise ValueError(f"{where}: 'expect' is required (one of {', '.join(VALID_EXPECTATIONS)})")
    expect = str(expect).upper()
    if expect not in VALID_EXPECTATIONS:
        raise ValueError(
            f"{where}: invalid expect {expect!r}, must be one of {', '.join(VALID_EXPECTATIONS)}"
        )

    arms = _parseSequence(item.get("arms"), str, "arms", where)
    spectrographs = _parseSequence(item.get("spectrographs"), int, "spectrographs", where)

    return GoldenVisit(
        visits=visits,
        expect=expect,
        arms=arms,
        spectrographs=spectrographs,
        seqType=_optionalStr(item.get("seqType")),
        metric=_optionalStr(item.get("metric")),
        reason=_optionalStr(item.get("reason")),
        note=_optionalStr(item.get("note")),
        placeholder=placeholder,
    )


def _parseVisits(item: dict, where: str, placeholder: bool) -> tuple[int, ...]:
    """Expand an entry's ``visit`` or ``visitRange`` into visit numbers.

    Parameters
    ----------
    item : `dict`
        The raw entry.
    where : `str`
        Human-readable location, used in error messages.
    placeholder : `bool`
        Whether the entry is a placeholder, in which case a null visit is
        allowed and expands to no visits.

    Returns
    -------
    `tuple` [`int`]
        The visits covered, in ascending order.

    Raises
    ------
    ValueError
        If neither or both of ``visit`` and ``visitRange`` are given, or if a
        value is not an integer, or if the range is inverted.
    """
    visit = item.get("visit")
    visitRange = item.get("visitRange")

    if visit is not None and visitRange is not None:
        raise ValueError(f"{where}: give either 'visit' or 'visitRange', not both")

    if visit is None and visitRange is None:
        if placeholder:
            return ()
        raise ValueError(f"{where}: one of 'visit' or 'visitRange' is required")

    if visitRange is not None:
        if not isinstance(visitRange, Sequence) or isinstance(visitRange, str) or len(visitRange) != 2:
            raise ValueError(f"{where}: 'visitRange' must be a two-element list [first, last]")
        first, last = (_asInt(value, "visitRange", where) for value in visitRange)
        if last < first:
            raise ValueError(f"{where}: 'visitRange' is inverted: [{first}, {last}]")
        return tuple(range(first, last + 1))

    return (_asInt(visit, "visit", where),)


def _asInt(value: Any, name: str, where: str) -> int:
    """Coerce a YAML scalar to `int`, or raise.

    Parameters
    ----------
    value : `Any`
        The raw value.
    name : `str`
        Field name, used in error messages.
    where : `str`
        Human-readable location, used in error messages.

    Returns
    -------
    `int`
        The coerced value.

    Raises
    ------
    ValueError
        If the value is not an integer. ``bool`` is rejected explicitly because
        it is a subclass of `int` and would otherwise pass silently.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: '{name}' must be an integer, got {value!r}")
    return value


def _parseSequence(value: Any, itemType: type, name: str, where: str) -> tuple:
    """Parse an optional list selector.

    Parameters
    ----------
    value : `Any`
        The raw value; ``None`` yields an empty tuple, meaning "every value".
    itemType : `type`
        ``str`` or ``int``.
    name : `str`
        Field name, used in error messages.
    where : `str`
        Human-readable location, used in error messages.

    Returns
    -------
    `tuple`
        The parsed selector.

    Raises
    ------
    ValueError
        If the value is not a list of the expected type.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{where}: '{name}' must be a list, got {type(value).__name__}")
    if itemType is int:
        return tuple(_asInt(item, name, where) for item in value)
    return tuple(str(item) for item in value)


def _optionalStr(value: Any) -> str | None:
    """Return ``value`` as a string, or ``None`` when it is absent.

    Parameters
    ----------
    value : `Any`
        The raw value.

    Returns
    -------
    `str` or `None`
        The stringified value, or ``None``.
    """
    return None if value is None else str(value)
