"""Loader for the validation visit set.

The validation visit set is a small, fixed list of visits with known verdicts,
shipped with the package as ``data/validationVisits.yaml``. It is the external
reference that makes a threshold defensible: a metric that flags a
``known_good`` visit, or passes a ``known_bad`` one, does not merge.

See ``docs/qa-principles.md`` for how it is used, and the file's own header
comment for the entry schema.

This module imports neither the LSST stack nor the Butler, so it runs in CI.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = [
    "ValidationVisit",
    "ValidationVisitSet",
    "defaultValidationVisitsPath",
    "formatTables",
    "loadValidationVisits",
    "matchRows",
    "selectRows",
    "unmatchedEntries",
    "visitExpression",
]

#: Verdicts an entry may expect, in increasing severity. Mirrors ``qaStatus``.
VALID_EXPECTATIONS = ("PASS", "WARN", "FAIL")

#: The schema version this loader understands.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ValidationVisit:
    """One entry in the validation visit set.

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
    unconfirmed : `bool`
        True when the verdict is suspected but not established: a real visit,
        noted as suspect in the run summary, whose metrics nobody has checked. It is loaded
        and reported like any other entry but never decides a verdict, because
        a guess that fails the build is worse than no entry.
    placeholder : `bool`
        True when the entry awaits a real visit number. Placeholders are dropped
        by `loadValidationVisits` unless asked for.
    """

    visits: tuple[int, ...]
    expect: str
    arms: tuple[str, ...] = ()
    spectrographs: tuple[int, ...] = ()
    seqType: str | None = None
    metric: str | None = None
    reason: str | None = None
    note: str | None = None
    unconfirmed: bool = False
    placeholder: bool = False

    def matches(
        self,
        visit: int,
        arm: str | None = None,
        spectrograph: int | None = None,
        seqType: str | None = None,
    ) -> bool:
        """Return True when this entry covers the given detector-visit.

        A selector passed as ``None`` is not applied: it means "unknown", not
        "no match".

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
class ValidationVisitSet:
    """The parsed contents of ``validationVisits.yaml``.

    Attributes
    ----------
    knownGood : `tuple` [`ValidationVisit`]
        Entries expected to pass every metric.
    knownBad : `tuple` [`ValidationVisit`]
        Entries expected to WARN or FAIL, for a stated reason.
    path : `pathlib.Path` or `None`
        Where the set was loaded from, for error messages and provenance.
    """

    knownGood: tuple[ValidationVisit, ...] = ()
    knownBad: tuple[ValidationVisit, ...] = ()
    path: Path | None = field(default=None, compare=False)

    def __iter__(self) -> Iterator[ValidationVisit]:
        """Iterate over every entry, good then bad."""
        yield from self.knownGood
        yield from self.knownBad

    def __len__(self) -> int:
        """Return the total number of entries."""
        return len(self.knownGood) + len(self.knownBad)

    @property
    def visits(self) -> tuple[int, ...]:
        """Every visit number in the set, sorted and deduplicated."""
        return tuple(sorted({visit for entry in self for visit in entry.visits}))

    @property
    def goodVisits(self) -> tuple[int, ...]:
        """Every ``known_good`` visit number, sorted and deduplicated."""
        return tuple(sorted({visit for entry in self.knownGood for visit in entry.visits}))

    @property
    def confirmedBad(self) -> tuple[ValidationVisit, ...]:
        """The ``known_bad`` entries whose verdict is established.

        These are the ones a threshold must separate.
        """
        return tuple(entry for entry in self.knownBad if not entry.unconfirmed)

    @property
    def unconfirmedBad(self) -> tuple[ValidationVisit, ...]:
        """The ``known_bad`` entries whose verdict is only suspected."""
        return tuple(entry for entry in self.knownBad if entry.unconfirmed)

    def find(
        self,
        visit: int,
        arm: str | None = None,
        spectrograph: int | None = None,
        seqType: str | None = None,
    ) -> tuple[ValidationVisit, ...]:
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
        `tuple` [`ValidationVisit`]
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


def matchRows(frame, entries: Iterable[ValidationVisit]) -> np.ndarray:
    """Return a boolean mask of the metrics rows covered by any of ``entries``.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        Metrics rows with a ``visit`` column. ``arm``, ``spectrograph`` and
        ``seqName`` (the ``W_SEQNAM`` value) are used when present; a missing
        column, or a null value in one, does not restrict the match.
    entries : iterable of `ValidationVisit`
        The entries to match.

    Returns
    -------
    `numpy.ndarray` [`bool`]
        One element per row, in order; independent of the frame's index.

    Raises
    ------
    KeyError
        If ``frame`` has no ``visit`` column.
    """
    if "visit" not in frame.columns:
        raise KeyError("metrics table has no 'visit' column; cannot match it against validation visits")

    def selector(column: str, values: Sequence) -> np.ndarray:
        if not values or column not in frame.columns:
            return np.ones(len(frame), dtype=bool)
        series = frame[column]
        return (series.isna() | series.isin(values)).to_numpy()

    mask = np.zeros(len(frame), dtype=bool)
    for entry in entries:
        mask |= (
            frame["visit"].isin(entry.visits).to_numpy()
            & selector("arm", entry.arms)
            & selector("spectrograph", entry.spectrographs)
            & selector("seqName", (entry.seqType,) if entry.seqType is not None else ())
        )
    return mask


def selectRows(frame, entries: Iterable[ValidationVisit]):
    """Select the metrics rows covered by any of ``entries``.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        Metrics rows; see `matchRows`.
    entries : iterable of `ValidationVisit`
        The entries to match.

    Returns
    -------
    `pandas.DataFrame`
        The matching rows, in their original order.
    """
    return frame[matchRows(frame, entries)]


def unmatchedEntries(frame, visitSet: ValidationVisitSet) -> tuple[ValidationVisit, ...]:
    """Return the entries that match no row of a metrics table.

    An entry can match nothing because its visits were not processed, or
    because a selector is wrong: a ``seqType`` must equal ``W_SEQNAM``
    exactly, typos included. Either way its check silently does not run.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        Metrics rows; see `matchRows`.
    visitSet : `ValidationVisitSet`
        The validation visit set.

    Returns
    -------
    `tuple` [`ValidationVisit`]
        The entries with no matching row, ``known_good`` first.
    """
    return tuple(entry for entry in visitSet if not matchRows(frame, [entry]).any())


def visitExpression(visits: Iterable[int]) -> str:
    """Return a Butler query expression selecting ``visits``.

    Consecutive visits collapse into ``first..last`` ranges, so the expression
    stays short enough to paste.

    Parameters
    ----------
    visits : iterable of `int`
        Visit numbers, in any order; duplicates are ignored.

    Returns
    -------
    `str`
        E.g. ``"visit IN (133025..133055, 149217)"``.

    Raises
    ------
    ValueError
        If there are no visits.
    """
    ordered = sorted(set(visits))
    if not ordered:
        raise ValueError("No visits")
    ranges = [[ordered[0], ordered[0]]]
    for visit in ordered[1:]:
        if visit == ranges[-1][1] + 1:
            ranges[-1][1] = visit
        else:
            ranges.append([visit, visit])
    terms = [str(first) if first == last else f"{first}..{last}" for first, last in ranges]
    return f"visit IN ({', '.join(terms)})"


def formatTables(visitSet: ValidationVisitSet) -> str:
    """Return the set as Markdown tables, for ``docs/validation-visits.md``.

    Parameters
    ----------
    visitSet : `ValidationVisitSet`
        The set, loaded with ``includePlaceholders=True`` so that the gaps show.

    Returns
    -------
    `str`
        One table per kind of entry: known good, known bad, unconfirmed and
        placeholders, each preceded by a heading.
    """
    sections = (
        ("Known good", visitSet.knownGood, "note"),
        ("Known bad", tuple(e for e in visitSet.confirmedBad if not e.placeholder), "reason"),
        ("Unconfirmed", visitSet.unconfirmedBad, "reason"),
        ("Placeholders", tuple(e for e in visitSet.knownBad if e.placeholder), "reason"),
    )
    blocks = []
    for title, entries, textField in sections:
        if not entries:
            continue
        lines = [
            f"### {title}",
            "",
            f"| Visits | Arms | Spectrographs | Sequence | Expect | Metric | {textField.capitalize()} |",
            "|---|---|---|---|---|---|---|",
        ]
        for entry in entries:
            if entry.visits:
                first, last = entry.visits[0], entry.visits[-1]
                visits = str(first) if first == last else f"{first}–{last}"
            else:
                visits = "*to find*"
            cells = (
                visits,
                ", ".join(entry.arms) or "all",
                ", ".join(str(s) for s in entry.spectrographs) or "all",
                entry.seqType or "any",
                entry.expect,
                f"`{entry.metric}`" if entry.metric else "",
                " ".join((getattr(entry, textField) or "").split()),
            )
            lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def defaultValidationVisitsPath() -> Path:
    """Return the path of the validation visit set shipped with the package.

    Returns
    -------
    `pathlib.Path`
        ``pfs/drp/qa/metrics/data/validationVisits.yaml``.
    """
    return Path(str(files("pfs.drp.qa.metrics") / "data" / "validationVisits.yaml"))


def loadValidationVisits(
    path: Path | str | None = None,
    includePlaceholders: bool = False,
) -> ValidationVisitSet:
    """Load and validate the validation visit set.

    Parameters
    ----------
    path : `pathlib.Path` or `str`, optional
        The YAML file to read. Defaults to `defaultValidationVisitsPath`.
    includePlaceholders : `bool`, optional
        Include entries marked ``placeholder: true``. Default is False, so that
        an unfilled entry can never validate a threshold.

    Returns
    -------
    `ValidationVisitSet`
        The parsed set.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the schema version is unrecognised, a section is not a list, or an
        entry is malformed. Validation is strict on purpose: a set that quietly
        drops a malformed entry stops being a reference.
    """
    path = Path(path) if path is not None else defaultValidationVisitsPath()
    if not path.exists():
        raise FileNotFoundError(f"Validation visit set not found: {path}")

    with path.open() as fd:
        doc = yaml.safe_load(fd)

    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(doc).__name__}")

    version = doc.get("version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema version {version!r}, expected {SCHEMA_VERSION}")

    unknown = set(doc) - {"version", "known_good", "known_bad"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys: {', '.join(sorted(unknown))}")

    sections = {}
    for section, defaultExpect in (("known_good", "PASS"), ("known_bad", None)):
        raw = doc.get(section) or []
        if not isinstance(raw, list):
            raise ValueError(f"{path}: '{section}' must be a list, got {type(raw).__name__}")
        entries = []
        for index, item in enumerate(raw):
            entry = _parseEntry(item, defaultExpect, f"{path}: {section}[{index}]")
            if section == "known_good" and entry.expect != "PASS":
                raise ValueError(f"{path}: {section}[{index}]: a known_good entry must expect PASS")
            if entry.placeholder and not includePlaceholders:
                continue
            entries.append(entry)
        sections[section] = tuple(entries)

    return ValidationVisitSet(knownGood=sections["known_good"], knownBad=sections["known_bad"], path=path)


#: Keys an entry may carry. Anything else is a typo, and a typo in a selector
#: would silently widen the entry to every value.
_ENTRY_KEYS = frozenset(
    {
        "visit",
        "visitRange",
        "arms",
        "spectrographs",
        "seqType",
        "expect",
        "metric",
        "reason",
        "note",
        "unconfirmed",
        "placeholder",
    }
)


def _parseEntry(item: Any, defaultExpect: str | None, where: str) -> ValidationVisit:
    """Parse and validate one entry.

    Parameters
    ----------
    item : `Any`
        The raw YAML value for the entry.
    defaultExpect : `str` or `None`
        Verdict to assume when the entry states none. ``None`` makes ``expect``
        mandatory, as it is for ``known_bad``.
    where : `str`
        Human-readable location, used in error messages.

    Returns
    -------
    `ValidationVisit`
        The parsed entry.

    Raises
    ------
    ValueError
        If the entry is malformed.
    """
    if not isinstance(item, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(item).__name__}")

    unknown = set(item) - _ENTRY_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys: {', '.join(sorted(unknown))}")

    placeholder = _asBool(item.get("placeholder", False), "placeholder", where)
    visits = _parseVisits(item, where, placeholder)

    expect = item.get("expect", defaultExpect)
    if expect is None:
        raise ValueError(f"{where}: 'expect' is required (one of {', '.join(VALID_EXPECTATIONS)})")
    expect = str(expect).upper()
    if expect not in VALID_EXPECTATIONS:
        raise ValueError(
            f"{where}: invalid expect {expect!r}, must be one of {', '.join(VALID_EXPECTATIONS)}"
        )

    return ValidationVisit(
        visits=visits,
        expect=expect,
        arms=_parseSequence(item.get("arms"), str, "arms", where),
        spectrographs=_parseSequence(item.get("spectrographs"), int, "spectrographs", where),
        seqType=_optionalStr(item.get("seqType")),
        metric=_optionalStr(item.get("metric")),
        reason=_optionalStr(item.get("reason")),
        note=_optionalStr(item.get("note")),
        unconfirmed=_asBool(item.get("unconfirmed", False), "unconfirmed", where),
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
        If neither or both of ``visit`` and ``visitRange`` are given, a value
        is not an integer, or the range is inverted.
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


def _asBool(value: Any, name: str, where: str) -> bool:
    """Return a YAML boolean, or raise.

    Truthiness is not enough: ``placeholder: "false"`` is a non-empty string,
    and `bool` would read it as True and drop the entry.

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
    `bool`
        The value.

    Raises
    ------
    ValueError
        If the value is not a YAML boolean.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{where}: '{name}' must be true or false, got {value!r}")
    return value


def _asInt(value: Any, name: str, where: str) -> int:
    """Return a YAML integer, or raise.

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
        The value.

    Raises
    ------
    ValueError
        If the value is not an integer. `bool` is rejected explicitly, being a
        subclass of `int`.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: '{name}' must be an integer, got {value!r}")
    return value


def _parseSequence(value: Any, itemType: type, name: str, where: str) -> tuple:
    """Parse an optional list selector.

    Parameters
    ----------
    value : `Any`
        The raw value; ``None`` gives an empty tuple, meaning "every value".
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
    """Return ``value`` as a string, or ``None`` when it is absent."""
    return None if value is None else str(value)


def main(argv: Sequence[str] | None = None) -> None:
    """Print the visit set for a command line or for the docs.

    ``expression`` prints the Butler query expression for every visit in the
    set, to paste into ``pipetask -d``; ``tables`` prints the Markdown tables
    of ``docs/validation-visits.md``.

    Parameters
    ----------
    argv : sequence of `str`, optional
        Command-line arguments; defaults to `sys.argv`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pfs.drp.qa.metrics.validationVisits",
        description="Print the validation visit set.",
    )
    parser.add_argument("what", choices=("expression", "tables"))
    parser.add_argument("--path", default=None, help="Visit set YAML (default: the one in the package).")
    args = parser.parse_args(argv)
    if args.what == "expression":
        print(visitExpression(loadValidationVisits(args.path).visits))
    else:
        print(formatTables(loadValidationVisits(args.path, includePlaceholders=True)), end="")


if __name__ == "__main__":
    main()
