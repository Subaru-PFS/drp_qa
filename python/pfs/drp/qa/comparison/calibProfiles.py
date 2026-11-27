"""Report on a candidate ``fiberProfiles`` calibration before it is certified.

Each candidate is a collection holding ``fiberProfiles`` (a build's output, or the inputs as they
are). ``fiberProfilesQa`` measures each (``drpQa.py calib profiles run``), and this module turns the
stored tables into a report for the person about to certify: what to do, the fibers over the limit
(one row per fiber, an arm per column) and each detector's numbers, with the change from the
calibration in force. Stack-free: DataFrames in, text out.

A profile is **over the limit** when its wing fraction is more than
`pfs.drp.qa.metrics.fiberProfiles.WING_EXCESS_LIMIT` above the measured fibers beside it on the
slit. A profile is either **measured** or a **copy** (the build filled it in with the mean of nearby
fibers' profiles): the two are reported apart. A profile that goes below
`pfs.drp.qa.metrics.fiberProfiles.PROFILE_MINIMUM_LIMIT` of its peak in any swath is listed too
(`negativeProfiles`): the build had nothing to fit there, and what is stored is not a shape.
"""

from collections.abc import Mapping

import pandas as pd

from pfs.drp.qa.metrics.fiberProfiles import PROFILE_MINIMUM_LIMIT, WING_EXCESS_LIMIT

__all__ = [
    "DETECTOR_COLUMNS",
    "buildProfileReport",
    "calibrationSources",
    "detectorTable",
    "fibersOverLimit",
    "negativeProfiles",
    "overLimit",
    "whatToDo",
]

#: Arms in the order they are shown.
_ARMS = "brnm"

#: Columns of `detectorTable`, in order.
DETECTOR_COLUMNS = (
    "detector",
    "calibVisit0",
    "nProfiles",
    "nReplaced",
    "nMissing",
    "nUnmeasured",
    "medProfileFwhm",
    "medWingFrac",
    "wingFracChange",
    "fwhmChange",
    "nOverMeasured",
    "nOverCopies",
)


def _withDetector(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``table`` with a ``detector`` column (``r4``) and an ordering key."""
    table = table.copy()
    table["detector"] = table["arm"].astype(str) + table["spectrograph"].astype(int).astype(str)
    table["_order"] = table["arm"].astype(str).map(_ARMS.find) * 10 + table["spectrograph"].astype(int)
    return table


def overLimit(fibers: pd.DataFrame, limit: float = WING_EXCESS_LIMIT) -> pd.DataFrame:
    """Return the profiles whose wing excess is above ``limit``, one row per fiber and arm.

    Parameters
    ----------
    fibers : `pandas.DataFrame`
        ``fiberProfilesQaFibers`` of one calibration, any number of detectors.
    limit : `float`
        Wing excess above which a profile is over the limit.

    Returns
    -------
    `pandas.DataFrame`
        ``detector``, ``arm``, ``spectrograph``, ``fiberId``, ``wingExcess``, ``replaced`` and
        ``donors``, by fiber and then by arm.
    """
    columns = ["detector", "arm", "spectrograph", "fiberId", "wingExcess", "replaced", "donors"]
    if fibers.empty:
        return pd.DataFrame(columns=columns)
    table = _withDetector(fibers)
    table = table[table["wingExcess"] > limit].sort_values(["fiberId", "_order"])
    return table[columns].reset_index(drop=True)


def fibersOverLimit(
    fibers: pd.DataFrame, previous: pd.DataFrame | None = None, limit: float = WING_EXCESS_LIMIT
) -> pd.DataFrame:
    """Return the fibers over the limit in any arm, one row per fiber, an arm per column.

    A fiber has a profile on each arm of its spectrograph, so a fault of the fiber shows in several
    arms at once: one row per fiber says so, where a row per profile repeats the fiber.

    Parameters
    ----------
    fibers : `pandas.DataFrame`
        ``fiberProfilesQaFibers`` of the candidate.
    previous : `pandas.DataFrame`, optional
        The same of the calibration in force.
    limit : `float`
        Wing excess above which a profile is over the limit.

    Returns
    -------
    `pandas.DataFrame`
        ``spectrograph``, ``fiberId``, one column per arm the calibration holds (the wing excess
        where the profile is over the limit, NaN where it is not), ``copy`` (the profile is a copy
        in an arm it is over the limit in), ``donors`` (the fibers it is the mean of) and, with
        ``previous``, ``new`` (it is over the limit in an arm in which the calibration in force is
        not). Ordered by fiber.
    """
    arms = [arm for arm in _ARMS if not fibers.empty and arm in set(fibers["arm"].astype(str))]
    columns = ["spectrograph", "fiberId", *arms, "copy", "donors", *(["new"] if previous is not None else [])]
    over = overLimit(fibers, limit)
    if over.empty:
        return pd.DataFrame(columns=columns)
    keys = ["spectrograph", "fiberId"]
    wide = over.set_index([*keys, "arm"])["wingExcess"].unstack("arm").reindex(columns=arms)
    grouped = over.groupby(keys)
    wide["copy"] = grouped["replaced"].agg(lambda values: bool(values.astype(bool).any()))
    wide["donors"] = grouped["donors"].agg(lambda values: next((str(v) for v in values if str(v)), ""))
    if previous is not None:
        before = overLimit(previous, limit)
        was = set(zip(before["arm"], before["spectrograph"], before["fiberId"], strict=True))
        isNew = [
            key not in was for key in zip(over["arm"], over["spectrograph"], over["fiberId"], strict=True)
        ]
        wide["new"] = over.assign(isNew=isNew).groupby(keys)["isNew"].any()
    return wide.reset_index().sort_values("fiberId").reset_index(drop=True)[columns]


def negativeProfiles(fibers: pd.DataFrame, limit: float = PROFILE_MINIMUM_LIMIT) -> pd.DataFrame:
    """Return the profiles whose lowest value is below ``limit`` of their peak, one row per fiber and arm.

    Parameters
    ----------
    fibers : `pandas.DataFrame`
        ``fiberProfilesQaFibers`` of a calibration.
    limit : `float`
        Lowest value, as a fraction of the profile's peak, below which a profile is listed.

    Returns
    -------
    `pandas.DataFrame`
        ``detector``, ``spectrograph``, ``fiberId``, ``minimum``, ``copy`` and ``donors``, in order
        along the slit and then by arm. Empty for a table that holds no ``minimum``. Placeholders
        are left out.
    """
    columns = ["detector", "spectrograph", "fiberId", "minimum", "copy", "donors"]
    if "minimum" not in fibers:
        return pd.DataFrame(columns=columns)
    table = _withDetector(fibers)
    if "placeholder" in table:
        table = table[~table["placeholder"].astype(bool)]
    table = table[table["minimum"] < limit].assign(_order=lambda t: t["arm"].map(_ARMS.index))
    table = table.sort_values(["fiberId", "_order"]).rename(columns={"replaced": "copy"})
    return table[columns].reset_index(drop=True)


def detectorTable(
    fibers: pd.DataFrame,
    metrics: pd.DataFrame,
    previous: pd.DataFrame | None = None,
    limit: float = WING_EXCESS_LIMIT,
) -> pd.DataFrame:
    """Return one row per detector of the candidate.

    Parameters
    ----------
    fibers, metrics : `pandas.DataFrame`
        ``fiberProfilesQaFibers`` and ``fiberProfilesQaMetrics`` of the candidate.
    previous : `pandas.DataFrame`, optional
        ``fiberProfilesQaFibers`` of the calibration in force.
    limit : `float`
        Wing excess above which a profile is over the limit.

    Returns
    -------
    `pandas.DataFrame`
        `DETECTOR_COLUMNS`: the stored metrics; ``wingFracChange`` and ``fwhmChange``, the change of
        the median over the fibers both calibrations measured, candidate minus previous (NaN
        without ``previous``, or where it lacks the detector); and the measured profiles and the
        copies over the limit.
    """
    if metrics.empty:
        return pd.DataFrame(columns=list(DETECTOR_COLUMNS))
    # One row per detector and calibration; a metric that is NaN (nMissing without a pfsConfig) keeps its column.
    keys = ["_order", "detector", "calibVisit0", "metric"]
    wide = _withDetector(metrics).drop_duplicates(keys).set_index(keys)["value"].unstack("metric")
    wide = wide.reset_index().sort_values("_order")
    over = overLimit(fibers, limit)
    counts = over.groupby(["detector", over["replaced"].astype(bool)]).size().unstack(fill_value=0)
    for name, isCopy in (("nOverMeasured", False), ("nOverCopies", True)):
        wide[name] = wide["detector"].map(counts.get(isCopy, pd.Series(dtype=int))).fillna(0).astype(int)
    wide["wingFracChange"] = float("nan")
    wide["fwhmChange"] = float("nan")
    if previous is not None and not previous.empty:
        new, old = _withDetector(fibers), _withDetector(previous)
        both = new.merge(old, on=["detector", "fiberId"], suffixes=("", "Previous"))
        both = both[~both["replaced"].astype(bool) & ~both["replacedPrevious"].astype(bool)]
        change = both.assign(
            wingFracChange=both["wingFrac"] - both["wingFracPrevious"],
            fwhmChange=both["fwhm"] - both["fwhmPrevious"],
        )
        medians = change.groupby("detector")[["wingFracChange", "fwhmChange"]].median()
        wide["wingFracChange"] = wide["detector"].map(medians["wingFracChange"])
        wide["fwhmChange"] = wide["detector"].map(medians["fwhmChange"])
    for name in DETECTOR_COLUMNS:
        if name not in wide:
            wide[name] = float("nan")
    return wide[list(DETECTOR_COLUMNS)].reset_index(drop=True)


def calibrationSources(fibers: pd.DataFrame, collection: str = "") -> tuple[dict[str, list[str]], list[str]]:
    """Return where each detector's calibration was read from, and the detectors not read from the candidate.

    The candidate's collection is first in the inputs, but a detector it holds no calibration for is
    resolved from the inputs after it: the calibration in force, measured under the candidate's name.

    Parameters
    ----------
    fibers : `pandas.DataFrame`
        ``fiberProfilesQaFibers`` of one candidate.
    collection : `str`
        The candidate's collection; empty for the inputs as they are.

    Returns
    -------
    sources : `dict` [`str`, `list` [`str`]]
        The detectors read from each RUN (``calibRun``), in the order detectors are shown. Empty when
        the table has no ``calibRun``.
    elsewhere : `list` [`str`]
        The detectors whose RUN is not the candidate's collection or inside it, in the order
        detectors are shown; empty when ``collection`` is.
    """
    if fibers.empty or "calibRun" not in fibers:
        return {}, []
    table = _withDetector(fibers).drop_duplicates("detector").sort_values("_order")
    sources: dict[str, list[str]] = {}
    name = collection.rstrip("/")
    elsewhere = []
    for run, detector in zip(table["calibRun"].astype(str), table["detector"], strict=True):
        sources.setdefault(run, []).append(detector)
        if name and run != name and not run.startswith(name + "/"):
            elsewhere.append(detector)
    return sources, elsewhere


def _count(number: int, one: str, many: str) -> str:
    """Return ``number`` with the noun and verb that fit it: ``1 copy is``, ``3 copies are``."""
    return f"{number} {one if number == 1 else many}"


def _bySpectrograph(table: pd.DataFrame) -> str:
    """Return the fibers of ``table`` as ``spectrograph 3: 1321, 1503; spectrograph 4: 2108``."""
    parts = []
    for spectrograph, group in table.groupby("spectrograph", sort=True):
        fibers = ", ".join(str(int(ff)) for ff in sorted(group["fiberId"]))
        parts.append(f"spectrograph {int(spectrograph)}: {fibers}")
    return "; ".join(parts)


def whatToDo(
    byFiber: pd.DataFrame,
    detectors: pd.DataFrame,
    elsewhere: list[str],
    previous: str | None = None,
    limit: float = WING_EXCESS_LIMIT,
    uncompared: list[str] | None = None,
    negative: pd.DataFrame | None = None,
) -> list[str]:
    """Return what the report asks of the person about to certify, one item per finding.

    Parameters
    ----------
    byFiber : `pandas.DataFrame`
        `fibersOverLimit` of the candidate.
    detectors : `pandas.DataFrame`
        `detectorTable` of the candidate.
    elsewhere : `list` [`str`]
        The detectors whose calibration was not read from the candidate (`calibrationSources`).
    previous : `str`, optional
        Label of the calibration in force, when the candidate was set against one.
    limit : `float`
        Wing excess above which a profile is over the limit; a detector whose median wing fraction
        moved by more is named too.
    uncompared : `list` [`str`], optional
        The detectors the calibration in force was not read for, so that the candidate has nothing
        to be set against there.
    negative : `pandas.DataFrame`, optional
        `negativeProfiles` of the candidate.

    Returns
    -------
    `list` [`str`]
        Markdown list items. When nothing stands against certifying, one item says so.
    """
    items = []
    if uncompared:
        items.append(
            f"**No comparison for {' '.join(uncompared)}.** The collection of {previous} holds no calibration for "
            "them, so every fiber over the limit there counts as new."
        )
    if elsewhere:
        items.append(
            f"**Measure again: {' '.join(elsewhere)} are not the candidate's.** Its collection holds no "
            "calibration for them, so the inputs after it were measured under its name. A build's detectors can "
            "be in several RUNs under one parent collection: name the parent."
        )
    missing = detectors["nMissing"].sum(min_count=1)
    unmeasured = detectors["nUnmeasured"].sum(min_count=1)
    if pd.notna(missing) and missing > 0:
        named = " ".join(detectors.loc[detectors["nMissing"] > 0, "detector"])
        fibers = _count(int(missing), "expected fiber has", "expected fibers have")
        items.append(f"**Do not certify: {fibers} no profile** ({named}).")
    if pd.notna(unmeasured) and unmeasured > 0:
        named = " ".join(detectors.loc[detectors["nUnmeasured"] > 0, "detector"])
        items.append(
            f"**Do not certify: {_count(int(unmeasured), 'profile', 'profiles')} could not be measured** "
            f"({named}): masked, or with no positive peak."
        )
    isCopy = byFiber["copy"].astype(bool)
    measured, copies = byFiber[~isCopy], byFiber[isCopy]
    if previous is not None and len(measured):
        isNew = measured["new"].astype(bool)
        fresh, old = measured[isNew], measured[~isNew]
    else:
        fresh, old = measured, measured.iloc[0:0]
    if len(fresh):
        since = f" and not in {previous}" if previous is not None else ""
        items.append(
            f"**Do not certify as built: {_count(len(fresh), 'measured fiber is', 'measured fibers are')} over "
            f"the limit{since}.** Each has more than {limit:g} of its light in the wings beyond the fibers beside "
            "it, and extraction would use that shape for the run. Rebuild so that they are measured cleanly, or "
            "leave them out of the build so that each takes the mean of its neighbours. "
            f"{_bySpectrograph(fresh).capitalize()}."
        )
    if len(old):
        items.append(
            f"{_count(len(old), 'measured fiber is', 'measured fibers are')} over the limit as already in "
            f"{previous}: no worse than what is in force. {_bySpectrograph(old).capitalize()}."
        )
    if len(copies):
        items.append(
            f"{_count(len(copies), 'copy is', 'copies are')} over the limit. Nothing to do for them directly: "
            "each is the mean of the fibers named in its row, and clears when they do."
        )
    if negative is not None and len(negative):
        named = ", ".join(
            f"{row.detector} {int(row.fiberId)} ({row.minimum:.2f})" for row in negative.itertuples()
        )
        items.append(
            f"**{_count(len(negative), 'profile goes', 'profiles go')} far below zero: {named}.** A profile is a "
            "share of the fiber's light, so the build had nothing usable to fit in part of that trace, a masked "
            "region of the detector for one, and what it stored there is not a shape. Look at the detector at "
            "those fibers before certifying."
        )
    for row in detectors[detectors["wingFracChange"].abs() > limit].itertuples():
        items.append(
            f"**{row.detector}: the whole detector's median wing fraction changed by {row.wingFracChange:+.3f}** "
            f"(to {row.medWingFrac:.3f}). Every fiber moves together, so no fiber stands out against its "
            "neighbours for it."
        )
    if not items:
        items.append(
            "Nothing here stands against certifying: no measured fiber is over the limit, none is missing or "
            "unmeasured, no profile goes far below zero, and every detector's calibration is the candidate's."
        )
    return items


def _number(value: float, spec: str) -> str:
    """Format a number, or leave the cell empty when it is missing."""
    return "" if pd.isna(value) else format(value, spec)


def _fiberLines(byFiber: pd.DataFrame, withNew: bool) -> list[str]:
    """Return the Markdown tables of fibers over the limit, one per spectrograph."""
    arms = [name for name in byFiber.columns if name in tuple(_ARMS)]
    lines = []
    for spectrograph, group in byFiber.groupby("spectrograph", sort=True):
        lines += ["", f"### Spectrograph {int(spectrograph)}", ""]
        lines.append("| fiber | " + " | ".join(arms) + " | copy of |" + (" new |" if withNew else ""))
        lines.append("|---|" + "---|" * len(arms) + "---|" + ("---|" if withNew else ""))
        for row in group.itertuples():
            cells = [_number(getattr(row, arm), ".3f") for arm in arms]
            line = f"| {int(row.fiberId)} | " + " | ".join(cells) + f" | {row.donors if row.copy else ''} |"
            if withNew:
                line += f" {'new' if row.new else ''} |"
            lines.append(line)
    return lines


def _detectorLines(detectors: pd.DataFrame, withChange: bool) -> list[str]:
    """Return the Markdown table of detectors."""
    head = ["detector", "built from visit", "profiles", "copies", "missing", "unmeasured", "median FWHM (px)"]
    head += ["change"] if withChange else []
    head += ["median wing fraction"]
    head += ["change"] if withChange else []
    head += ["over the limit: measured", "over the limit: copies"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in detectors.itertuples():
        cells = [
            row.detector,
            str(int(row.calibVisit0)),
            _number(row.nProfiles, ".0f"),
            _number(row.nReplaced, ".0f"),
            _number(row.nMissing, ".0f"),
            _number(row.nUnmeasured, ".0f"),
            _number(row.medProfileFwhm, ".3f"),
            *([_number(row.fwhmChange, "+.3f")] if withChange else []),
            _number(row.medWingFrac, ".4f"),
            *([_number(row.wingFracChange, "+.4f")] if withChange else []),
            str(row.nOverMeasured),
            str(row.nOverCopies),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def buildProfileReport(
    tables: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    candidates: Mapping[str, str],
    title: str,
    previous: str | None = None,
    note: str = "",
    limit: float = WING_EXCESS_LIMIT,
    collections: Mapping[str, str] | None = None,
    figures: Mapping[str, str] | None = None,
    minimumLimit: float = PROFILE_MINIMUM_LIMIT,
) -> str:
    """Return the Markdown report on candidate ``fiberProfiles`` calibrations.

    Parameters
    ----------
    tables : mapping of `str` to (`pandas.DataFrame`, `pandas.DataFrame`)
        ``fiberProfilesQaFibers`` and ``fiberProfilesQaMetrics`` of each candidate, by label.
    candidates : mapping of `str` to `str`
        How each candidate is shown: its collection.
    title : `str`
        The report's title.
    previous : `str`, optional
        Label of the calibration in force. The others are set against it, and it gets no section of
        its own.
    note : `str`, optional
        A line under the title: the visit the calibrations were resolved for.
    limit : `float`
        Wing excess above which a profile is over the limit.
    collections : mapping of `str` to `str`, optional
        Each candidate's collection, empty for the inputs as they are: the report then names the
        detectors whose calibration was not read from it.
    figures : mapping of `str` to `str`, optional
        A figure file for a candidate, by label (`pfs.drp.qa.plotting.fiberProfiles.plotWingExcess`).
    minimumLimit : `float`
        Lowest value, as a fraction of its peak, below which a profile is listed (`negativeProfiles`).

    Returns
    -------
    `str`
        The report: for each candidate, what to do, the figure, the fibers over the limit, the
        profiles that go far below zero (when there are any) and the detectors.
    """
    out = [f"# {title}", ""]
    if note:
        out += [note, ""]
    out += ["| | collection |", "|---|---|"]
    for label in tables:
        shown = candidates.get(label, "")
        role = "in force" if label == previous else "candidate"
        out.append(f"| {label} ({role}) | {shown if '`' in shown else f'`{shown}`'} |")
    # A detector measured on several visits is reported once: the first rows of it.
    tables = {
        label: (
            fibers.drop_duplicates(["arm", "spectrograph", "fiberId"]),
            metrics.drop_duplicates(["arm", "spectrograph", "metric"]),
        )
        for label, (fibers, metrics) in tables.items()
    }
    before = tables[previous][0] if previous is not None and previous in tables else None
    against = previous if before is not None else None
    uncompared: list[str] = []
    if before is not None:
        # A detector the calibration in force was not read for is no reference: leave it out of the comparison.
        _, uncompared = calibrationSources(before, (collections or {}).get(previous, ""))
        if uncompared:
            before = before[~_withDetector(before)["detector"].isin(uncompared).to_numpy()]
    for label, (fibers, metrics) in tables.items():
        if label == previous:
            continue
        byFiber = fibersOverLimit(fibers, before, limit)
        detectors = detectorTable(fibers, metrics, before, limit)
        sources, elsewhere = calibrationSources(fibers, (collections or {}).get(label, ""))
        out += ["", f"## {label}: what to do", ""]
        negative = negativeProfiles(fibers, minimumLimit)
        items = whatToDo(byFiber, detectors, elsewhere, against, limit, uncompared, negative)
        out += [f"- {item}" for item in items]
        if figures and label in figures:
            out += ["", f"![Wing light above the neighbours, along the slits ({label})]({figures[label]})"]
        out += [
            "",
            f"## {label}: fibers over the limit",
            "",
            "Over the limit: the fiber's wing fraction (the share of its profile's light 4 px or more from the "
            f"center) is more than {limit:g} above the median of the measured fibers beside it on the slit. A "
            "copy is a profile the build filled in with the mean of nearby fibers' profiles, not a measurement. "
            "Each cell is the excess in that arm; an empty cell is under the limit.",
        ]
        out += _fiberLines(byFiber, before is not None) if len(byFiber) else ["", "None."]
        if len(negative):
            out += [
                "",
                f"## {label}: profiles far below zero",
                "",
                f"Profiles whose lowest value is below {minimumLimit:g} of their peak in some part of the trace.",
                "",
                "| detector | fiber | lowest value | copy of |",
                "|---|---|---|---|",
            ]
            out += [
                f"| {row.detector} | {int(row.fiberId)} | {row.minimum:.2f} | {row.donors if row.copy else ''} |"
                for row in negative.itertuples()
            ]
        out += ["", f"## {label}: detectors", ""]
        out += _detectorLines(detectors, before is not None)
        if before is not None:
            out += [
                "",
                f"Change: the median over the fibers both calibrations measured, {label} minus {previous}.",
            ]
        if sources:
            out += ["", "Calibration read from:", ""]
            out += [f"- `{run}`: {' '.join(dets)}" for run, dets in sources.items()]
    return "\n".join(out) + "\n"
