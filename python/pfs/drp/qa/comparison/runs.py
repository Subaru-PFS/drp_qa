"""Observing runs and their pre-run calibration periods.

A *period* is a run's nights, or the nights before it when its calibrations were taken
(``run29-pre``). Comparison mode reads and gates one period at a time. The periods are listed
in ``data/observingRuns.yaml``, package data.

Times are HST, as the opdb stores them. A night runs from noon to noon and is named by its
evening's date, so an exposure at 03:00 on 07-24 belongs to the night of 07-23.
"""

import datetime
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from itertools import pairwise
from pathlib import Path

import pandas as pd
import yaml

__all__ = [
    "NIGHT_START",
    "Period",
    "defaultRunsPath",
    "loadPeriods",
    "nightOf",
    "periodOf",
]

#: The hour (HST) at which a night begins.
NIGHT_START = 12


@dataclass(frozen=True)
class Period:
    """A run, or the calibration period before one.

    Parameters
    ----------
    name : `str`
        ``run30``, or ``run30-pre`` for the period before it.
    run : `str`
        The run it belongs to: ``run30`` for both.
    firstNight, lastNight : `datetime.date`
        The first and last nights, inclusive, named by their evening's date.
    """

    name: str
    run: str
    firstNight: datetime.date
    lastNight: datetime.date

    @property
    def isPreRun(self) -> bool:
        """Whether this is the calibration period before a run."""
        return self.name != self.run

    @property
    def start(self) -> datetime.datetime:
        """The start of the first night (HST), inclusive."""
        return datetime.datetime.combine(self.firstNight, datetime.time(NIGHT_START))

    @property
    def end(self) -> datetime.datetime:
        """The end of the last night (HST), exclusive."""
        return datetime.datetime.combine(
            self.lastNight + datetime.timedelta(days=1), datetime.time(NIGHT_START)
        )

    def __contains__(self, when: datetime.datetime) -> bool:
        return self.start <= when < self.end


def defaultRunsPath() -> Path:
    """Return the path of the packaged ``observingRuns.yaml``."""
    return Path(str(files("pfs.drp.qa.comparison") / "data" / "observingRuns.yaml"))


def loadPeriods(path: Path | str | None = None) -> dict[str, Period]:
    """Read the observing runs and their pre-run periods.

    Parameters
    ----------
    path : `pathlib.Path` or `str`, optional
        The YAML file. Defaults to the packaged ``observingRuns.yaml``.

    Returns
    -------
    `dict` [`str`, `Period`]
        Periods by name, in time order.

    Raises
    ------
    ValueError
        If an entry is malformed, a period ends before it starts, or two
        periods overlap.
    """
    path = Path(path) if path is not None else defaultRunsPath()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("runs"), list):
        raise ValueError(f"{path}: expected a top-level 'runs' list")

    periods = []
    for item in raw["runs"]:
        if not isinstance(item, dict) or "name" not in item or "nights" not in item:
            raise ValueError(f"{path}: each run needs a name and nights: {item!r}")
        run = str(item["name"])
        periods.append(_period(run, run, item["nights"], path))
        if "preRun" in item:
            periods.append(_period(f"{run}-pre", run, item["preRun"], path))

    periods.sort(key=lambda period: period.firstNight)
    for before, after in pairwise(periods):
        if after.firstNight <= before.lastNight:
            raise ValueError(f"{path}: {before.name} and {after.name} overlap")
    names = [period.name for period in periods]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate run names")
    return {period.name: period for period in periods}


def nightOf(when: pd.Series | Iterable[datetime.datetime]) -> pd.Series:
    """Return the night of each time, named by its evening's date.

    Parameters
    ----------
    when : `pandas.Series` or iterable of `datetime.datetime`
        Times (HST).

    Returns
    -------
    `pandas.Series`
        `datetime.date` for each time.
    """
    when = pd.Series(pd.to_datetime(pd.Series(when)).to_numpy())
    return (when - pd.Timedelta(hours=NIGHT_START)).dt.date


def periodOf(when: pd.Series | Iterable[datetime.datetime], periods: Iterable[Period]) -> pd.Series:
    """Return the name of the period containing each time.

    Parameters
    ----------
    when : `pandas.Series` or iterable of `datetime.datetime`
        Times (HST).
    periods : iterable of `Period`
        The periods, e.g. ``loadPeriods().values()``.

    Returns
    -------
    `pandas.Series`
        The period's name, or `None` for a time outside every period.
    """
    when = pd.Series(pd.to_datetime(pd.Series(when)).to_numpy())
    names = pd.Series([None] * len(when), dtype=object)
    for period in periods:
        names[(when >= period.start) & (when < period.end)] = period.name
    return names


def _period(name: str, run: str, nights, path: Path) -> Period:
    """Parse one ``[first, last]`` pair of nights."""
    if not isinstance(nights, list | tuple) or len(nights) != 2:
        raise ValueError(f"{path}: {name}: expected [first, last] nights, not {nights!r}")
    try:
        first, last = (datetime.date.fromisoformat(str(night)) for night in nights)
    except ValueError as error:
        raise ValueError(f"{path}: {name}: {error}") from None
    if last < first:
        raise ValueError(f"{path}: {name} ends before it starts")
    return Period(name=name, run=run, firstNight=first, lastNight=last)
