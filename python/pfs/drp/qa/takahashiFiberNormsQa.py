import eups
from lsst.daf.butler import Butler, DatasetRef
from pfs.datamodel import FiberStatus, TargetType
from pfs.drp.stella.datamodel import PfsConfig, PfsFiberNorms, PfsArm
from pfs.utils.fiberids import FiberIds

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pytz
from scipy.signal import medfilt

import argparse
import contextlib
import dataclasses
import datetime
import enum
import functools
import json
import logging
import logging.handlers
import multiprocessing
import os
import random
import re
import shutil
import signal
import sys
import textwrap
import traceback
import typing
import warnings

from collections.abc import Callable, Generator, Iterable
from typing import Any, Literal
from typing import ParamSpec, TypeVar  # These won't be necessary in future pythons.


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
U = TypeVar("U")


SpectrographId = Literal[1, 2, 3, 4]
"""Spectrograph ID in PFS.
"""

ArmId = Literal["b", "r", "m", "n"]
"""Arm ID in PFS
"""


def main() -> int:
    """Takahashi's fiberNormsQa.

    This QA monitors fiber-to-fiber and visit-to-visit throughput variation.
    Though index-matching gel has been applied to all fibers by run20,
    it is still crucial to monitor such variation in MTP/fiber units.

    We take the ratio of a target quartz flux to a reference quartz
    to show the uniformity of the target quartz in the focal plane
    and how the spectra change relative to those of the reference quartz.
    """
    parser = argparse.ArgumentParser(
        description=layout_paragraphs(typing.cast(str, main.__doc__)),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparser = parser.add_argument_group("Input/output options")
    subparser.add_argument(
        "-b",
        "--butler-config",
        metavar="PATH",
        required=True,
        help="""
        Path to a butler config file or its parent directory.
        """,
    )
    subparser.add_argument(
        "-i",
        "--input",
        metavar="COLLECTION[,...]",
        required=True,
        type=argtype_comma_separated(str, nargs="+"),
        help="""
        Input collections.
        """,
    )
    subparser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="""
        Output directory.
        """,
    )
    subparser.add_argument(
        "-d",
        "--data-query",
        metavar="QUERY",
        default="",
        help="""
        Data selection expression.
        """,
    )
    subparser.add_argument(
        "-r",
        "--reference",
        metavar="VISIT",
        type=int,
        help="""
        Reference visit. (default: pfsArm with the smallest visit number)
        """,
    )

    subparser = parser.add_argument_group("Execution options")
    subparser.add_argument(
        "--fail-fast",
        action="store_true",
        help="""
        Stop processing at first error.
        """,
    )
    subparser.add_argument(
        "-j",
        "--jobs",
        metavar="NUM",
        type=int,
        help="""
        Number of parallel jobs. (default: use all cores)
        """,
    )
    subparser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="""
        Logging level.
        Messages are shown if their importance is higher than or equal to this level.
        """,
    )

    subparser = parser.add_argument_group("Config options")
    config = TakahashiFiberNormsQaConfig()
    add_dataclass_to_argparser(config, subparser, prefix="config")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s:%(process)d:%(name)s:%(message)s"
    )
    logging.captureWarnings(True)
    log = logging.getLogger("main")

    log.info("Querying datasets...")

    butler = Butler.from_config(args.butler_config, collections=args.input)
    dataset_groups = typing.cast(
        list[TakahashiFiberNormsQaInput],
        get_datasets(
            butler,
            args.data_query,
            ["instrument", "visit", "arm"],
            [
                ("pfsConfig", Multiplicity.SIMPLE),
                ("pfsArm", Multiplicity.MULTIPLE),
                ("fiberNorms", Multiplicity.SIMPLE),
            ],
        ),
    )

    if args.reference is not None:
        ref_visit = args.reference
    else:
        ref_visit = min(typing.cast(int, group["pfsArm"][0].dataId["visit"]) for group in dataset_groups)

    log.info("Use visit=%d as the reference.", ref_visit)

    # Make each group equipped with reference pfsArms.
    for group in dataset_groups:
        if ref_visit == group["pfsArm"][0].dataId["visit"]:
            continue

        refArmRefs: list[list[DatasetRef]] = [
            butler.query_datasets(
                "pfsArm",
                data_id={
                    "instrument": ref.dataId["instrument"],
                    "visit": ref_visit,
                    "arm": ref.dataId["arm"],
                    "spectrograph": ref.dataId["spectrograph"],
                },
                find_first=True,
                limit=None,
                explain=False,
            )
            for ref in group["pfsArm"]
        ]
        if any(len(refs) > 1 for refs in refArmRefs):
            raise RuntimeError(
                "multiple pfsArms (for reference) are found for a group"
                " (maybe database is broken hopelessly)."
            )

        # Redefine `group["pfsArm"]` so that it will contain
        # only those spectrographs for which references exist.
        pfsArmRefs = [arm for arm, refs in zip(group["pfsArm"], refArmRefs) if refs]
        if not pfsArmRefs:
            continue

        group["pfsArm"] = pfsArmRefs
        group["refArm"] = [refs[0] for refs in refArmRefs if refs]

    dataset_groups = [group for group in dataset_groups if "refArm" in group]

    if not dataset_groups:
        raise RuntimeError(
            'No matching tuples among "pfsConfig", "pfsArm" and "fiberNorms" have reference pfsArm.'
        )

    num_groups = len(dataset_groups)
    log.info("%d dataset group(s) found.", num_groups)

    arglist: list[_ThreadProcArgs] = [
        {
            "job_id": i,
            "num_jobs": num_groups,
            "butler": butler,
            "output": args.output,
            "input": group,
            "config": config,
            "log": log,
        }
        for i, group in enumerate(dataset_groups, start=1)
    ]

    success_count = 0
    error_count = 0

    with unordered_parallel_map(args.jobs, _main_threadproc, arglist) as statuses:
        for status in statuses:
            if status == _ThreadProcStatus.SUCCESS:
                success_count += 1
            else:
                error_count += 1
            if status == _ThreadProcStatus.FATAL:
                break
            if args.fail_fast and status == _ThreadProcStatus.ERROR:
                break

    num_remainders = num_groups - success_count - error_count

    if error_count == 0 and num_remainders == 0:
        log.info("%d jobs all succeeded.", success_count)
        return 0
    else:
        log.warning(
            "%d jobs succeeded, %d jobs failed, %d jobs were interrupted or not executed.",
            success_count,
            error_count,
            num_remainders,
        )
        return 1


class _ThreadProcArgs(typing.TypedDict):
    """Argument of ``_main_threadproc()``

    Keys
    ----
    job_id :  `int`
        Job ID. 1 <= job_id <= num_jobs.
    num_jobs : `int`
        Total number of jobs.
    butler : `Butler`
        Read-only butler.
    output : `str`
        Output directory.
    input : `TakahashiFiberNormsQaInput`
        Input to ``TakahashiFiberNormsQa``.
    config : `TakahashiFiberNormsQaConfig`
        Config of ``TakahashiFiberNormsQa``.
    log : `logging.Logger`
        Parent logger.
    """

    job_id: int
    num_jobs: int
    butler: Butler
    output: str
    input: "TakahashiFiberNormsQaInput"
    config: "TakahashiFiberNormsQaConfig"
    log: logging.Logger


class _ThreadProcStatus(enum.Enum):
    """Return value of ``_main_threadproc()``"""

    SUCCESS = 0
    ERROR = 1
    FATAL = 2


def _main_threadproc(args: _ThreadProcArgs) -> _ThreadProcStatus:
    """Threading part of ``main()`` function.

    This function runs a single ``TakahashiFiberNormsQa``.
    with the given arguments.

    Parameters
    ----------
    args : `_ThreadProcArgs`
        Arguments.

    Returns
    -------
    status : `_ThreadProcStatus`
        ``SUCCESS`` on success, and ``ERROR`` on error.
        ``FATAL`` if the main function should exit immediately.
    """
    log = args.get("log") or logging.getLogger()
    job_id = args.get("job_id", 0)
    num_jobs = args.get("num_jobs", 0)

    status = _ThreadProcStatus.SUCCESS

    try:
        butler = args["butler"]
        output = args["output"]
        input = args["input"]
        config = args["config"]

        log.info("[%d/%d] Starting a new job...", job_id, num_jobs)
        task = TakahashiFiberNormsQa(butler, output, input, config=config, log=log)
        task.run()
        log.info("[%d/%d] Job done", job_id, num_jobs)
    except Exception:
        log.error("[%d/%d] %s", job_id, num_jobs, traceback.format_exc())
        status = _ThreadProcStatus.ERROR
    except BaseException:
        log.error("[%d/%d] %s", job_id, num_jobs, traceback.format_exc())
        status = _ThreadProcStatus.FATAL

    return status


def argtype_comma_separated(
    type: Callable[[str], T], *, nargs: int | Literal["*", "+"], restype: type = list
) -> Callable[[str], Iterable[T]]:
    """Comma-separated list

    The return value of this function is intended to be used as the ``type``
    argument of `argtype.ArgumentParser.add_argument`

    Parameters
    ----------
    type : `Callable` [[`str`], `T`]
        Type of elements, or converter to the type.
    nargs : `int` | `Literal` ["*", "+"]
        Number of elements.
        Special values are accepted: "*" (zero or more), "+" (one or more).
    restype : `type`
        Result type (default: `list`)

    Returns
    -------
    func : `Callable`
        This function takes a string and returns a list of elements of type
        ``typ``.
    """

    def _argtype_comma_separated(s: str) -> Iterable[T]:
        """Comma-separated list

        This function is intended to be used as the ``type`` argument of
        `argtype.ArgumentParser.add_argument`

        Parameters
        ----------
        s : `str`
            Comma-separated list.

        Returns
        -------
        elements : `restype`
            List of elements.
        """
        elements: list[T] = []
        for elem in s.split(","):
            elem = elem.strip()
            if not elem:
                continue
            try:
                t = type(elem)
            except Exception:
                typename = getattr(type, "__name__", "")
                if typename:
                    typename += " "
                raise argparse.ArgumentTypeError(f"invalid {typename}value: '{elem}'.") from None

            elements.append(t)

        if isinstance(nargs, str) and nargs == "*":
            pass
        if isinstance(nargs, str) and nargs == "+" and not elements:
            raise argparse.ArgumentTypeError("one or more values required.")
        if isinstance(nargs, int) and nargs != len(elements):
            raise argparse.ArgumentTypeError(f"number of values must be {nargs}")

        return restype(elements)

    return _argtype_comma_separated


class DataclassStoreAction(argparse.Action):
    """A variant of "store" action for `argparse.ArgumentParser`.

    This action stores values not in a given `argparse.Namespace`
    but in another object (dataclass).

    Though all subclasses of `argparse.Action` must have the same constructor
    signature, this class has a signature different from that of
    `argparse.Action.` Users must use lambda like:

        parser.add_argument(
            ...,
            action=(
                lambda *args, **kwargs:
                    DataclassStoreAction(dataclass, "", *args, **kwargs)
            )
        )

    Parameters
    ----------
    dataclass : `Any`
        Dataclass instance in which to store an argument.
    prefix : `str`
        Prefix of destination to be removed. For example, if prefix == "config"
        and dest == "config_x", then an argument is stored in `dataclass.x`.
    *args
        Positional arguments passed to `argparse.Action`
    **kwargs
        Keyword arguments passed to `argparse.Action`
    """

    def __init__(self, dataclass: Any, prefix: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dataclass = dataclass
        self.__prefix = prefix

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        dest = self.dest[len(self.__prefix) + 1 :] if self.__prefix else self.dest
        if not hasattr(self.__dataclass, dest):
            raise RuntimeError(f"dataclass does not have a member named '{dest}'")
        setattr(self.__dataclass, dest, values)


def add_dataclass_to_argparser(
    dataclass: Any, parser: argparse._ActionsContainer, *, prefix: str = ""
) -> None:
    """Add members of a dataclass to argparser

    The types of the members of ``dataclass`` must be `str`, `int`, `float`,
    `tuple` [`str`, ...], `tuple` [`int`, ...], or `tuple` [`float`, ...].
    The length of the tuple must actually be positive and it must be finite.

    If the definition of a member of the ``dataclass`` is like
    ``member: int = dataclasses.field(default=0, metadata={...})``
    and the metadata has "doc" member, the help message will contain
    ``metadata["doc"]``.

    Parameters
    ----------
    dataclass : `Any`
        Dataclass object.
    parser : `argparse._ActionsContainer`
        Command line parser
    prefix : `str`
        String to be prefixed to member names. For example, if prefix="config",
        then member "x" is exposed as a command line option "--config-x".
    """

    store_in_dataclass = typing.cast(
        type[argparse.Action],
        lambda *args, **kwargs: DataclassStoreAction(dataclass, prefix, *args, **kwargs),
    )

    for field in dataclasses.fields(dataclass):
        hyphened = field.name.replace("_", "-")
        optionname = f"--{prefix}-{hyphened}" if prefix else f"--{hyphened}"
        fieldtype = typing.cast(type, field.type)
        doc = field.metadata.get("doc", "")
        if fieldtype is str or fieldtype is int or fieldtype is float:
            parser.add_argument(
                optionname,
                metavar=fieldtype.__name__.upper(),
                type=fieldtype,
                action=store_in_dataclass,
                help=f"""
                {doc} (default: {getattr(dataclass, field.name)})
                """,
            )
            continue
        if typing.get_origin(fieldtype) is tuple:
            argtypes = typing.get_args(fieldtype)
            nargs = len(argtypes)
            if nargs >= 1 and (
                all(t is str for t in argtypes)
                or all(t is int for t in argtypes)
                or all(t is float for t in argtypes)
            ):
                parser.add_argument(
                    optionname,
                    metavar=",".join(t.__name__.upper() for t in argtypes),
                    type=argtype_comma_separated(argtypes[0], nargs=nargs, restype=tuple),
                    action=store_in_dataclass,
                    help=f"""
                    {doc} (default: {",".join(str(x) for x in getattr(dataclass, field.name))})
                    """,
                )
                continue

        raise RuntimeError(f"dataclass member {field.name} cannot be made a command line option.")


def layout_paragraphs(text: str) -> str:
    """Lay out paragraphs so that they will fill the columns of the terminal.

    Parameters
    ----------
    text : `str`
        Text consisting of paragraphs separated by two linebreaks.

    Returns
    -------
    newtext : `str`
        Layed-out text.
    """
    width = max(1, shutil.get_terminal_size().columns - 2)
    return (
        "\n\n".join(
            textwrap.fill(re.sub(r"\s+", " ", paragraph.strip()), width=width)
            for paragraph in text.strip().split("\n\n")
        )
        + "\n"
    )


def unordered_parallel_map(
    processes: int | None, func: Callable[[T], U], arglist: Iterable[T]
) -> contextlib.AbstractContextManager[Iterable[U]]:
    """Unordered, parallel version of ``map``.

    This is a wrapper of ``multiprocessing.Pool.imap_unordered``.

    Parameters
    ----------
    processes : `int` | None
        Number of worker processes. If ``None``, all cores are used.
    func : `Callable` [[`T`], `U`]
        Function that maps `T` to `U`.
    arglist : `Iterable` [`T`]
        List of `T`.

    Returns
    -------
    results : `contextlib.AbstractContextManager` [`Iterable` [`U`]]
        Equivalent to iterable ``(func(t) for t in arglist)``, except that
        the order of the elements is unpredictable.
    """

    def _generator() -> Generator[U, None, None]:
        """Yield ``func(t)`` for ``t`` in ``arglist``.

        Yields
        -------
        result : `U`
            Result of ``func(t)``, where ``t`` belongs to ``arglist``.
        """
        if processes is not None and processes <= 1:
            yield from map(func, arglist)
        else:
            with install_centralized_logger():
                with multiprocessing.Pool(processes, initializer=_poolworker_ignore_signal) as pool:
                    yield from pool.imap_unordered(func, arglist)

    # `pool` variable in _generator() is closed only if the return value of
    # _generator() is iterated thoroughly or is closed. If it were not for
    # `contextlib.closing`, `pool` might stay alive after iteration is
    # interrupted by `break` or an exception.
    return contextlib.closing(_generator())


@contextlib.contextmanager
def install_centralized_logger() -> Generator[None, None, None]:
    """Install a centralized logger.

    This context manager installs a centralized logger in place of the current
    root logger. Centralizing the logger prevents concurrent log messages by
    multiple processes from interleaving in the middle of a line.
    """
    log = logging.getLogger()
    oldhandlers = list(log.handlers)
    newhandler: logging.handlers.QueueHandler | None = None
    listener: logging.handlers.QueueListener | None = None
    try:
        if oldhandlers:
            q: multiprocessing.Queue = multiprocessing.Queue()
            newhandler = logging.handlers.QueueHandler(q)
            listener = logging.handlers.QueueListener(q, *oldhandlers)
            log.addHandler(newhandler)
            for handler in oldhandlers:
                log.removeHandler(handler)

            listener.start()

        try:
            yield
        finally:
            if listener is not None:
                listener.stop()

    finally:
        while log.handlers:
            log.removeHandler(log.handlers[-1])
        for handler in oldhandlers:
            log.addHandler(handler)


def _poolworker_ignore_signal() -> None:
    """Ignore SIGINT.

    A worker process of `multiprocessing.Pool` must not die abruptly even if
    it receives a signal. We cannot avoid such a death by catching
    `KeyboardInterrupt` exceptions in a mapper function, because the worker
    process may be asleep outside the mapper function, having nothing to do.
    An exception handler in the mapper function cannot catch
    `KeyboardInterrupt` raised outside the function. If the worker process die
    abruptly, the main process will wait a reply from the dead worker,
    forever.

    Fortunately, all we have to do is to set signal handlers to SIG_IGN in
    worker processes. Worker processes do not have to kill themselves because
    the main process will kill workers if it wants to die, receiving a signal.
    """
    # It seems that signals other than SIGINT should not be ignored:
    # for example, if we were to ignore SIGTERM in addition to SIGINT,
    # the behavior of `multiprocessing.Pool` on Ctrl+C would be buggy again
    # (python-3.11).
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class Multiplicity(enum.Enum):
    """Multiplicity: either SIMPLE or MULTIPLE."""

    SIMPLE = 0
    MULTIPLE = 1


def get_datasets(
    butler: Butler, where: str, dimensions: list[str], dataset_types: list[tuple[str, Multiplicity]]
) -> list[dict[str, DatasetRef | list[DatasetRef]]]:
    """Get datasets grouped by ``dimensions``

    At least one group is returned.
    If no group is found, this function raises an exception.

    Parameters
    ----------
    butler : `Butler`
        Read-only butler.
    where : `str`
        Query string.
    dimensions : `list` [`str`]
        Dimensions by which to group datasets.
    dataset_types : `list` [tuple[`str`, `Multiplicity`]]
        Dataset types to get. If the second element in a tuple is
        ``Multiplicity.SIMPLE``, only one dataset per group is allowed.
        If it is ``Multiplicity.MULTIPLE``, multiple datasets per group are
        allowed.

    Returns
    -------
    datasets : `list` [`dict` [`str`, `DatasetRef` | `list` [`DatasetRef`]]]
        Dataset groups. Each element is a mapping from dataset type to either
        `DatasetRef` (when ``Multiplicity.SIMPLE``) or `list` [`DatasetRef`]
        (when ``Multiplicity.MULTIPLE``).
    """
    refsdict: dict[str, list[DatasetRef]] = {
        dataset_type: list(
            set(
                butler.query_datasets(
                    dataset_type,
                    where=where,
                    find_first=True,
                    limit=None,
                )
            )
        )
        for dataset_type, multiplicity in dataset_types
    }

    groupkeys = set[DatasetCoord]()
    for refs in refsdict.values():
        groupkeys = set(coordinates_get(ref, dimensions) for ref in refs)
        break
    for refs in refsdict.values():
        groupkeys = coordinates_intersect(groupkeys, set(coordinates_get(ref, dimensions) for ref in refs))

    groups = [
        {
            dataset_type: [
                ref for ref in refs if coordinates_iscompat(groupkey, coordinates_get(ref, dimensions))
            ]
            for dataset_type, refs in refsdict.items()
        }
        for groupkey in groupkeys
    ]

    groups = [group for group in groups if all(len(refs) > 0 for refs in group.values())]
    if not groups:
        raise RuntimeError(f'no dataset group matching the query ("{where}") is found.')

    for dataset_type, multiplicity in dataset_types:
        if multiplicity == Multiplicity.SIMPLE:
            if any(len(group[dataset_type]) > 1 for group in groups):
                raise RuntimeError(
                    f"multiple '{dataset_type}' are found (maybe database or program is broken hopelessly)."
                )

    multiplicities = {dataset_type: multiplicity for dataset_type, multiplicity in dataset_types}

    return [
        {
            dataset_type: (refs[0] if multiplicities[dataset_type] == Multiplicity.SIMPLE else refs)
            for dataset_type, refs in group.items()
        }
        for group in groups
    ]


DatasetCoord = frozenset[tuple[str, Any]]


def coordinates_get(ref: DatasetRef, dimensions: list[str]) -> DatasetCoord:
    """Get coordinates spanned by ``dimensions``.

    The returned value is like ``{("instrument", "PFS"), ("visit", 1000)}``.

    Parameters
    ----------
    ref : `DatasetRef`
        Dataset reference.
    dimensions : `list` [`str`]
        List of dimension names.

    Returns
    -------
    coordinates : DatasetCoord
        Coordinates.
    """
    return DatasetCoord((dim, ref.dataId[dim]) for dim in dimensions if dim in ref.dataId)


def coordinates_iscompat(coord1: DatasetCoord, coord2: DatasetCoord) -> bool:
    """Compare ``coord1`` and ``coord2`` not strictly.

    For example,
        coord1 = {("visit", 1000)}
        coord2 = {("visit", 1000), (arm, "b")}
    are considered compatible with each other.

    Parameters
    ----------
    coord1 : `DatasetCoord`
        Left-hand side of the comparison.
    coord2 : `DatasetCoord`
        Right-hand side of the comparison.

    Returns
    -------
    compat : bool
        True if two `DatasetCoord` are compatible.
    """
    return coord1 <= coord2 or coord2 <= coord1


def coordinates_intersect(coords1: set[DatasetCoord], coords2: set[DatasetCoord]) -> set[DatasetCoord]:
    """Take the intersection between two sets of `DatasetCoord`.

    Comparison between two `DatasetCoord` is based on ``coordinates_iscompat``.
    All `DatasetCoord` in ``coords1`` are assumed to have the same set of
    dimensions. So are all `DatasetCoord` in ``coords2``.

    Parameters
    ----------
    coords1 : `set` [`DatasetCoord`]
        Left-hand side of the intersection.
    coords2 : `set` [`DatasetCoord`]
        Right-hand side of the intersection.

    Returns
    -------
    intersection : `set` [`DatasetCoord`]
        Intersection of the two sets.
    """
    for coord in coords1:
        dims1 = set(key for key, value in coord)
        break
    else:
        return set()

    for coord in coords2:
        dims2 = set(key for key, value in coord)
        break
    else:
        return set()

    if dims1 == dims2:
        return coords1 & coords2

    if dims1 < dims2:
        pass
    elif dims2 < dims1:
        dims1, dims2 = dims2, dims1
        coords1, coords2 = coords2, coords1
    else:
        return set()

    # Note: lifting[y] = {x : P(x) = y},
    # where P is the projection from dim2 to dim1.
    lifting: dict[DatasetCoord, list[DatasetCoord]] = {}
    for coord in coords2:
        proj = DatasetCoord((key, value) for key, value in coord if key in dims1)
        lifted = lifting.get(proj)
        if lifted is None:
            lifted = lifting[proj] = []
        lifted.append(coord)

    intersection: set[DatasetCoord] = set()
    for coord in coords1 & set(lifting.keys()):
        intersection.update(lifting[coord])

    return intersection


def ignore_numpy_warnings(func: Callable[P, R]) -> Callable[P, R]:
    """Ignore numpy warnings caused by NaN and division by zero.

    This is a function decorator.

    Parameters
    ----------
    func : `Callable`
        A function.

    Returns
    -------
    func : `Callable`
        The same function as the argument, except that the function body is
        executed in an  environment where warnings are suppressed.
    """

    @functools.wraps(func)
    def with_numpy_warnings_ignored(*args: P.args, **kwargs: P.kwargs) -> R:
        """Run ``func`` in an  environment where warnings are suppressed."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r".*(?:NaN|invalid value encountered|divide by zero)")
            return func(*args, **kwargs)

    return with_numpy_warnings_ignored


@dataclasses.dataclass
class TakahashiFiberNormsQaStat:
    """Statistics.

    Parameters
    ----------
    hst : `str`
        Date of observation, Hawaii standard time.
    insrot : `float`
        Instrument rotation.
    azimuth : `float`
        Azimuthal angle of the telescope.
    sigma_std_median : `float`
        median of per-fiber standard deviations of normalized flux ratios.
    sigma_std_95 : `float`
        95 percentile of per-fiber standard deviations of normalized flux ratios.
    twosigma_std : `float`
        2 times ``sigma_std_median``
    twosigma_std_95 : `float`
        2 times ``sigma_std_95``
    sigma_iqr : `float`
        median of per-fiber scaled interquartile ranges of normalized flux ratios.
    df : `pd.DataFrame`
        Data frame. The following columns exist:

        - fiberId : `int`
            Fiber ID.
        - mtp_A : `str`
            MTP ID on Spectrograph System
        - mtpGroup : `str`
            MTP group.
        - arm : `ArmId`
            Arm name.
        - SM : `SpectrographId` | `Literal` [0]
            Spectrograph name. (Invalid if zero)
        - sigma : `float`
            Per-fiber standard deviation with respect to wavelength.
    """

    hst: str
    insrot: float
    azimuth: float
    sigma_std_median: float
    sigma_std_95: float
    twosigma_std: float
    twosigma_std_95: float
    sigma_iqr: float
    df: pd.DataFrame


class TakahashiFiberNormsQaInput(typing.TypedDict):
    """Input dataset references for `TakahashiFiberNormsQa`

    Keys
    ----
    pfsConfig : `DatasetRef` (pointing to `PfsConfig`)
        Top-end fiber configuration.
    pfsArm : `list` [`DatasetRef`] (pointing to `PfsConfig`)
        PfsArm for assessment.
        All elements in the list must be equal up to "spectrograph".
    fiberNorms : `DatasetRef` (pointing to `PfsFiberNorms`)
        Fiber norms.
    refArm : `list` [`DatasetRef`] (pointing to `PfsConfig`)
        PfsArm for reference.
        All elements in the list must be equal up to "spectrograph".
    """

    pfsConfig: DatasetRef
    pfsArm: list[DatasetRef]
    fiberNorms: DatasetRef
    refArm: list[DatasetRef]


@dataclasses.dataclass
class TakahashiFiberNormsQaConfig:
    """Config for `TakahashiFiberNormsQa`."""

    fontsize: float = dataclasses.field(
        default=20,
        metadata={"doc": "Font size in points."},
    )
    figsize: tuple[float, float] = dataclasses.field(
        default=(30, 55), metadata={"doc": "Figure size (width, height) in inches."}
    )

    random_seed: str = dataclasses.field(
        default="TakahashiFiberNormsQa random seed",
        metadata={"doc": "Random seed. This can be any text string."},
    )
    vmin: float = dataclasses.field(
        default=0.97, metadata={"doc": "min limit of normalized flux value in plots"}
    )
    vmax: float = dataclasses.field(
        default=1.03, metadata={"doc": "max limit of normalized flux value in plots"}
    )
    vmin2: float = dataclasses.field(default=0.0, metadata={"doc": "min limit of sigma in plots"})
    vmax2: float = dataclasses.field(default=0.05, metadata={"doc": "max limit of sigma in plots"})

    wavelength_range_b: tuple[float, float] = dataclasses.field(
        default=(450, 650), metadata={"doc": "Wavelength range to use (b-arm)"}
    )
    wavelength_range_r: tuple[float, float] = dataclasses.field(
        default=(650, 950), metadata={"doc": "Wavelength range to use (r-arm)"}
    )
    wavelength_range_m: tuple[float, float] = dataclasses.field(
        default=(650, 950), metadata={"doc": "Wavelength range to use (m-arm)"}
    )
    wavelength_range_n: tuple[float, float] = dataclasses.field(
        default=(950, 1250), metadata={"doc": "Wavelength range to use (n-arm)"}
    )

    sigma_flag_threshold_b: float = dataclasses.field(
        default=0.01090,
        metadata={"doc": "Fibers are flagged if stddev over wavelength is above this threshold (b-arm)"},
    )
    sigma_flag_threshold_r: float = dataclasses.field(
        default=0.00799,
        metadata={"doc": "Fibers are flagged if stddev over wavelength is above this threshold (r-arm)"},
    )
    sigma_flag_threshold_m: float = dataclasses.field(
        default=0.00799,
        metadata={"doc": "Fibers are flagged if stddev over wavelength is above this threshold (m-arm)"},
    )
    sigma_flag_threshold_n: float = dataclasses.field(
        default=0.00799,
        metadata={"doc": "Fibers are flagged if stddev over wavelength is above this threshold (n-arm)"},
    )

    @functools.cached_property
    def fontsize_large(self) -> float:
        """Get large font size.

        Returns
        -------
        large : `float`
            Large font size in points.
        """
        return self.fontsize * 1.25

    @functools.cached_property
    def fontsize_small(self) -> float:
        """Get small font size.

        Returns
        -------
        small : `float`
            Small font size in points.
        """
        return self.fontsize * 0.85

    @functools.cached_property
    def fontsize_x_small(self) -> float:
        """Get extra-small font size.

        Returns
        -------
        x_small : `float`
            Extra-small font size in points.
        """
        return self.fontsize * 0.75

    @functools.cached_property
    def fontsize_xx_small(self) -> float:
        """Get extra-extra-small font size.

        Returns
        -------
        xx_small : `float`
            Extra-extra-small font size in points.
        """
        return self.fontsize * 0.7

    @functools.cached_property
    def wavelength_ranges(self) -> dict[ArmId, tuple[float, float]]:
        """Get a mapping from arm name to wavelength range.

        Returns
        -------
        wavelength_ranges : `dict` [`ArmId`, `tuple` [`float`, `float`]]
            Mapping from arm name to wavelength range.
        """
        return {
            "b": self.wavelength_range_b,
            "r": self.wavelength_range_r,
            "m": self.wavelength_range_m,
            "n": self.wavelength_range_n,
        }

    @functools.cached_property
    def sigma_flag_thresholds(self) -> dict[ArmId, float]:
        """Get a mapping from arm name to ``sigma_flag_threshold``.

        Returns
        -------
        sigma_flag_thresholds : `dict` [`ArmId`, `float`]
            Mapping from arm name to wavelength range.
        """
        return {
            "b": self.sigma_flag_threshold_b,
            "r": self.sigma_flag_threshold_r,
            "m": self.sigma_flag_threshold_m,
            "n": self.sigma_flag_threshold_n,
        }


class TakahashiFiberNormsQa:
    """Takahashi's FiberNormsQa.

    Parameters
    ----------
    butler : `Butler`
        Read-only butler.
    output : `str`
        Output directory path.
    input : `TakahashiFiberNormsQaInput`
        Input dataset references.
    config : `TakahashiFiberNormsQaConfig`
        Configuration.
    log : `logging.Logger` | None
        Parent logger.
    """

    def __init__(
        self,
        butler: Butler,
        output: str,
        input: TakahashiFiberNormsQaInput,
        *,
        config=TakahashiFiberNormsQaConfig(),
        log: logging.Logger | None = None,
    ) -> None:
        if log is not None:
            self.log = log.getChild(type(self).__qualname__)
        else:
            self.log = logging.getLogger(type(self).__qualname__)

        self._validate_input(input)

        self.log = self.log.getChild(
            f'[visit={input["pfsArm"][0].dataId["visit"]} arm={input["pfsArm"][0].dataId["arm"]}]'
        )

        self.log.info("Reading datasets...")

        self.output = output
        self.ref_visit = typing.cast(int, input["refArm"][0].dataId["visit"])
        self.visit = typing.cast(int, input["pfsArm"][0].dataId["visit"])
        self.arm = typing.cast(str, input["pfsArm"][0].dataId["arm"])
        self.config = config

        self.pfsConfig: PfsConfig = butler.get(input["pfsConfig"])
        self.fiberNorms: PfsFiberNorms = butler.get(input["fiberNorms"])
        self.pfsArmDict = typing.cast(
            dict[SpectrographId, PfsArm],
            {ref.dataId["spectrograph"]: butler.get(ref) for ref in input["pfsArm"]},
        )
        self.refArmDict = typing.cast(
            dict[SpectrographId, PfsArm],
            {ref.dataId["spectrograph"]: butler.get(ref) for ref in input["refArm"]},
        )

        self.spectrographs = sorted(self.pfsArmDict.keys())

        self.wavelength_range = self.config.wavelength_ranges[self.arm]

        # Threshold for odd fiber
        self.sigma_flag_entire = self.config.sigma_flag_thresholds[self.arm]

        self.random = random.Random(f"{self.visit} {self.ref_visit} {self.arm} {self.config.random_seed}")

        # These two are used in captions of figures.
        self.datastore = str(butler._config.configDir)
        self.collections = str(butler.collections.defaults)

        self._cache_pfsArmRatio: dict[SpectrographId, PfsArmRatio] = {}
        self._cache_cleaned_fiberNorms: PfsFiberNorms | None = None

    def run(self) -> None:
        """Do the task."""
        self.log.info("Computing statistics...")

        dataframe_path = os.path.join(
            self.output, f"sigma_perFiber_{self.arm}_{self.visit}_over_{self.ref_visit}.csv"
        )
        stat_path = os.path.join(
            self.output, f"sigma_perFiber_{self.arm}_{self.visit}_over_{self.ref_visit}.json"
        )
        figure_path = os.path.join(
            self.output, f"takahashiFiberNormsQa_{self.arm}_{self.visit}_over_{self.ref_visit}.png"
        )

        stat = self.get_stat()

        os.makedirs(self.output, exist_ok=True)
        self.log.info("Writing %s...", dataframe_path)
        stat.df.to_csv(dataframe_path, index=False)

        # save estimated value
        jsonobj = dataclasses.asdict(stat)
        jsonobj.pop("df")
        self.log.info("Writing %s...", stat_path)
        with open(stat_path, "w") as fcomp:
            json.dump(jsonobj, fcomp, indent="  ", ensure_ascii=False)

        try:
            matplotlib.use("agg")
            fig = self.make_plot(stat.df)
            self.log.info("Writing %s...", figure_path)
            fig.savefig(figure_path, bbox_inches="tight")
        finally:
            plt.close()

        self.log.info("Done.")

    def get_stat(self) -> TakahashiFiberNormsQaStat:
        """Get statistics.

        Returns
        -------
        stat : `TakahashiFiberNormsQaStat`
            statistics.
        """
        df = pd.DataFrame(FiberIds().data)[["fiberId", "mtp_A"]]
        # MTP group of, say, mtp_A="D3-3-4-28-26" is ""D3-3-4" (first 6 chars)
        df["mtpGroup"] = df["mtp_A"].apply(lambda mtp_A: mtp_A[0:6])
        df["arm"] = self.arm
        df["SM"] = int(0)
        df["sigma"] = np.nan

        sigma_list: list[np.ndarray[tuple[int], np.dtype[np.floating]]] = []

        for spec in self.spectrographs:
            pfsArmRatio = self._get_pfsArmRatio(spec)

            pfsArm = pfsArmRatio.pfsArm
            ratio = pfsArmRatio.ratio

            df.loc[df["fiberId"].isin(pfsArm.fiberId), "SM"] = spec

            # per-fiber standard deviation with respect to wavelength.
            sigma = np.nanstd(ratio, axis=1)
            sigma_list.append(sigma)

            q3, q1 = np.nanpercentile(ratio, [75, 25], axis=1)
            sigma_iqr = 0.741 * (q3 - q1)

            # mineo: Is it guaranteed that the orders of fiberId on both sides of "=" are equal?
            df.loc[df["fiberId"].isin(pfsArm.fiberId), ["sigma"]] = sigma_iqr

        arr = np.concatenate(sigma_list)

        return TakahashiFiberNormsQaStat(
            hst=self._get_metadata("HST"),
            insrot=self._get_metadata("INSROT"),
            azimuth=self._get_metadata("AZIMUTH"),
            sigma_std_median=float(np.median(arr)),
            sigma_std_95=float(np.nanpercentile(arr, 95)),
            twosigma_std=float(np.median(arr * 2)),
            twosigma_std_95=float(np.nanpercentile(arr * 2, 95)),
            sigma_iqr=float(np.nanmedian(df["sigma"])),
            df=df,
        )

    def _get_metadata(self, key: str) -> Any:
        """Get pfsArm's metadata.

        It is assumed that all ``pfsArm`` in ``self.pfsArmDict``
        have the same value for ``key``.

        Parameters
        ----------
        key : `str`
            Name of the metadata to get.

        Returns
        -------
        value : `Any`
            Metadata value.
        """
        for pfsArm in self.pfsArmDict.values():
            # Any pfsArm will do.
            return pfsArm.metadata[key]
        raise RuntimeError("self.pfsArmDict is empty.")

    def _get_pfsArmRatio(self, spec: SpectrographId) -> "PfsArmRatio":
        """Get an instance of `PfsArmRatio`.

        The return value will be cached.

        Returns
        -------
        pfsArmRatio : `PfsArmRatio`
            Ratio of a pfsArm to another pfsArm.
        """
        pfsArmRatio = self._cache_pfsArmRatio.get(spec)
        if pfsArmRatio is None:
            pfsArmRatio = self._cache_pfsArmRatio[spec] = PfsArmRatio(
                self.refArmDict[spec], self.pfsArmDict[spec], self.pfsConfig, self.wavelength_range
            )

        return pfsArmRatio

    def _get_cleaned_fiberNorms(self) -> PfsFiberNorms:
        """Clean ``self.fiberNorms`` and get it.

        The return value will be cached.

        Returns
        -------
        fiberNorms : `PfsFiberNorms`
            Cleaned fiberNorms.
        """
        cache = self._cache_cleaned_fiberNorms
        if cache is not None:
            return cache

        wmin, wmax = self.wavelength_range
        pfsConfig = self.pfsConfig.select(fiberStatus=FiberStatus.GOOD)

        fiberNorms = self.fiberNorms
        fiberNorms = fiberNorms[np.isin(fiberNorms.fiberId, pfsConfig.fiberId)]
        fiberNorms.values = np.where(
            (wmin <= fiberNorms.wavelength) & (fiberNorms.wavelength <= wmax), fiberNorms.values, np.nan
        )
        fiberNorms.values = np.where(np.isinf(fiberNorms.values), np.nan, fiberNorms.values)

        self._cache_cleaned_fiberNorms = fiberNorms
        return fiberNorms

    @staticmethod
    def _validate_input(input: TakahashiFiberNormsQaInput) -> None:
        """Check the consistency of input dataset references.

        Parameters
        ----------
        input : `TakahashiFiberNormsQaInput`
            Input dataset references.

        Raises
        ------
        RuntimeError
            If ``input`` is inconsistent.
        """
        if len(input["pfsArm"]) != len(input["refArm"]):
            raise RuntimeError("number of input pfsArms is different from that of refArms.")
        if not input["pfsArm"]:
            raise RuntimeError("number of input pfsArms is zero.")

        instrument = input["pfsArm"][0].dataId["instrument"]
        visit = input["pfsArm"][0].dataId["visit"]
        arm = input["pfsArm"][0].dataId["arm"]

        if (
            any(instrument != ref.dataId["instrument"] for ref in input["pfsArm"])
            or any(instrument != ref.dataId["instrument"] for ref in input["refArm"])
            or instrument != input["pfsConfig"].dataId["instrument"]
            or instrument != input["fiberNorms"].dataId["instrument"]
        ):
            raise RuntimeError("instruments of input datasets are not unique.")

        if (
            any(visit != ref.dataId["visit"] for ref in input["pfsArm"])
            or visit != input["pfsConfig"].dataId["visit"]
            or visit != input["fiberNorms"].dataId["visit"]
        ):
            raise RuntimeError("visits of input datasets are not unique.")

        if (
            any(arm != ref.dataId["arm"] for ref in input["pfsArm"])
            or any(arm != ref.dataId["arm"] for ref in input["refArm"])
            or arm != input["fiberNorms"].dataId["arm"]
        ):
            raise RuntimeError("arms of input datasets are not unique.")

        ref_visit = input["refArm"][0].dataId["visit"]
        if any(ref_visit != ref.dataId["visit"] for ref in input["refArm"]):
            raise RuntimeError("reference visits of input datasets are not unique.")

        specs = set(ref.dataId["spectrograph"] for ref in input["pfsArm"])
        if len(specs) != len(input["pfsArm"]):
            raise RuntimeError("there are duplicate pfsArms in the input.")

        ref_specs = set(ref.dataId["spectrograph"] for ref in input["refArm"])
        if specs != ref_specs:
            raise RuntimeError("set of spectrographs of pfsArms is different from that of refArms.")

    def make_plot(self, df: pd.DataFrame) -> plt.Figure:
        """Make a figure with matplotlib.

        Parameters
        ----------
        df : `pd.DataFrame`
            Data frame. The following columns must exist:

            - fiberId : `int`
                Fiber ID.
            - mtp_A : `str`
                MTP ID on Spectrograph System
            - mtpGroup : `str`
                MTP group.
            - arm : `ArmId`
                Arm name.
            - SM : `SpectrographId` | `Literal` [0]
                Spectrograph name. (Invalid if zero)
            - sigma : `float`
                Per-fiber standard deviation with respect to wavelength.

        Returns
        -------
        figure : `plt.Figure`
            Figure.
        """
        df = df[df["mtpGroup"].str.contains("U") | df["mtpGroup"].str.contains("D")]
        df = df[df["SM"] != 0]
        df = df.reset_index()
        ins = self._get_metadata("INSROT")
        azi = np.round(self._get_metadata("AZIMUTH"), 2)

        fig = plt.figure(
            num="fiberNormsQA", figsize=self.config.figsize, clear=True, facecolor="w", edgecolor="k"
        )

        # Number of columns of odd fibers shown.
        ncols_odd = 5
        # Number of rows of odd fibers shown.
        nrows_odd = 3

        # extract odd MTPs/fibers
        df_odd = df[df["sigma"] > self.sigma_flag_entire]
        # select a number of odd fibers with the largest sigma.
        df_odd = df_odd.sort_values("sigma", ascending=False)
        df_odd = df_odd[0 : ncols_odd * nrows_odd]
        df_odd = df_odd.reset_index()
        n_odd = len(df_odd)

        nrows = 4 + 4  # last four rows are for showing spectra

        gs_master = gridspec.GridSpec(
            nrows=nrows, ncols=6, height_ratios=np.full(nrows, 1), wspace=0.7, hspace=0.5
        )

        ax5 = fig.add_subplot(gs_master[1, 0], aspect="equal")  # --For PFI image with pfsArm
        ax6 = fig.add_subplot(gs_master[1, 1], aspect="equal")  # --For PFI image with pfsArm
        ax7 = fig.add_subplot(gs_master[2, 0], aspect="equal")  # --For PFI image with pfsArm
        ax8 = fig.add_subplot(gs_master[2, 1], aspect="equal")  # --For PFI image with pfsArm
        ax9 = fig.add_subplot(gs_master[3, 0], aspect="equal")  # --For PFI image with fiberNorms
        ax10 = fig.add_subplot(gs_master[3, 1])  # --For PFI image with fiberNorms
        # For 2D spectral image
        ax13 = fig.add_subplot(gs_master[1:3, 2:5])
        # For sigma per fiber
        ax14 = fig.add_subplot(gs_master[3, 2:5])
        # MTP group vs. spectrograph ID
        ax15 = fig.add_subplot(gs_master[1:4, 5])

        # [5] Measured sigma for each fiber.
        self._make_plot_sigma_per_fiber(ax14, df)

        # [3] Used MTPs compared with corresponding spectrograph ID
        self._make_plot_used_mtps(ax15, df)

        # [2] 2D spectrum of quartz ratio.
        self._make_plot_quartz_ratio_by_wavelength(ax13, df)

        # [1] PFI IMAGES (pfsArm.flux/pfsArm.norm)
        fig.text(0.15, 0.78, "[1] Quartz ratio measured from pfsArm", fontsize=self.config.fontsize)
        self._make_plot_quartz_ratio_by_position_n_sigma(ax5, df, n=2)
        self._make_plot_quartz_ratio_by_position_median(ax6)
        self._make_plot_quartz_ratio_by_position_at_pixel(ax7, pixel_index=1500)
        self._make_plot_quartz_ratio_by_position_at_pixel(ax8, pixel_index=3500)

        # [4] PFI IMAGES (fiberNorms)
        fig.text(0.15, 0.59, "[4] fiberNorms.values of target quartz", fontsize=self.config.fontsize)
        self._make_plot_fiberNorms_by_position(ax9)
        self._make_plot_fiberNorms_by_wavelength(ax10)

        is_dirty = (self.visit != self.ref_visit) and (len(df_odd) > 0)

        if is_dirty:
            # [6] 1D spectra measured from pfsArm
            gs_spec = gridspec.GridSpecFromSubplotSpec(
                nrows=nrows_odd, ncols=ncols_odd, subplot_spec=gs_master[4:7, :], wspace=0.3, hspace=0.5
            )
            fig.text(
                0.35,
                0.49,
                "[6] Example spectra for flagged fibers with large flux scatters (red marked in [2])",
                fontsize=self.config.fontsize,
            )
            for i in range(n_odd):
                axs = fig.add_subplot(gs_spec[i // ncols_odd, i % ncols_odd])
                self._make_plot_1d_spectra(axs, df_odd.iloc[i])

            # [7] 1D spectra measured from fiberNorms
            gs_spec2 = gridspec.GridSpecFromSubplotSpec(
                nrows=1, ncols=ncols_odd, subplot_spec=gs_master[7:, :], wspace=0.3, hspace=0.5
            )
            fig.text(
                0.35,
                0.19,
                "[7] Randomly selected spectra obtained by fiberNorms.values",
                fontsize=self.config.fontsize,
            )
            # extract fiberIds for plotting example spectra
            list_ex_fibernorms = sorted(self.random.sample(list(df["fiberId"]), ncols_odd))
            for i in range(ncols_odd):
                axs = fig.add_subplot(gs_spec2[i // ncols_odd, i % ncols_odd])
                self._make_plot_example_fiberNorm(axs, df, list_ex_fibernorms[i])

        sigma_typ = np.nanmedian(df["sigma"])

        if sigma_typ >= self.sigma_flag_entire:
            flag = "Yes"
            strcolor = "red"
        else:
            flag = "No"
            strcolor = "black"

        obsdate = utc2hst(self.fiberNorms.metadata["DATEOBS"])

        fig.text(0.1, 0.85, "fiberNormsQA ver. 1.0", fontsize=self.config.fontsize_x_small)
        fig.text(
            0.7,
            0.85,
            f"Flag for fiber throughput variation={flag}",
            color=strcolor,
            fontsize=self.config.fontsize_large,
            bbox=dict(facecolor="yellow", alpha=1.0),
        )
        fig.text(
            0.1,
            0.83,
            f"visit_target={self.visit}",
            fontweight="bold",
            fontsize=self.config.fontsize_large,
        )
        fig.text(
            0.1,
            0.82,
            f"visit_reference={self.ref_visit}, arm={self.arm}, obsdate={obsdate},"
            + f" insrot={ins}deg, azimuth={azi}deg",
            fontsize=self.config.fontsize_large,
        )
        pfs_pipe2d = eups.getSetupVersion("pfs_pipe2d")
        fig.text(
            0.1,
            0.81,
            f"datastore={self.datastore}, collections={self.collections}, pfs_pipe2d={pfs_pipe2d}",
            fontsize=self.config.fontsize_large,
        )

        footnote: str = (
            "fiberNormsQA is to monitor fiber throughput variation. "
            "We took quartz flux ratios for some of the figures "
            "to investigate how much quartz flux\n varies with time, "
            "i.e., quartz ratio = pfsArm.flux(visit_target)/pfsArm.flux(visit_reference). "
            "Referenced quartz is the first one in the data set \n basically "
            "(see visit number above).\n"
            "Sigma is measured from 0.741*(Q75-Q25). "
            "Descriptions for each sub-fig component are as follows.\n"
            "[1] PFI image of quartz ratios. "
            "2 sigma per fier and median flux ratio per fiber are represent "
            "in the upper left and in the upper right figure, \n respectively. "
            "We also represent median flux ratios at two different wavelength point via PFI images.\n"
            "[2] 2D spectrum of quartz ratio. If measure sigma of a fiber is larger than "
            f"{self.sigma_flag_entire}, red closs marks fiber Id in the left side.\n"
            "[3] Used MTPs compared with corresponding spectrograph ID. "
            "The MTP groups in which all MTPs were used were represented in green.\n"
            "[4] fiberNorms.values of target quartz. "
            "Plotting bounds are 2.5$\\sigma$\n"
            "[5] Measured sigma for each fiber. "
            "The median values in all MTP groups represent in blue dashed line "
            "and 4 sigma \nrepresent in red dashed line. "
            "This figure is to check flux variation with MTP unit. "
            "MTP groups above the red dashed line, \n "
            "indicating that the MTP group has a large flux variation. \n"
        )

        if is_dirty:
            footnote += (
                "[6] Spectra of FIBER with large sigma with a maximum of 15. "
                "Blue plots represent normalized spectra obtained by quartz ratio \n "
                "(i.e. pfsArm.flux(target)/pfsArm.norm(target)/"
                "pfsArm.flux(reference)/pfsArm.norm(reference)) "
                "and light green plots represent median filtered \n spectra.\n"
                "FiberId are shown in upper left. "
                "Mesured sigma per fiber are shown in upper right. "
                "Red dashed lines represent 2sigma lines of median filtered spectra.\n"
                "[7] Randomly selected spectra obtained from fiberNorms.values of target quartz"
            )

        if is_dirty:
            plt.annotate(
                footnote,
                (-5.5, -2.5),
                fontsize=self.config.fontsize_large,
                xycoords="axes fraction",
            )
        else:
            plt.annotate(
                footnote,
                (-95, -1.5),
                fontsize=self.config.fontsize_large,
                xycoords="axes fraction",
            )

        fig.tight_layout()
        return fig

    def _make_plot_sigma_per_fiber(self, ax: matplotlib.axes.Axes, df: pd.DataFrame) -> None:
        """Plot measured sigma for each fiber

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        df : `pd.DataFrame`
            Data frame. The following columns must exist:

            - fiberId : `int`
                Fiber ID.
            - mtpGroup : `str`
                MTP group.
            - sigma : `float`
                Per-fiber standard deviation with respect to wavelength.
        """
        ax.set_title("[5] Sigma per fiber", fontsize=self.config.fontsize)
        ax.scatter(df["mtpGroup"], df["sigma"], c="black")

        sigma_median = np.nanmedian(df["sigma"])
        ax.axhline(sigma_median, ls="dashed", c="blue", zorder=0)
        ax.annotate(
            "median",
            (0.9, 0.1),
            c="blue",
            fontsize=self.config.fontsize_xx_small,
            xycoords="axes fraction",
        )
        clip = 4
        sigma_of_sigma = sigma_median + clip * np.std(df["sigma"])
        ax.axhline(sigma_of_sigma, ls="dashed", c="red", zorder=0)
        ax.text("D2-1-4", 1.2 * sigma_of_sigma, "4 sigma", c="red", fontsize=self.config.fontsize_xx_small)
        ax.text(
            0.85,
            0.9,
            rf"${clip}\sigma$ = {sigma_of_sigma:.4f}",
            color="red",
            transform=ax.transAxes,
            fontsize=self.config.fontsize_small,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax.text(
            0.85,
            0.8,
            f"median = {sigma_median:.4f}",
            color="blue",
            transform=ax.transAxes,
            fontsize=self.config.fontsize_small,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        ax.grid(axis="both", which="both", linestyle="--", linewidth=0.5)
        ax.set_xlabel("MTP group", fontsize=self.config.fontsize_x_small)
        ax.set_ylabel(r"$\sigma$ (0.741*(Q75-Q25))", fontsize=self.config.fontsize_x_small)
        ax.tick_params("x", labelrotation=90)

    def _make_plot_used_mtps(self, ax: matplotlib.axes.Axes, df: pd.DataFrame) -> None:
        """Plot used MTPs compared with corresponding spectrograph ID.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        df : `pd.DataFrame`
            Data frame. The following columns must exist:

            - fiberId : `int`
                Fiber ID.
            - mtpGroup : `str`
                MTP group.
            - SM : `SpectrographId` | `Literal` [0]
                Spectrograph name. (Invalid if zero)
        """
        ax.set_title("[3] Used MTPs", fontsize=self.config.fontsize)
        xdata, ydata = df["SM"][df["SM"] != 0], df["mtpGroup"][df["SM"] != 0]
        ax.scatter(xdata, ydata, c="olivedrab")

        ax.grid(axis="both", which="both", linestyle="--", linewidth=0.5)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlabel("spectrograph ID", fontsize=self.config.fontsize)
        ax.set_ylabel("MTP group", fontsize=self.config.fontsize_x_small)

    def _make_plot_quartz_ratio_by_wavelength(self, ax: matplotlib.axes.Axes, df: pd.DataFrame) -> None:
        """Plot 2D spectrum of quartz ratio.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        df : `pd.DataFrame`
            Data frame. The following columns must exist:

            - fiberId : `int`
                Fiber ID.
            - SM : `SpectrographId` | `Literal` [0]
                Spectrograph name. (Invalid if zero)
            - sigma : `float`
                Per-fiber standard deviation with respect to wavelength.
        """
        # [2] 2D spec of quartz ratio
        ax.set_title(
            "[2] 2D spectrum of quartz ratio measured from pfsArm.flux", fontsize=self.config.fontsize
        )
        ax.set_xlabel("wavelength [nm]", fontsize=self.config.fontsize_large)
        ax.set_ylabel("fiberId", fontsize=self.config.fontsize_large)

        wmin, wmax = self.wavelength_range

        for spec in self.spectrographs:
            pfsArmRatio = self._get_pfsArmRatio(spec)
            pfsArm = pfsArmRatio.pfsArm
            ratio = pfsArmRatio.ratio
            fibs = np.array(
                [
                    np.full_like(pfsArm.wavelength[np.where(pfsArm.fiberId == f)[0][0]], f)
                    for f in pfsArm.fiberId
                ]
            )

            # plot quartz ratio as contour (fiberId vs wavelength)
            sc = ax.scatter(
                pfsArm.wavelength,
                fibs,
                c=ratio,
                vmin=self.config.vmin,
                vmax=self.config.vmax,
                s=0.6,
                alpha=1.0,
                label="quartz",
            )

            oddfib = df["fiberId"][(df["SM"] == spec) & (df["sigma"] > self.sigma_flag_entire)].values
            for i_fib in oddfib:
                ax.scatter(wmin, i_fib, marker="+", color="red", s=150, alpha=0.8)

        divider = make_axes_locatable(ax)  # AxesDivider related to ax
        cax = divider.append_axes("right", size="2%", pad=0.1)  # create new axes

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, cax=cax)

    def _make_plot_quartz_ratio_by_position_n_sigma(
        self, ax: matplotlib.axes.Axes, df: pd.DataFrame, n: float
    ) -> None:
        """Plot PFI image of quartz ratios (2 sigma per fier).

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        df : `pd.DataFrame`
            Data frame. The following columns must exist:

            - fiberId : `int`
                Fiber ID.
            - SM : `SpectrographId` | `Literal` [0]
                Spectrograph name. (Invalid if zero)
            - sigma : `float`
                Per-fiber standard deviation with respect to wavelength.
        n : `float`
            ``n`` of ``n * sigma``.
        """
        for spec in self.spectrographs:
            pfsconfig = self._get_pfsArmRatio(spec).pfsConfig
            sc = ax.scatter(
                pfsconfig.pfiCenter[:, 0],
                pfsconfig.pfiCenter[:, 1],
                c=np.array(df["sigma"][df["SM"] == spec]) * n,
                vmin=self.config.vmin2,
                vmax=self.config.vmax2,
                s=30.0,
                alpha=1.0,
                label=f"{n}sigma",
            )
            ax.set_xlim(xmin=-250, xmax=250)
            ax.set_ylim(ymin=-250, ymax=250)
            ax.yaxis.set_ticks_position("left")
            ax.set_title(f"{n}sigma", fontsize=self.config.fontsize)
            ax.set_xlabel("X(PFI) [mm]", fontsize=self.config.fontsize)
            ax.set_ylabel("Y(PFI) [mm]", fontsize=self.config.fontsize)

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, ax=ax, location="right", fraction=0.04, alpha=1.0)

    def _make_plot_quartz_ratio_by_position_median(self, ax: matplotlib.axes.Axes) -> None:
        """Plot PFI image of quartz ratios (median flux ratio per fiber).

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        """
        for spec in self.spectrographs:
            pfsArmRatio = self._get_pfsArmRatio(spec)
            ratio = pfsArmRatio.ratio
            pfsconfig = pfsArmRatio.pfsConfig

            # Plot median
            median_array = np.array([np.nanmedian(ratio[i_fiber]) for i_fiber in range(len(ratio))])
            sc = ax.scatter(
                pfsconfig.pfiCenter[:, 0],
                pfsconfig.pfiCenter[:, 1],
                c=median_array,
                vmin=self.config.vmin,
                vmax=self.config.vmax,
                s=30.0,
                alpha=1.0,
                label="median per fiber",
            )
            ax.set_xlim(xmin=-250, xmax=250)
            ax.set_ylim(ymin=-250, ymax=250)
            ax.yaxis.set_ticks_position("left")
            ax.set_title("median per fiber", fontsize=self.config.fontsize)
            ax.set_xlabel("X(PFI) [mm]", fontsize=self.config.fontsize)
            ax.set_ylabel("Y(PFI) [mm]", fontsize=self.config.fontsize)

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, ax=ax, location="right", fraction=0.04, alpha=1.0)

    def _make_plot_quartz_ratio_by_position_at_pixel(
        self, ax: matplotlib.axes.Axes, pixel_index: int
    ) -> None:
        """Plot PFI image of quartz ratios (specific wavelength).

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        pixel_index : `int`
            Index in the wavelength array.
        """
        for spec in self.spectrographs:
            pfsArmRatio = self._get_pfsArmRatio(spec)
            ratio = pfsArmRatio.ratio
            pfsconfig = pfsArmRatio.pfsConfig
            pfsArm = pfsArmRatio.pfsArm

            ratio_lam = np.array([ratio[i_fiber][pixel_index] for i_fiber in range(len(ratio))])
            sc = ax.scatter(
                pfsconfig.pfiCenter[:, 0],
                pfsconfig.pfiCenter[:, 1],
                c=ratio_lam,
                vmin=self.config.vmin,
                vmax=self.config.vmax,
                s=30.0,
                alpha=1.0,
            )
            ax.set_xlim(xmin=-250, xmax=250)
            ax.set_ylim(ymin=-250, ymax=250)
            ax.yaxis.set_ticks_position("left")
            lam_point = np.round(pfsArm.wavelength[0][pixel_index], 3)
            ax.set_title(f"at {lam_point} [nm]", fontsize=self.config.fontsize)
            ax.set_xlabel("X(PFI) [mm]", fontsize=self.config.fontsize)
            ax.set_ylabel("Y(PFI) [mm]", fontsize=self.config.fontsize)

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, ax=ax, location="right", fraction=0.04, alpha=1.0)

    def _make_plot_fiberNorms_by_position(self, ax: matplotlib.axes.Axes) -> None:
        """Plot fiberNorms of the target quartz (median per fiber).

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        """
        pfsConfig = self.pfsConfig.select(fiberStatus=FiberStatus.GOOD)
        fiberNorms = self._get_cleaned_fiberNorms()
        fiberNorms.plot(pfsConfig, axes=ax, lower=2.5, upper=2.5)

        ax.set_title("median per fiber", fontsize=self.config.fontsize)
        ax.set_xlabel("X(PFI) [mm]", fontsize=self.config.fontsize)
        ax.set_ylabel("Y(PFI) [mm]", fontsize=self.config.fontsize)

    def _make_plot_fiberNorms_by_wavelength(self, ax: matplotlib.axes.Axes) -> None:
        """Plot fiberNorms of the target quartz (2D spectrum).

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        """
        # [4] 2D image of fiberNorms.values
        fiberNorms = self._get_cleaned_fiberNorms()

        values = np.nanmedian(fiberNorms.values, axis=1)
        good = np.isfinite(values)
        median = np.median(values[good])
        lower = median - 2.5 * np.std(values)
        upper = median + 2.5 * np.std(values)
        fib_norm = np.array(
            [
                np.full_like(fiberNorms.wavelength[np.where(fiberNorms.fiberId == f)[0][0]], f)
                for f in fiberNorms.fiberId
            ]
        )
        sc = ax.scatter(
            fiberNorms.wavelength,
            fib_norm,
            c=fiberNorms.values,
            vmin=lower,
            vmax=upper,
            s=10,
            cmap="coolwarm",
        )
        divider = make_axes_locatable(ax)  # AxesDivider related to ax
        cax = divider.append_axes("right", size="5%", pad=0.1)  # create new axes

        ax.set_ylabel("fiber index", fontsize=self.config.fontsize_small)
        ax.set_xlabel("wavelength", fontsize=self.config.fontsize_small)
        ax.set_title("2D spectrum", fontsize=self.config.fontsize)

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, cax=cax)

    def _make_plot_1d_spectra(self, ax: matplotlib.axes.Axes, df_row: pd.Series) -> None:
        """Plot a normalized spectra.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        df_row : `pd.Series`
            A row of an data frame. The following fields must exist:

            - fiberId : `int`
                Fiber ID.
            - mtp_A : `str`
                MTP ID on Spectrograph System
            - SM : `SpectrographId` | `Literal` [0]
                Spectrograph name. (Invalid if zero)
            - sigma : `float`
                Per-fiber standard deviation with respect to wavelength.
        """
        spec = df_row["SM"]

        pfsArmRatio = self._get_pfsArmRatio(spec)
        pfsArm = pfsArmRatio.pfsArm

        ratio = pfsArmRatio.flattened_ratio
        ratio_mf = pfsArmRatio.filtered_flattened_ratio

        fib = df_row["fiberId"]
        inx = np.where(pfsArm.fiberId == fib)[0]
        n_sigma = 2

        xdata = pfsArm.wavelength[inx]
        ydata = ratio[inx]
        ax.scatter(xdata, ydata, color="royalblue", s=10, label=f"{fib}")

        xdata_m = pfsArm.wavelength[inx]
        ydata_m = ratio_mf[inx]
        std = np.nanstd(ydata_m)
        ax.scatter(xdata_m, ydata_m, color="limegreen", s=10, label=f"{fib}(mf)")

        ax.text(
            0.68,
            0.9,
            rf'$\sigma$ ={df_row["sigma"]:.4f}',
            transform=ax.transAxes,
            fontsize=self.config.fontsize_small,
            bbox=dict(facecolor="yellow", alpha=1.0),
        )
        ax.axhline(
            y=np.nanmedian(ydata_m) + n_sigma * df_row["sigma"],
            color="tomato",
            linestyle="dashed",
        )
        ax.axhline(
            y=np.nanmedian(ydata_m) - n_sigma * df_row["sigma"],
            color="tomato",
            linestyle="dashed",
        )

        ax.set(xlabel="wavelength [nm]", ylabel="normalized flux")

        ax.set_ylim(
            ymin=np.nanmedian(ydata_m) - (n_sigma + 6) * std,
            ymax=np.nanmedian(ydata_m) + (n_sigma + 6) * std,
        )
        ax.legend(loc="upper left", fontsize=self.config.fontsize_x_small)
        ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5)

        ax.set_title(f'{df_row["mtp_A"]}', fontsize=self.config.fontsize)

    def _make_plot_example_fiberNorm(self, ax: matplotlib.axes.Axes, df: pd.DataFrame, fiber_id: int) -> None:
        """Plot a spectrum in fiberNorms.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes.
        df : `pd.DataFrame`
            Data frame. The following columns must exist:

            - fiberId : `int`
                Fiber ID.
            - mtp_A : `str`
                MTP ID on Spectrograph System
        fiber_id : `int`
            Fiber ID.
        """
        fiberNorms = self._get_cleaned_fiberNorms()

        inx_nrm = np.where(fiberNorms.fiberId == fiber_id)[0]
        ax.scatter(
            fiberNorms.wavelength[inx_nrm],
            fiberNorms.values[inx_nrm],
            color="navy",
            s=10,
            label=f"{fiber_id}",
        )

        xdata_m = fiberNorms.wavelength[inx_nrm]
        xdata_m = xdata_m[~np.isnan(fiberNorms.values[inx_nrm])]
        ydata_m = medfilt(fiberNorms.values[inx_nrm][~np.isnan(fiberNorms.values[inx_nrm])], kernel_size=15)

        std = np.nanstd(ydata_m)
        ax.scatter(xdata_m, ydata_m, color="limegreen", s=10, label=f"{fiber_id}(mf)")

        n_sigma = 2

        ax.axhline(y=np.nanmedian(ydata_m) + n_sigma * std, color="red", linestyle="dashed")
        ax.axhline(y=np.nanmedian(ydata_m) - n_sigma * std, color="red", linestyle="dashed")

        ax.set(xlabel="wavelength [nm]", ylabel="normalized flux")
        ax.set_ylim(
            ymin=np.nanmedian(ydata_m) - (n_sigma + 2) * std,
            ymax=np.nanmedian(ydata_m) + (n_sigma + 2) * std,
        )
        ax.legend(loc="upper left", fontsize=self.config.fontsize_x_small)
        ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5)

        ax.set_title(
            f'{df["mtp_A"][df["fiberId"]==fiber_id].to_string(index=False)}',
            fontsize=self.config.fontsize,
        )


class PfsArmRatio:
    """Ratio of a pfsArm to another pfsArm.

    Parameters
    ----------
    refArm : `PfsArm`
        Reference, or denominator.
    pfsArm : `PfsArm`
        Numerator.
    pfsConfig : `PfsConfig`
        Top-end fiber configuration.
    wavelength_range : `tuple` [`float`, `float`]
        Wavelength range to use.

    Attributes
    ----------
    refArm : `PfsArm`
        Reference pfsArm (denominator).
        Only those fibers present in ``pfsConfig`` are retained.
    pfsArm : `PfsArm`
        pfsArm (numerator).
        Only those fibers present in ``pfsConfig`` are retained.
    pfsConfig : `PfsConfig`
        Top-end fiber configuration.
        Only those fibers belonging to the same spectrograph as that of
        ``pfsArm`` are retained.
    """

    def __init__(
        self, refArm: PfsArm, pfsArm: PfsArm, pfsConfig: PfsConfig, wavelength_range: tuple[float, float]
    ) -> None:
        pfsConfig = pfsConfig.select(targetType=~TargetType.ENGINEERING)
        pfsConfig = pfsConfig.select(fiberStatus=FiberStatus.GOOD)
        pfsConfig = pfsConfig.select(spectrograph=pfsArm.identity.spectrograph)

        refArm = refArm[np.isin(refArm.fiberId, pfsConfig.fiberId)]
        pfsArm = pfsArm[np.isin(pfsArm.fiberId, pfsConfig.fiberId)]

        wmin, wmax = wavelength_range

        self.pfsConfig = pfsConfig
        self.refArm = refArm
        self.pfsArm = pfsArm
        self._wmin = wmin
        self._wmax = wmax

    @functools.cached_property
    @ignore_numpy_warnings
    def ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Ratio of a pfsArm to another pfsArm.

        The return value is not a naive ratio but is:

            naive[fiber][lam] * mean(naive)
                / (mean_over_fiber(naive)[lam] * mean_over_lam(naive)[fiber])

        for every fiber and lam; where

            naive[fiber][lam]
                = pfsArm.flux[fiber][lam] / refArm.flux[fiber][lam].

        Returns
        -------
        ratio : `np.ndarray` [`tuple` [`int`, `int`], `np.dtype` [`np.floating`]]
            Normalized flux ratio. Shape (``num_fibers``, ``num_wavelengths``).
        """
        return self._normalize_2d_array(self._naive_ratio)

    @functools.cached_property
    @ignore_numpy_warnings
    def flattened_ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Ratio of a pfsArm (divided by norm)
        to another pfsArm (divided by norm).

        The return value is not a naive ratio but is:

            naive[fiber][lam] * mean(naive)
                / (mean_over_fiber(naive)[lam] * mean_over_lam(naive)[fiber])

        for every fiber and lam; where

            naive[fiber][lam]
                = (pfsArm.flux[fiber][lam] / norm)
                    / (refArm.flux[fiber][lam] / norm).

        Returns
        -------
        flattened_ratio : `np.ndarray` [`tuple` [`int`, `int`], `np.dtype` [`np.floating`]]
            Normalized ratio of (flux / norm).
            Shape (``num_fibers``, ``num_wavelengths``).
        """
        return self._normalize_2d_array(self._naive_flattened_ratio)

    @functools.cached_property
    @ignore_numpy_warnings
    def filtered_flattened_ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Ratio of a pfsArm (divided by norm)
        to another pfsArm (divided by norm).

        A median filter is applied.

        The return value is not a naive ratio but is:

            naive[fiber][lam] * mean(naive)
                / (mean_over_fiber(naive)[lam] * mean_over_lam(naive)[fiber])

        for every fiber and lam; where

            naive[fiber][lam]
                = (pfsArm.flux[fiber][lam] / norm)
                    / (refArm.flux[fiber][lam] / norm),

        with median filter applied.

        Returns
        -------
        filtered_flattened_ratio : `np.ndarray` [`tuple` [`int`, `int`], `np.dtype` [`np.floating`]]
            Normalized ratio of (flux / norm), filtered.
            Shape (``num_fibers``, ``num_wavelengths``).
        """
        ratio = medfilt(self._naive_flattened_ratio, kernel_size=(1, 15))
        return self._normalize_2d_array(ratio)

    @functools.cached_property
    @ignore_numpy_warnings
    def _naive_ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Naive ratio of a pfsArm to another pfsArm.

        Returns
        -------
        ratio : `np.ndarray` [`tuple` [`int`, `int`], `np.dtype` [`np.floating`]]
            Flux ratio.
            Shape (``num_fibers``, ``num_wavelengths``).
        """
        ratio = self.pfsArm.flux / self.refArm.flux
        # discard around the edge of wavelength (noisy)
        ratio = np.where(
            (self._wmin <= self.pfsArm.wavelength) & (self.pfsArm.wavelength <= self._wmax), ratio, np.nan
        )
        ratio = np.where(np.isinf(ratio), np.nan, ratio)
        return ratio

    @functools.cached_property
    @ignore_numpy_warnings
    def _naive_flattened_ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Naive ratio of a pfsArm (divided by norm)
        to another pfsArm (divided by norm).

        Returns
        -------
        flattened_ratio : `np.ndarray` [`tuple` [`int`, `int`], `np.dtype` [`np.floating`]]
            Ratio of (flux / norm).
            Shape (``num_fibers``, ``num_wavelengths``).
        """
        ratio = (self.pfsArm.flux / self.pfsArm.norm) / (self.refArm.flux / self.refArm.norm)
        # discard around the edge of wavelength (noisy)
        ratio = np.where(
            (self._wmin <= self.pfsArm.wavelength) & (self.pfsArm.wavelength <= self._wmax), ratio, np.nan
        )
        ratio = np.where(np.isinf(ratio), np.nan, ratio)
        return ratio

    @staticmethod
    @ignore_numpy_warnings
    def _normalize_2d_array(
        arr: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Normalize 2D array ``arr``.

            arr'[i][j] = arr[i][j] * mean(arr)
                            / (mean_over_i(arr)[j] * mean_over_j(arr)[i])

        Means are computed robustly.

        Parameters
        ----------
        arr : `np.ndarray` [`tuple` [`int`, `int`], `np.dtype` [`np.floating`]]
            2D array.

        Returns
        -------
        normalized_arr : `np.ndarray` [`tuple` [`int`, `int`], `np.dtype` [`np.floating`]]
            Normalized 2D array.
        """
        mean_over_i = np.nanmedian(arr, axis=0, keepdims=True)
        mean_over_j = np.nanmedian(arr, axis=1, keepdims=True)
        mean_over_j /= np.nanmedian(arr, keepdims=True)
        return arr / (mean_over_i * mean_over_j)


def utc2hst(utc_str: str) -> str:
    """Convert UTC to HST

    Parameters
    ----------
    utc_str : `str`
        UTC in ISO format.

    Returns
    -------
    hst : `str`
        HST in ISO format (``YYYY-mm-ddTHH:MM:SS.S``)
    """
    # ISO to datetime
    utc_dt = datetime.datetime.fromisoformat(utc_str.replace("Z", "+00:00"))

    # set time zone
    utc_dt = pytz.utc.localize(utc_dt) if utc_dt.tzinfo is None else utc_dt

    # convert to the hst time zone (UTC-10:00)
    hst_tz = pytz.timezone("US/Hawaii")
    hst_dt = utc_dt.astimezone(hst_tz)

    hst_formatted = hst_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    hst_formatted = hst_formatted[:-5]

    return hst_formatted


if __name__ == "__main__":
    sys.exit(main())
