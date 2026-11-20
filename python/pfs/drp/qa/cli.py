"""``drpQa.py``: one command, with a subcommand for each QA job.

Each subcommand is a function in the module that does the job, imported only
when it is asked for, so those that read only qadb (``values``, ``judge``,
``summary``) run without the LSST stack.

=================== ====================================================================================
``load``            Copy a collection's QA numbers from the Butler to qadb and judge them (qaActor, after
                    each visit). A twilight sequence is measured and judged once it is complete.
``values``          Make, compare and accept a run's normal values, then judge the frames they cover.
``judge``           Judge stored numbers again, for a visit range of a collection.
``summary``         Give the verdict of each visit of a collection, with an exit status.
``calib inputs``    Choose a calibration set's exposures.
``calib validate``  Validate candidate calibrations.
``fiberNorms``      The fiber-norm QA.
``agReport``        The end-of-run guider report.
=================== ====================================================================================

The QA tasks themselves run under ``pipetask``, from ``pipelines/drpQA.yaml``.
"""

import importlib
import sys
from collections.abc import Sequence

__all__ = ["COMMANDS", "main"]

#: Each subcommand: the module and function that run it (called with the remaining arguments, returning the
#: exit status), and one line of help.
COMMANDS: dict[tuple[str, ...], tuple[str, str, str]] = {
    ("load",): (
        "pfs.drp.qa.qadb.cli",
        "loadMain",
        "copy a collection's numbers from the Butler to qadb and judge them",
    ),
    ("values",): ("pfs.drp.qa.qadb.cli", "valuesMain", "make, compare and accept a run's normal values"),
    ("judge",): ("pfs.drp.qa.qadb.cli", "judgeMain", "judge stored numbers again for a visit range"),
    ("summary",): ("pfs.drp.qa.qadb.visitSummary", "main", "the verdict of each visit, with an exit status"),
    ("calib", "inputs"): (
        "pfs.drp.qa.comparison.calibInputsCli",
        "main",
        "choose a calibration set's exposures",
    ),
    ("calib", "validate"): (
        "pfs.drp.qa.comparison.calibValidateCli",
        "main",
        "validate candidate calibrations",
    ),
    ("fiberNorms",): ("pfs.drp.qa.fiberNormsQa", "main", "the fiber-norm QA"),
    ("agReport",): ("pfs.drp.qa.guiders.runReport", "main", "the end-of-run guider report"),
}

#: Exit status for a command line that names no subcommand, as argparse's.
EXIT_USAGE = 2


def usage(prog: str = "drpQa.py") -> str:
    """Return the list of subcommands."""
    width = max(len(" ".join(name)) for name in COMMANDS)
    lines = [f"usage: {prog} <subcommand> [options]", "", "subcommands:"]
    lines += [f"  {' '.join(name):<{width}}  {text}" for name, (_, _, text) in COMMANDS.items()]
    lines += ["", f"{prog} <subcommand> --help lists a subcommand's options."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, prog: str = "drpQa.py") -> int:
    """Run a subcommand; return its exit status.

    Parameters
    ----------
    argv : sequence of `str`, optional
        The arguments after the program name; default ``sys.argv[1:]``.
    prog : `str`, optional
        The program name, for usage messages.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    for name, (module, function, _) in COMMANDS.items():
        if tuple(args[: len(name)]) != name:
            continue
        rest = args[len(name) :]
        run = getattr(importlib.import_module(module), function)
        # The subcommand's own parser names itself from argv[0].
        saved = sys.argv
        sys.argv = [" ".join([prog, *name]), *rest]
        try:
            status = run(rest)
        finally:
            sys.argv = saved
        return int(status or 0)
    wantsHelp = args[:1] in (["-h"], ["--help"])
    print(usage(prog), file=sys.stdout if wantsHelp else sys.stderr)
    return 0 if wantsHelp else EXIT_USAGE
