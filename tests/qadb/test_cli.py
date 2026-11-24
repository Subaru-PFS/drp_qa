"""``drpQa.py``: every subcommand resolves, and those that read only qadb run without the LSST stack."""

import subprocess
import sys

import pytest

from pfs.drp.qa import cli

#: The subcommands that must run where the stack is not installed: they read and write qadb alone.
QADB_ONLY = (("values",), ("judge",), ("summary",))

#: The subcommands whose module is importable without the stack (it is imported when they run, not before).
STACK_FREE_MODULES = (
    *QADB_ONLY,
    ("redo",),
    ("load",),
    ("markReduceFailed",),
    ("calib", "inputs"),
    ("calib", "validate"),
    ("agReport",),
)


def testSubcommands():
    assert list(cli.COMMANDS) == [
        ("redo",),
        ("load",),
        ("values",),
        ("judge",),
        ("markReduceFailed",),
        ("summary",),
        ("calib", "inputs"),
        ("calib", "validate"),
        ("fiberNorms",),
        ("agReport",),
    ]
    # Twilights are judged by ``load``: there is no subcommand for them.
    assert ("twilight",) not in cli.COMMANDS


def testUsage(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    assert "subcommands:" in capsys.readouterr().err
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert all(" ".join(name) in out for name in cli.COMMANDS)
    assert cli.main(["noSuchCommand"]) == cli.EXIT_USAGE
    # A subcommand of two words needs both.
    assert cli.main(["calib"]) == cli.EXIT_USAGE


@pytest.mark.parametrize("name", STACK_FREE_MODULES, ids=" ".join)
def testHelp(name, capsys):
    """Each subcommand's own parser answers ``--help``, naming itself with the subcommand."""
    with pytest.raises(SystemExit) as exit_:
        cli.main([*name, "--help"])
    assert exit_.value.code == 0
    assert f"usage: drpQa.py {' '.join(name)}" in capsys.readouterr().out


@pytest.mark.parametrize("name", QADB_ONLY, ids=" ".join)
def testReadsOnlyQadbWithoutTheStack(name):
    """Importing and parsing the arguments of a qadb-only subcommand loads no stack or Butler module."""
    module, function, _ = cli.COMMANDS[name]
    code = (
        "import importlib, sys\n"
        f"main = getattr(importlib.import_module({module!r}), {function!r})\n"
        "try:\n"
        "    main(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'lsst' or m.startswith('pfs.drp.stella'))\n"
        "loader = 'pfs.drp.qa.qadb.loader' in sys.modules\n"
        "print(bad, loader)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip().endswith("[] False"), result.stdout


def testTheCheckSeesAStackImport():
    """Negative control: the same probe reports a module it is told to import."""
    code = "import sys, json\nprint(sorted(m for m in sys.modules if m.split('.')[0] == 'json'))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() != "[]"


def testExitStatusIsTheSubcommands(monkeypatch):
    monkeypatch.setitem(cli.COMMANDS, ("probe",), ("pfs.drp.qa.cli", "usage", ""))
    monkeypatch.setattr(cli, "usage", lambda rest: 7)
    assert cli.main(["probe", "x"]) == 7
    monkeypatch.setattr(cli, "usage", lambda rest: None)
    assert cli.main(["probe"]) == 0
