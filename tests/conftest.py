"""Pytest configuration for the ``drp_qa`` test suite.

Some test modules import the LSST/PFS stack (``lsst.utils.tests``, ``lsst.log``,
``pfs.drp.stella``) at module scope. Where the stack is absent — notably in CI —
those modules cannot even be *collected*: the import fails during collection and
pytest aborts the whole run, taking the stack-free tests down with it.

An in-test ``try: import ... except ImportError: skipTest(...)`` does not help,
because the module-level import has already failed before any test body runs.

So the stack-dependent modules are listed in ``_STACK_MODULES`` and ignored when
their imports cannot be satisfied. Rather than probe one hand-picked package
name, each listed module's own top-level imports are read from its source and
checked individually. A partial environment — say ``lsst.utils`` present but
``lsst.log`` missing — is then detected correctly, and the check cannot drift
out of step when a module's imports change.

When adding a test module that imports the stack at module scope, add it to
``_STACK_MODULES``. Better still, keep the logic under test in a pure function
that takes arrays or DataFrames so the test needs no stack at all — see
``doc/qa-rebuild-plan.md``.
"""

import ast
import importlib.util
import sys
from pathlib import Path

# Test modules that import the LSST/PFS stack at module scope.
#
# Empty today: every stack-dependent module fetches the stack with
# ``pytest.importorskip`` at module level instead, which skips cleanly on its
# own. Prefer that in new tests -- it keeps the guard next to the import it
# guards. This list stays as the escape hatch for a module that genuinely cannot
# use importorskip, e.g. one that subclasses ``lsst.utils.tests.TestCase``.
_STACK_MODULES: list[str] = []

_HERE = Path(__file__).parent

# The package is deliberately not installed in CI (see .github/workflows/tests.yml),
# and `setup -r .` only puts `python/` on PYTHONPATH for an EUPS shell. Put it on
# sys.path here so the stack-free tests can import `pfs.drp.qa.*` either way.
_SOURCE = _HERE.parent / "python"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))


def _moduleLevelImports(path: Path) -> set[str]:
    """Return the names a module imports at module scope, without importing it.

    Parameters
    ----------
    path : `pathlib.Path`
        Python source file to inspect.

    Returns
    -------
    `set` [`str`]
        Imported module names. Relative imports are skipped, since they resolve
        within the test directory rather than against the environment.
    """
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        # Unreadable or unparseable: let pytest report it rather than guessing.
        return set()

    names: set[str] = set()
    for node in tree.body:  # module scope only; nested imports cannot break collection
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _importable(name: str) -> bool:
    """Return True when ``name`` can be located without importing it fully.

    Parameters
    ----------
    name : `str`
        Dotted module name.

    Returns
    -------
    `bool`
        True if the module can be found, False otherwise.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        # A parent package that is missing, or present but not a package,
        # raises rather than returning None.
        return False


def _unsatisfied(filename: str) -> bool:
    """Return True when a test module's module-scope imports cannot be met.

    Parameters
    ----------
    filename : `str`
        Test module file name, relative to this directory.

    Returns
    -------
    `bool`
        True if the module should be ignored during collection.
    """
    path = _HERE / filename
    if not path.exists():
        return False
    return any(not _importable(name) for name in _moduleLevelImports(path))


collect_ignore = [name for name in _STACK_MODULES if _unsatisfied(name)]
