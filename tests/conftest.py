"""Pytest configuration for the ``drp_qa`` test suite.

Some test modules import the LSST/PFS stack (``lsst.utils.tests``,
``lsst.log``, ``pfs.drp.stella``) at module scope. Where the stack is absent —
notably in CI — those modules cannot even be *collected*: the import fails
during collection and pytest aborts the whole run, taking the stack-free tests
down with it.

An in-test ``try: import ... except ImportError: skipTest(...)`` does not help,
because the module-level import has already failed before any test body runs.

So the stack-dependent modules are listed here and ignored when the stack is
unavailable. With the stack set up they are collected and run normally.

When adding a test module that imports ``lsst.*`` or ``pfs.drp.stella`` at
module scope, add it to ``_STACK_MODULES``. Better still, keep the logic under
test in a pure function that takes arrays or DataFrames so the test needs no
stack at all — see ``doc/qa-rebuild-plan.md``.
"""

import importlib.util

# Test modules that import the LSST/PFS stack at module scope.
_STACK_MODULES = [
    "test_dmResiduals.py",
]


def _stackAvailable() -> bool:
    """Return True when the LSST/PFS stack can be imported.

    Returns
    -------
    bool
        True if ``lsst.utils`` is importable, False otherwise.
    """
    try:
        return importlib.util.find_spec("lsst.utils") is not None
    except (ImportError, ValueError):
        # ValueError: ``lsst`` present but not a package with a spec.
        return False


collect_ignore = [] if _stackAvailable() else list(_STACK_MODULES)
