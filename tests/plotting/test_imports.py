"""The plotting and metrics subpackages import no Butler, stack or task module.

Checked twice: statically, over every import statement at any scope, and at
run time, by importing them in a fresh interpreter and looking at
``sys.modules``, which also catches what they import indirectly.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pfs.drp.qa.metrics
import pfs.drp.qa.plotting

# The stack (and with it the Butler), and the task modules.
FORBIDDEN = (
    "lsst",
    "pfs.drp.stella",
    "pfs.drp.qa.dmResiduals",
    "pfs.drp.qa.dmCombinedResiduals",
    "pfs.drp.qa.extractionQa",
    "pfs.drp.qa.extractionQaCombined",
    "pfs.drp.qa.fiberNormsQa",
    "pfs.drp.qa.fluxCalQa",
    "pfs.drp.qa.imageQualityQa",
    "pfs.drp.qa.skySubtractionQa",
    "pfs.drp.qa.storageClasses",
    "pfs.drp.qa.formatters",
)

PACKAGES = {
    "pfs.drp.qa.plotting": Path(pfs.drp.qa.plotting.__file__).parent,
    "pfs.drp.qa.metrics": Path(pfs.drp.qa.metrics.__file__).parent,
}


def isForbidden(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN)


def forbiddenImports(path: Path) -> list[str]:
    """Return the forbidden modules a source file imports, at any scope."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return sorted(name for name in names if isForbidden(name))


@pytest.mark.parametrize("package", PACKAGES)
def testNoForbiddenImportStatements(package):
    offenders = {
        source.name: bad
        for source in sorted(PACKAGES[package].glob("*.py"))
        if (bad := forbiddenImports(source))
    }
    assert offenders == {}


def testForbiddenImportsAreFound():
    # Negative control: the task module the plots came from imports the stack.
    source = Path(pfs.drp.qa.plotting.__file__).parents[1] / "dmResiduals.py"
    bad = forbiddenImports(source)
    assert "pfs.drp.stella" in bad
    assert any(name.startswith("lsst.") for name in bad)


def testNothingForbiddenIsLoaded():
    code = (
        "import json, sys\n"
        "import pfs.drp.qa.plotting, pfs.drp.qa.metrics.fitStats\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    loaded = json.loads(result.stdout)
    assert "pfs.drp.qa.plotting.dmResiduals" in loaded
    assert [name for name in loaded if isForbidden(name)] == []
