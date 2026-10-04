"""Consistency checks over the Butler connections of the tasks in drpQA.yaml.

The task modules are read as source, not imported, so these run without the
LSST stack. The defect they catch only shows when every task is resolved into
one graph, which needs a Butler repository: running the tasks one at a time
with ``pipetask run -p drpQA.yaml#label`` hides it.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipelines" / "drpQA.yaml"

# The lsst.pipe.base.connectionTypes a connection can be built from.
CONNECTION_TYPES = {"Input", "Output", "PrerequisiteInput", "InitInput", "InitOutput"}


def pipelineModules(pipeline: Path = PIPELINE) -> list[Path]:
    """Return the source files of the tasks a pipeline registers.

    The file is read with a regular expression rather than a YAML parser so
    that this module needs only the standard library; drpQA.yaml gives each
    task as ``class: <module>.<TaskClass>``.

    Parameters
    ----------
    pipeline : `pathlib.Path`
        Pipeline definition.

    Returns
    -------
    paths : `list` [`pathlib.Path`]
        One path per task module, without duplicates.
    """
    paths = []
    for target in re.findall(r"^\s+class:\s*([\w.]+)\s*$", pipeline.read_text(), re.MULTILINE):
        module = target.rsplit(".", 1)[0]
        path = ROOT / "python" / Path(*module.split(".")).with_suffix(".py")
        if path not in paths:
            paths.append(path)
    return paths


def connectionAliases(tree: ast.Module) -> dict[str, str]:
    """Map the names a module uses for connection types to the types.

    The task modules import them under aliases, e.g.
    ``from lsst.pipe.base.connectionTypes import Input as InputConnection``.

    Parameters
    ----------
    tree : `ast.Module`
        Parsed module.

    Returns
    -------
    aliases : `dict` [`str`, `str`]
        Connection type, by local name.
    """
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "lsst.pipe.base.connectionTypes":
            for alias in node.names:
                if alias.name in CONNECTION_TYPES:
                    aliases[alias.asname or alias.name] = alias.name
    return aliases


def iterConnections(tree: ast.Module):
    """Yield every connection a module's classes declare.

    Parameters
    ----------
    tree : `ast.Module`
        Parsed module.

    Yields
    ------
    className : `str`
        Name of the declaring class.
    attribute : `str`
        Class attribute the connection is assigned to.
    connectionType : `str`
        Connection type, e.g. ``"PrerequisiteInput"``.
    call : `ast.Call`
        The connection's constructor call.
    """
    aliases = connectionAliases(tree)
    for classNode in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        for statement in classNode.body:
            if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
                continue
            func = statement.value.func
            target = statement.targets[0]
            if isinstance(func, ast.Name) and func.id in aliases and isinstance(target, ast.Name):
                yield classNode.name, target.id, aliases[func.id], statement.value


def datasetTypeName(call: ast.Call, attribute: str) -> str:
    """Return the dataset type a connection refers to.

    Parameters
    ----------
    call : `ast.Call`
        The connection's constructor call.
    attribute : `str`
        Class attribute the connection is assigned to, used when the call
        doesn't pass ``name``.

    Returns
    -------
    name : `str`
        Dataset type name.
    """
    for keyword in call.keywords:
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return attribute


def collectConnections(sources: list[str]) -> dict[str, dict[str, set[str]]]:
    """Read the connections declared in some module sources.

    Parameters
    ----------
    sources : `list` [`str`]
        Python source of each module.

    Returns
    -------
    connections : `dict` [`str`, `dict` [`str`, `set` [`str`]]]
        The declaring classes, by connection type, by dataset type name.
    """
    found: dict[str, dict[str, set[str]]] = {}
    for source in sources:
        for className, attribute, connectionType, call in iterConnections(ast.parse(source)):
            dataset = datasetTypeName(call, attribute)
            found.setdefault(dataset, {}).setdefault(connectionType, set()).add(className)
    return found


def mixedDatasetTypes(connections: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, list[str]]]:
    """Return the dataset types declared both as a prerequisite and as an input.

    Parameters
    ----------
    connections : `dict`
        As returned by `collectConnections`.

    Returns
    -------
    mixed : `dict` [`str`, `dict` [`str`, `list` [`str`]]]
        The declaring classes, by connection type, for each such dataset type.
    """
    return {
        dataset: {kind: sorted(classes) for kind, classes in kinds.items()}
        for dataset, kinds in connections.items()
        if "PrerequisiteInput" in kinds and "Input" in kinds
    }


def prerequisiteMinimum(source: str, attribute: str) -> int | None:
    """Return the ``minimum`` a module's prerequisite connection passes.

    Parameters
    ----------
    source : `str`
        Python source of the module.
    attribute : `str`
        Class attribute the connection is assigned to.

    Returns
    -------
    minimum : `int` or `None`
        The value passed, or `None` if the call doesn't pass one.

    Raises
    ------
    LookupError
        Raised if the module declares no such prerequisite.
    """
    for _, name, connectionType, call in iterConnections(ast.parse(source)):
        if name == attribute and connectionType == "PrerequisiteInput":
            for keyword in call.keywords:
                if keyword.arg == "minimum":
                    return keyword.value.value
            return None
    raise LookupError(f"no prerequisite connection {attribute!r}")


# A task module declaring pfsConfig the way imageQualityQa used to: a plain
# input, while TaskB has it as a prerequisite.
MIXED_SOURCE = """
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

class TaskAConnections:
    pfsConfig = InputConnection(name="pfsConfig", minimum=0)

class TaskBConnections:
    config = PrerequisiteConnection(name="pfsConfig")
"""


@pytest.fixture(scope="module")
def connections():
    """Return the connections of the tasks in drpQA.yaml."""
    return collectConnections([path.read_text() for path in pipelineModules()])


def testPipelineModulesExist():
    paths = pipelineModules()
    assert len(paths) >= 5
    for path in paths:
        assert path.exists(), f"{PIPELINE.name} registers a task whose module is missing: {path}"


def testConnectionsCollected(connections):
    assert set(connections["pfsConfig"]) == {"PrerequisiteInput"}
    assert "detectorMap" in connections


def testNoDatasetTypeIsBothPrerequisiteAndInput(connections):
    """A dataset type must be a prerequisite to every task in a graph, or none.

    Otherwise the graph fails to build with ``ConnectionTypeConsistencyError``.
    """
    assert mixedDatasetTypes(connections) == {}

    # Negative control: the check finds a dataset type declared both ways,
    # even under another attribute name.
    assert mixedDatasetTypes(collectConnections([MIXED_SOURCE])) == {
        "pfsConfig": {"Input": ["TaskAConnections"], "PrerequisiteInput": ["TaskBConnections"]}
    }


def testOptionalPfsConfigSaysMinimumZero():
    """The optional pfsConfig of imageQualityQa says minimum=0.

    A prerequisite defaults to ``minimum=1`` and is resolved when the graph is
    built: without ``minimum=0`` a collection lacking a pfsConfig gets no
    quantum at all.
    """
    source = (ROOT / "python" / "pfs" / "drp" / "qa" / "imageQualityQa.py").read_text()
    assert prerequisiteMinimum(source, "pfsConfig") == 0

    # Negative controls: a prerequisite without minimum, and no prerequisite.
    assert prerequisiteMinimum(MIXED_SOURCE, "config") is None
    with pytest.raises(LookupError):
        prerequisiteMinimum(MIXED_SOURCE, "pfsConfig")
