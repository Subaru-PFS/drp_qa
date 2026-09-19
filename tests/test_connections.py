"""Consistency checks over the pipeline's Butler connections.

These read the task modules as source rather than importing them, so they run
without the LSST stack. That matters here: the defect this file exists to catch
only appears when every task is resolved into one graph, which needs a Butler
repository and the stack, so in practice nobody saw it. Running the tasks one at
a time with ``pipetask run -p ...#label`` hides it completely.
"""

import ast
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "python" / "pfs" / "drp" / "qa"
_PIPELINE = _ROOT / "pipelines" / "drpQA.yaml"

#: The ``lsst.pipe.base.connectionTypes`` names a connection can be built from.
_CONNECTION_TYPES = {"Input", "Output", "PrerequisiteInput", "InitInput", "InitOutput"}


def _connectionAliases(tree: ast.Module) -> dict[str, str]:
    """Map the local names a module uses to the connection type they refer to.

    Every task module imports these under an alias, e.g.
    ``from lsst.pipe.base.connectionTypes import Input as InputConnection``.

    Parameters
    ----------
    tree : `ast.Module`
        Parsed module.

    Returns
    -------
    `dict` [`str`, `str`]
        Local name to connection type name.
    """
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "lsst.pipe.base.connectionTypes":
            for alias in node.names:
                if alias.name in _CONNECTION_TYPES:
                    aliases[alias.asname or alias.name] = alias.name
    return aliases


def _datasetTypeName(node: ast.Call, attribute: str) -> str:
    """Return the Butler dataset type a connection refers to.

    Parameters
    ----------
    node : `ast.Call`
        The connection constructor call.
    attribute : `str`
        The class attribute the connection is assigned to, used when the call
        does not pass ``name=`` explicitly.

    Returns
    -------
    `str`
        The dataset type name.
    """
    for keyword in node.keywords:
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return attribute


def pipelineModules() -> list[Path]:
    """Return the source files of the tasks registered in ``drpQA.yaml``.

    The consistency rule binds tasks that share a graph, not the package as a
    whole: a task in no pipeline, or in a different one, may legitimately
    declare the same dataset type differently. Scoping to the pipeline is what
    makes this test a statement about something that can actually break.

    Returns
    -------
    `list` [`pathlib.Path`]
        One path per registered task, deduplicated.
    """
    pipeline = yaml.safe_load(_PIPELINE.read_text())
    paths = []
    for task in pipeline["tasks"].values():
        target = task["class"] if isinstance(task, dict) else task
        module = target.rsplit(".", 1)[0]  # drop the class name
        path = _ROOT / "python" / Path(*module.split(".")).with_suffix(".py")
        if path not in paths:
            paths.append(path)
    return paths


def _collectConnections(paths: list[Path]) -> dict[str, dict[str, set[str]]]:
    """Read every connection declared by the given task modules.

    Parameters
    ----------
    paths : `list` [`pathlib.Path`]
        Modules to read.

    Returns
    -------
    `dict`
        ``{datasetTypeName: {connectionType: {taskClassName, ...}}}``.
    """
    found: dict[str, dict[str, set[str]]] = {}
    for path in paths:
        tree = ast.parse(path.read_text())
        aliases = _connectionAliases(tree)
        if not aliases:
            continue
        for classNode in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for statement in classNode.body:
                if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
                    continue
                func = statement.value.func
                local = func.id if isinstance(func, ast.Name) else None
                if local not in aliases:
                    continue
                target = statement.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                dataset = _datasetTypeName(statement.value, target.id)
                found.setdefault(dataset, {}).setdefault(aliases[local], set()).add(classNode.name)
    return found


@pytest.fixture(scope="module")
def connections():
    return _collectConnections(pipelineModules())


class TestConnectionTypeConsistency:
    def testEveryRegisteredTaskModuleExists(self):
        paths = pipelineModules()
        assert len(paths) >= 4
        for path in paths:
            assert path.exists(), f"{_PIPELINE.name} registers a task whose module is missing: {path}"

    def testSomethingWasCollected(self, connections):
        """Guard against the parser silently finding nothing."""
        assert "pfsConfig" in connections
        assert "detectorMap" in connections

    def testNoDatasetTypeIsBothPrerequisiteAndPlainInput(self, connections):
        """A dataset type must be a prerequisite to every task, or to none.

        Mixing the two resolves to ``ConnectionTypeConsistencyError`` and the
        whole pipeline fails to build -- which nothing catches while the tasks
        are run one at a time. ``pfsConfig`` was declared both ways: a
        `PrerequisiteInput` in the two extraction tasks and a plain `Input` in
        ``imageQualityQa``.
        """
        offenders = {
            dataset: {kind: sorted(tasks) for kind, tasks in kinds.items()}
            for dataset, kinds in connections.items()
            if "PrerequisiteInput" in kinds and "Input" in kinds
        }
        assert offenders == {}, (
            "dataset types declared as both a prerequisite and a plain input; "
            f"the pipeline will not resolve: {offenders}"
        )

    def testPfsConfigIsAPrerequisiteEverywhere(self, connections):
        kinds = connections["pfsConfig"]
        assert set(kinds) == {"PrerequisiteInput"}, f"pfsConfig declared as {sorted(kinds)}"

    def testTasksOutsideThePipelineAreNotSilentlyInconsistent(self):
        """A heads-up, not a failure: these tasks share no graph with the rest.

        ``skySubtractionQa``, ``fluxCalQa`` and ``fiberNormsQa`` are not in
        ``drpQA.yaml``. If one is ever added, its connections have to agree with
        the others', so record here what would have to change.
        """
        inPipeline = set(pipelineModules())
        others = [p for p in sorted(_SOURCE.rglob("*.py")) if p not in inPipeline]
        outside = _collectConnections(others)
        combined = _collectConnections(pipelineModules() + others)

        wouldClash = {
            dataset: {kind: sorted(tasks) for kind, tasks in kinds.items()}
            for dataset, kinds in combined.items()
            if "PrerequisiteInput" in kinds and "Input" in kinds
        }
        if wouldClash:
            print(
                "NOTE: these dataset types would clash if the out-of-pipeline tasks "
                f"were added to drpQA.yaml: {wouldClash}"
            )
        assert outside is not None  # the check is the printed note, not a verdict

    def testOptionalPrerequisitesSayMinimumZero(self):
        """A prerequisite defaults to ``minimum=1`` and resolves at graph build.

        An optional one that omits ``minimum=0`` produces no quantum at all
        against a collection that lacks it, and no runtime ``try/except`` can
        rescue a quantum that was never created. ``imageQualityQa`` treats
        ``pfsConfig`` as optional, so it has to say so.
        """
        tree = ast.parse((_SOURCE / "imageQualityQa.py").read_text())
        aliases = _connectionAliases(tree)
        for classNode in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for statement in classNode.body:
                if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
                    continue
                func = statement.value.func
                if not isinstance(func, ast.Name) or aliases.get(func.id) != "PrerequisiteInput":
                    continue
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id == "pfsConfig":
                    minimum = [kw for kw in statement.value.keywords if kw.arg == "minimum"]
                    assert minimum, "pfsConfig is optional but does not set minimum"
                    assert minimum[0].value.value == 0
                    return
        pytest.fail("pfsConfig prerequisite connection not found in imageQualityQa")
