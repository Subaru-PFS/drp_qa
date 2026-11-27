"""Tests of the ``fiberProfiles`` statistics, on synthetic profiles and real ones, and of the task that stores them.

The task module imports ``lsst.daf.butler``, ``lsst.pex.config``, ``lsst.pipe.base`` and ``pfs.datamodel`` at
module scope, but ``run`` and ``expectedFibers`` only call duck-typed methods on their inputs. So the module is
imported with minimal stand-ins for whichever of those are missing (`task`), and the stand-ins are removed from
`sys.modules` once it is loaded; with the stack present the real modules are used.

Needs numpy, pandas and scipy (for ``pfs.drp.qa.crossDispersion``); CI runs these tests in the ``deps`` job.
"""

import importlib
import logging
import sys
from types import ModuleType, SimpleNamespace

import numpy as np  # noqa: F401 (listed so the directory is skipped without it)
import pandas as pd  # noqa: F401
import pytest
import scipy  # noqa: F401

TASK_MODULE = "pfs.drp.qa.fiberProfilesQa"


def _missing(name: str) -> bool:
    """Return whether module ``name`` cannot be imported."""
    try:
        importlib.import_module(name)
    except ImportError:
        return True
    return False


def _module(name: str, **attributes) -> ModuleType:
    """Return a stand-in module with ``attributes``."""
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class _Field:
    """Stand-in for ``lsst.pex.config.Field``: a default."""

    def __init__(self, default=None, **kwargs):
        self.default = default


class _Config:
    """Stand-in for ``PipelineTaskConfig``: its fields' defaults, settable."""

    def __init_subclass__(cls, pipelineConnections=None, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self):
        for name, value in vars(type(self)).items():
            if isinstance(value, _Field):
                setattr(self, name, value.default)


class _Connection:
    """Stand-in for a connection type."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Connections:
    """Stand-in for ``PipelineTaskConnections``."""

    def __init_subclass__(cls, dimensions=(), **kwargs):
        super().__init_subclass__(**kwargs)


class _Task:
    """Stand-in for ``PipelineTask``: a config and a `logging` logger."""

    def __init__(self, config=None, **kwargs):
        self.config = config if config is not None else self.ConfigClass()
        self.log = logging.getLogger(self._DefaultName)


class _Handle:
    """Stand-in for ``DeferredDatasetHandle``: ``get`` returns what it was given."""

    def __init__(self, dataset):
        self._dataset = dataset

    def get(self):
        return self._dataset


class _NoWorkFoundError(Exception):
    """Stand-in for ``lsst.pipe.base.NoWorkFound``."""


def _standIns() -> dict[str, ModuleType]:
    """Return stand-ins for the task module's stack imports that are missing."""
    modules = {}
    if _missing("lsst.pipe.base") or _missing("lsst.pex.config") or _missing("lsst.daf.butler"):
        connectionTypes = _module(
            "lsst.pipe.base.connectionTypes",
            Input=_Connection,
            Output=_Connection,
            PrerequisiteInput=_Connection,
        )
        pipeBase = _module(
            "lsst.pipe.base",
            InputQuantizedConnection=object,
            NoWorkFound=_NoWorkFoundError,
            OutputQuantizedConnection=object,
            PipelineTask=_Task,
            PipelineTaskConfig=_Config,
            PipelineTaskConnections=_Connections,
            QuantumContext=object,
            Struct=SimpleNamespace,
            connectionTypes=connectionTypes,
        )
        pexConfig = _module("lsst.pex.config", Field=_Field)
        dafButler = _module("lsst.daf.butler", DeferredDatasetHandle=_Handle)
        modules |= {
            "lsst": _module(
                "lsst",
                daf=_module("lsst.daf", butler=dafButler),
                pex=_module("lsst.pex", config=pexConfig),
                pipe=_module("lsst.pipe", base=pipeBase),
            ),
            "lsst.daf.butler": dafButler,
            "lsst.pex.config": pexConfig,
            "lsst.pipe.base": pipeBase,
            "lsst.pipe.base.connectionTypes": connectionTypes,
        }
        modules["lsst.daf"] = modules["lsst"].daf
        modules["lsst.pex"] = modules["lsst"].pex
        modules["lsst.pipe"] = modules["lsst"].pipe
    if _missing("pfs.datamodel"):
        modules["pfs.datamodel"] = _module(
            "pfs.datamodel", TargetType=SimpleNamespace(SCIENCE=1, ENGINEERING=5)
        )
    return modules


@pytest.fixture(scope="session")
def task() -> ModuleType:
    """Return the ``pfs.drp.qa.fiberProfilesQa`` module, loaded with stand-ins for the stack where it is absent."""
    sys.modules.pop(TASK_MODULE, None)
    modules = _standIns()
    sys.modules.update(modules)
    try:
        return importlib.import_module(TASK_MODULE)
    finally:
        for name in [*modules, TASK_MODULE]:
            sys.modules.pop(name, None)
