"""Run ``ImageQualityQaTask.run`` without the LSST stack.

The task module imports ``lsst.afw.image``, ``lsst.pex.config``,
``lsst.pipe.base``, ``pfs.datamodel`` and ``pfs.drp.stella`` at module scope,
but ``run`` only calls duck-typed methods on its inputs. So the module is
imported with minimal stand-ins for whichever of those are missing, and the
tests hand ``run`` fakes for the detectorMap, calexp and fiberProfiles. The
stand-ins are removed from `sys.modules` once the task module is loaded, so
no other test sees them; with the stack present the real modules are used.

``computeImageQuality`` and ``addTraceLambdaToArclines`` are drp_stella code
the tests replace with `monkeypatch`, so the arc-line table is given directly.

Needs numpy, pandas, pyyaml and scipy (for ``pfs.drp.qa.metrics``); CI runs
these tests in the ``deps`` job.
"""

import importlib
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy  # noqa: F401 (pfs.drp.qa.metrics needs it; listed so the directory is skipped without it)
import yaml  # noqa: F401 (pfs.drp.qa.metrics needs it; listed so the directory is skipped without it)

TASK_MODULE = "pfs.drp.qa.imageQualityQa"

#: Default fiber pitch of the fake detector: wide, because most of these tests
#: are about which path is taken, not about the width estimator. PFS fibers are
#: 6.17 px apart (`PFS_PITCH`).
PITCH = 20.0
PFS_PITCH = 6.17


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
    """Stand-in for ``lsst.pex.config.Field`` and ``DictField``: a default."""

    def __init__(self, default=None, **kwargs):
        self.default = default


class _Config:
    """Stand-in for ``PipelineTaskConfig``: its fields' defaults, settable."""

    def __init_subclass__(cls, pipelineConnections=None, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self):
        for klass in reversed(type(self).__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, _Field):
                    default = value.default
                    setattr(self, name, dict(default) if isinstance(default, dict) else default)


class _Connections:
    """Stand-in for ``PipelineTaskConnections``."""

    def __init_subclass__(cls, dimensions=(), **kwargs):
        super().__init_subclass__(**kwargs)


class _Connection:
    """Stand-in for a connection type."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Task:
    """Stand-in for ``PipelineTask``: a config and a `logging` logger."""

    def __init__(self, config=None, **kwargs):
        self.config = config if config is not None else self.ConfigClass()
        self.log = logging.getLogger(self._DefaultName)


def _standIns() -> dict[str, ModuleType]:
    """Return stand-ins for the task module's stack imports that are missing."""
    modules = {}
    if _missing("lsst.pipe.base") or _missing("lsst.afw.image") or _missing("lsst.pex.config"):
        connectionTypes = _module(
            "lsst.pipe.base.connectionTypes",
            Input=_Connection,
            Output=_Connection,
            PrerequisiteInput=_Connection,
        )
        pipeBase = _module(
            "lsst.pipe.base",
            InputQuantizedConnection=object,
            OutputQuantizedConnection=object,
            PipelineTask=_Task,
            PipelineTaskConfig=_Config,
            PipelineTaskConnections=_Connections,
            QuantumContext=object,
            Struct=SimpleNamespace,
            connectionTypes=connectionTypes,
        )
        afwImage = _module("lsst.afw.image", Exposure=object)
        pexConfig = _module("lsst.pex.config", Field=_Field, DictField=_Field)
        modules |= {
            "lsst": _module(
                "lsst",
                afw=_module("lsst.afw", image=afwImage),
                pex=_module("lsst.pex", config=pexConfig),
                pipe=_module("lsst.pipe", base=pipeBase),
            ),
            "lsst.afw.image": afwImage,
            "lsst.pex.config": pexConfig,
            "lsst.pipe.base": pipeBase,
            "lsst.pipe.base.connectionTypes": connectionTypes,
        }
        modules["lsst.afw"] = modules["lsst"].afw
        modules["lsst.pex"] = modules["lsst"].pex
        modules["lsst.pipe"] = modules["lsst"].pipe
    if _missing("pfs.datamodel"):
        modules["pfs.datamodel"] = _module(
            "pfs.datamodel",
            FiberStatus=SimpleNamespace(GOOD=1, BROKENFIBER=2, BLACKSPOT=4),
            TargetType=SimpleNamespace(SCIENCE=1, FLUXSTD=3, UNASSIGNED=4, ENGINEERING=5, HOME=9),
            PfsConfig=object,
        )
    if _missing("pfs.drp.stella"):
        modules |= {
            "pfs.drp.stella": _module(
                "pfs.drp.stella", ArcLineSet=object, DetectorMap=object, FiberProfileSet=object
            ),
            "pfs.drp.stella.utils.quality": _module("pfs.drp.stella.utils.quality", computeImageQuality=None),
            "pfs.drp.stella.utils.stability": _module(
                "pfs.drp.stella.utils.stability", addTraceLambdaToArclines=None
            ),
        }
    return modules


@contextmanager
def _installed(modules: dict[str, ModuleType]) -> Iterator[None]:
    """Put ``modules`` in `sys.modules`, and take them out again with the task module."""
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in [*modules, TASK_MODULE]:
            sys.modules.pop(name, None)


@pytest.fixture(scope="session")
def iqqa() -> ModuleType:
    """Return the ``pfs.drp.qa.imageQualityQa`` module."""
    sys.modules.pop(TASK_MODULE, None)
    with _installed(_standIns()):
        return importlib.import_module(TASK_MODULE)


class FakeDetectorMap:
    """Straight vertical traces ``pitch`` apart; wavelength is the row."""

    def __init__(self, numFibers: int = 10, offset: float = 0.0, pitch: float = PITCH):
        self.fiberId = np.arange(1, numFibers + 1, dtype=np.int32)
        self.offset = offset
        self.pitch = pitch

    def getXCenter(self, fiberId, y):
        return self.pitch * np.asarray(fiberId, dtype=float) + self.offset + 0.0 * np.asarray(y)

    def findWavelength(self, fiberId, y):
        return 500.0 + 0.1 * np.asarray(y, dtype=float) + 0.0 * np.asarray(fiberId)


class FakeExposure:
    """A calexp: image, mask and header."""

    def __init__(self, image: np.ndarray, metadata: dict):
        self.image = SimpleNamespace(array=image)
        self.variance = SimpleNamespace(array=np.ones(image.shape))  # unit noise, as the images have
        self.mask = SimpleNamespace(
            array=np.zeros(image.shape, dtype=np.int32), getPlaneBitMask=lambda names: 1
        )
        self.metadata = metadata

    def getMetadata(self) -> dict:
        return self.metadata


def header(seqType: str, seqName: str) -> dict:
    """Return the header keys ``_classifyVisit`` reads."""
    return {"W_SEQTYP": seqType, "W_SEQNAM": seqName}


def traceImage(
    detectorMap: FakeDetectorMap,
    sigma: float,
    height: int = 400,
    peak: float = 1000.0,
    seed: int = 1,
    fibers: np.ndarray | None = None,
) -> np.ndarray:
    """Return an image of Gaussian traces of width ``sigma`` with unit noise; ``fibers`` lit, or all."""
    width = int(detectorMap.pitch * (len(detectorMap.fiberId) + 2))
    columns = np.arange(width, dtype=float)
    lit = detectorMap.fiberId if fibers is None else np.asarray(fibers)
    centers = detectorMap.getXCenter(lit, np.zeros(len(lit)))
    row = sum(peak * np.exp(-0.5 * ((columns - center) / sigma) ** 2) for center in centers)
    rng = np.random.default_rng(seed)
    return np.tile(row, (height, 1)) + rng.normal(0.0, 1.0, (height, width))


def noiseImage(detectorMap: FakeDetectorMap, height: int = 400, seed: int = 1) -> np.ndarray:
    """Return an image with no trace, so every calexp sample is flagged."""
    width = int(detectorMap.pitch * (len(detectorMap.fiberId) + 2))
    return np.random.default_rng(seed).normal(0.0, 1.0, (height, width))


class FakeFiberProfiles:
    """fiberProfiles whose swaths all have Gaussian width ``sigma``."""

    def __init__(self, detectorMap: FakeDetectorMap, sigma: float, numSwaths: int = 5):
        self.fiberId = detectorMap.fiberId
        self.sigma = sigma
        self.numSwaths = numSwaths

    def __getitem__(self, fiberId):
        rows = np.linspace(20.0, 380.0, self.numSwaths)
        index = np.arange(-33, 34) / 3.0  # radius 10, oversampled 3x, as the Run25 calibrations are
        profile = np.exp(-0.5 * (index / self.sigma) ** 2)
        return SimpleNamespace(
            rows=rows,
            norm=np.ones(400),
            index=index,
            profiles=np.ma.masked_array(np.tile(profile, (self.numSwaths, 1))),
        )


def arcLines(
    detectorMap: FakeDetectorMap,
    numPerFiber: int,
    fwhm: float = 2.5,
    flagged: int = 0,
    fibers: np.ndarray | None = None,
    description: str | None = None,
) -> pd.DataFrame:
    """Return what ``computeImageQuality`` gives: one row per line, the first ``flagged`` flagged.

    The lines are on ``fibers`` (default: all), and carry ``description`` when given
    (``"Trace"`` for trace centroids, which have no width).
    """
    fiberId = np.repeat(detectorMap.fiberId if fibers is None else np.asarray(fibers), numPerFiber)
    y = np.tile(np.linspace(10.0, 390.0, numPerFiber), len(fiberId) // max(numPerFiber, 1))
    flag = np.zeros(len(fiberId), dtype=bool)
    flag[:flagged] = True
    lines = pd.DataFrame(
        {
            "fiberId": fiberId,
            "x": detectorMap.getXCenter(fiberId, y),
            "y": y,
            "lam": detectorMap.findWavelength(fiberId, y),
            "fwhm": np.full(len(fiberId), fwhm),
            "theta": np.zeros(len(fiberId)),
            "flux": np.full(len(fiberId), 1000.0),
            "fluxErr": np.ones(len(fiberId)),
            "flag": flag,
            "status": np.zeros(len(fiberId), dtype=np.int32),
            "traceOnly": False,
        }
    )
    if description is not None:
        lines["description"] = description
        if description == "Trace":
            lines["fwhm"] = np.nan
    return lines


class FakePfsConfig:
    """The fiber columns of a pfsConfig: ``status`` (default GOOD) and ``targetType`` per fiber."""

    def __init__(self, fiberId, status=None, targetType=None):
        self.fiberId = np.asarray(fiberId, dtype=np.int32)
        self.fiberStatus = np.full(len(self.fiberId), 1) if status is None else np.asarray(status)
        self.targetType = np.full(len(self.fiberId), 9) if targetType is None else np.asarray(targetType)


def oneGroup(
    detectorMap: FakeDetectorMap, group: int = 0, groups: int = 4
) -> tuple[np.ndarray, FakePfsConfig]:
    """Return the lit fibers of a design lighting every ``groups``-th fiber, and its pfsConfig.

    The others are hidden as the fiberProfiles group designs hide them: ``fiberStatus``
    BLACKSPOT, ``targetType`` UNASSIGNED.
    """
    fiberId = detectorMap.fiberId
    lit = (np.arange(len(fiberId)) % groups) == group
    return fiberId[lit], FakePfsConfig(fiberId, status=np.where(lit, 1, 4), targetType=np.where(lit, 9, 4))


@pytest.fixture
def runTask(iqqa, monkeypatch):
    """Return a function that runs the task on fakes and returns its outputs.

    It takes the arc-line table ``lines``, the dataId's ``arm``, and the
    task's other inputs as keywords; ``thresholds``, when given, replaces the
    task's threshold tables (highest priority first). Otherwise the task
    judges by its config fields alone: these tests are about paths, not the
    shipped thresholds file.
    """

    def run(lines: pd.DataFrame, arm: str = "r", thresholds=None, **inputs):
        monkeypatch.setattr(iqqa, "addTraceLambdaToArclines", lambda arcLines, detectorMap: arcLines)
        monkeypatch.setattr(iqqa, "computeImageQuality", lambda arcLines: arcLines.copy())
        config = iqqa.ImageQualityQaConfig()
        config.thresholdsFile = ""
        task = iqqa.ImageQualityQaTask(config=config)
        if thresholds is not None:
            task.thresholds = thresholds
        defaults = {
            "detectorMap": FakeDetectorMap(),
            "fiberProfiles": None,
            "detectorMapCalib": None,
            "calexp": None,
            "pfsConfig": None,
        }
        return task.run(
            arcLines=lines, dataId={"visit": 1, "arm": arm, "spectrograph": 2}, **(defaults | inputs)
        )

    return run


@pytest.fixture
def fakes() -> SimpleNamespace:
    """Return the fake builders, since test modules can't import each other."""
    return SimpleNamespace(
        DetectorMap=FakeDetectorMap,
        Exposure=FakeExposure,
        FiberProfiles=FakeFiberProfiles,
        header=header,
        traceImage=traceImage,
        noiseImage=noiseImage,
        arcLines=arcLines,
        PfsConfig=FakePfsConfig,
        oneGroup=oneGroup,
        PFS_PITCH=PFS_PITCH,
    )
