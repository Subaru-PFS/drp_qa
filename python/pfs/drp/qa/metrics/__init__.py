"""Metric definitions, gating, and validation data for ``drp_qa``.

Everything in this subpackage is deliberately free of the LSST/PFS stack and of
the Butler, so that it can be unit-tested in CI. Butler access belongs in
``runQuantum``; the numbers and the verdicts derived from them live here.

See ``doc/qa-rebuild-plan.md``, Phase 1.
"""

from pfs.drp.qa.metrics.definitions import (
    buildImageQualityRegistry,
    imageQualityMetricDefs,
)
from pfs.drp.qa.metrics.goldenVisits import (
    GoldenVisit,
    GoldenVisitSet,
    defaultGoldenVisitsPath,
    loadGoldenVisits,
)
from pfs.drp.qa.metrics.longFormat import (
    LONG_COLUMNS,
    longRecords,
    toLongFrame,
    widen,
)
from pfs.drp.qa.metrics.registry import (
    STATUS_ORDER,
    GateResult,
    MetricDef,
    MetricRegistry,
    Thresholds,
    worstStatus,
)
from pfs.drp.qa.metrics.thresholds import (
    ThresholdSuggestion,
    deriveThresholds,
    roundToReadable,
)

__all__ = [
    "LONG_COLUMNS",
    "STATUS_ORDER",
    "GateResult",
    "GoldenVisit",
    "GoldenVisitSet",
    "MetricDef",
    "MetricRegistry",
    "ThresholdSuggestion",
    "Thresholds",
    "buildImageQualityRegistry",
    "defaultGoldenVisitsPath",
    "deriveThresholds",
    "imageQualityMetricDefs",
    "loadGoldenVisits",
    "longRecords",
    "roundToReadable",
    "toLongFrame",
    "widen",
    "worstStatus",
]
