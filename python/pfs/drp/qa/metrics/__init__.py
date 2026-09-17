"""Metric definitions, gating, and validation data for ``drp_qa``.

Everything in this subpackage is deliberately free of the LSST/PFS stack and of
the Butler, so that it can be unit-tested in CI. Butler access belongs in
``runQuantum``; the numbers and the verdicts derived from them live here.

See ``doc/qa-rebuild-plan.md``, Phase 1.
"""

from pfs.drp.qa.metrics.goldenVisits import (
    GoldenVisit,
    GoldenVisitSet,
    defaultGoldenVisitsPath,
    loadGoldenVisits,
)
from pfs.drp.qa.metrics.thresholds import (
    ThresholdSuggestion,
    deriveThresholds,
    roundToReadable,
)

__all__ = [
    "GoldenVisit",
    "GoldenVisitSet",
    "ThresholdSuggestion",
    "defaultGoldenVisitsPath",
    "deriveThresholds",
    "loadGoldenVisits",
    "roundToReadable",
]
