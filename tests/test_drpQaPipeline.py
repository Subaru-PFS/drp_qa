"""pipelines/drpQA.yaml registers the QA task labels a measure pass runs.

Standard library only: the labels are read as text, so this runs in the stack-free job too.
Building the pipeline needs the stack.
"""

import re
from pathlib import Path

PIPELINES = Path(__file__).parent.parent / "pipelines"

#: The labels a measure pass runs, as text, mirroring
#: ``pfs.drp.qa.comparison.reduction.MEASURE_LABELS`` (imported and checked against the pipeline in
#: ``tests/comparison/test_plan.py``); kept in step with that tuple.
MEASURE_LABELS = ("imageQualityQa", "dmLineResiduals", "reductionStats")


def testMeasureLabelsAreRegisteredInDrpQa():
    drpQa = (PIPELINES / "drpQA.yaml").read_text()
    for label in MEASURE_LABELS:
        assert re.search(rf"^  {label}:$", drpQa, flags=re.MULTILINE), label
    # Negative control: mergeArms is one of reduceExposure's labels, not one drpQA registers.
    assert not re.search(r"^  mergeArms:$", drpQa, flags=re.MULTILINE)
