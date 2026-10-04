"""pipelines/qaThresholds.yaml imports labels that exist.

Standard library only: the labels are read as text. Building the pipeline needs
the stack, and the notebook's ``pipetask qgraph`` step does that.
"""

import re
from pathlib import Path

PIPELINES = Path(__file__).parent.parent / "pipelines"


def importedLabels(text: str, key: str) -> list[str]:
    """Return the labels listed under every ``include:`` or ``exclude:`` key."""
    labels = []
    for block in re.findall(rf"^\s+{key}:\n((?:\s+- \S+\n)+)", text, flags=re.MULTILINE):
        labels += re.findall(r"- (\S+)", block)
    return labels


def testImportsImageQualityQaWithoutMergeArms():
    text = (PIPELINES / "qaThresholds.yaml").read_text()
    assert "$DRP_STELLA_DIR/pipelines/reduceExposure.yaml" in text
    assert "$DRP_QA_DIR/pipelines/drpQA.yaml" in text
    assert importedLabels(text, "include") == ["imageQualityQa"]
    assert importedLabels(text, "exclude") == ["mergeArms"]


def testIncludedLabelIsInDrpQa():
    drpQa = (PIPELINES / "drpQA.yaml").read_text()
    assert re.search(r"^  imageQualityQa:$", drpQa, flags=re.MULTILINE)
