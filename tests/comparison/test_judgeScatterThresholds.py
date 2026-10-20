"""Tests for ``bin.src/judgeScatterThresholds.py``, the scatter-threshold acceptance check."""

import importlib.util
from pathlib import Path

import pandas as pd

from pfs.drp.qa.metrics.lineResiduals import RESERVED, VISIT

SCRIPT = Path(__file__).parents[2] / "bin.src" / "judgeScatterThresholds.py"


def _module():
    spec = importlib.util.spec_from_file_location("judgeScatterThresholds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _long(visit, arm, species, rms, cameraSet="brn", reference=VISIT, statusType=RESERVED, obsType="arc"):
    return [
        {
            "visit": visit,
            "arm": arm,
            "spectrograph": 1,
            "cameraSet": cameraSet,
            "obsType": obsType,
            "reference": reference,
            "description": species,
            "statusType": statusType,
            "metric": metric,
            "value": value,
            "status": "",
        }
        for metric, value in (("nLines", 50), ("robustRmsDeltaX", rms), ("robustRmsDeltaY", rms))
    ]


def testScatterVerdictsAndSummary():
    module = _module()
    metrics = pd.DataFrame(
        [
            *_long(1, "r", "NeI", 0.05),
            *_long(2, "r", "NeI", 0.30),  # crosses
            *_long(3, "r", "KrI", 0.30),  # no entry: not judged
            *_long(4, "r", "NeI", 0.30, reference="calib", statusType="BOTH"),  # not the scatter rows
        ]
    )
    summary = pd.DataFrame({"visit": [1, 2, 3, 4], "arm": "r", "spectrograph": 1})
    thresholds = pd.DataFrame(
        [
            {
                "metric": "robustRmsDeltaX",
                "reference": VISIT,
                "statusType": RESERVED,
                "arm": "r",
                "description": "NeI",
                "cameraSet": "brn",
                "obsType": "arc",
                "warn": 0.1,
                "fail": 0.2,
                "higherIsWorse": True,
                "absolute": False,
                "provenance": "test",
            }
        ]
    )
    verdicts = module.scatterVerdicts(metrics, summary, thresholds)
    byVisit = verdicts[verdicts["metric"] == "robustRmsDeltaX"].set_index("visit")["status"]
    assert byVisit.to_dict() == {1: "PASS", 2: "FAIL", 3: ""}
    assert (verdicts.loc[verdicts["metric"] == "robustRmsDeltaY", "status"] == "").all()  # y: no entry
    counts = module.summarize(verdicts)
    assert counts.to_dict("records") == [
        {"arm": "r", "obsType": "arc", "description": "NeI", "judged": 2, "WARN": 0, "FAIL": 1, "FAIL%": 50.0}
    ]
    assert not verdicts["engineering"].any()  # no sequence names: nothing is set apart
    # A table stored before obsType was a column takes it from iqQaMetrics beside the report, whose
    # sequence names also set the engineering sequences apart: counted on their own, as the gate
    # leaves them unvalidated.
    older = metrics.drop(columns="obsType")
    kinds = pd.DataFrame(
        {"visit": [1, 2, 3, 4], "arm": "r", "spectrograph": 1, "obsType": "arc", "seqName": "Arc: Neon"}
    )
    assert (
        module.scatterVerdicts(older, summary, thresholds, kinds)["status"].tolist()
        == verdicts["status"].tolist()
    )
    bootstrap = module.scatterVerdicts(
        older, summary, thresholds, kinds.assign(seqName=["Bootstrap", *["Arc: Neon"] * 3])
    )
    assert bootstrap["engineering"].tolist() == [True, True] + [False] * (len(bootstrap) - 2)
    assert module.summarize(bootstrap)["judged"].tolist() == [1]  # visit 2 alone; visit 1 is the bootstrap
    assert module.summarize(bootstrap, engineering=True)["judged"].tolist() == [1]
    # Negative control: without the entry nothing is judged and the summary is empty; and without
    # the frame type the entry, which names one, matches nothing.
    assert module.summarize(module.scatterVerdicts(metrics, summary, thresholds.iloc[0:0])).empty
    unknown = module.scatterVerdicts(older, summary, thresholds)
    assert (unknown["status"] == "").all() and unknown["obsType"].isna().all()
