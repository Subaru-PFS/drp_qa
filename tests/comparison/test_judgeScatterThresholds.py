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


def _long(visit, arm, species, rms, cameraSet="brn", reference=VISIT, statusType=RESERVED):
    return [
        {
            "visit": visit,
            "arm": arm,
            "spectrograph": 1,
            "cameraSet": cameraSet,
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
        {"arm": "r", "description": "NeI", "judged": 2, "WARN": 0, "FAIL": 1, "FAIL%": 50.0}
    ]
    # Negative control: without the entry nothing is judged and the summary is empty.
    assert module.summarize(module.scatterVerdicts(metrics, summary, thresholds.iloc[0:0])).empty
