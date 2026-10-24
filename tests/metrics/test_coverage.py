"""Tests for the known-bad coverage check: every known-bad entry is caught, or says why not."""

import ast
import contextlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.coverage import CAUGHT, CoverageSource, knownBadCoverage
from pfs.drp.qa.metrics.gate import configThresholds, thresholdsPath
from pfs.drp.qa.metrics.lineResiduals import residualThresholds
from pfs.drp.qa.metrics.validationVisits import ValidationVisit, ValidationVisitSet, loadValidationVisits

ROOT = Path(__file__).parents[2]
DATA = Path(__file__).parent / "data"

#: The metrics each table answers for.
IMAGE_QUALITY = ("medFwhm", "pctFlagged", "nLines", "linesPerFiber", "medDxCenter", "dxCenterRms")
LINE_RESIDUALS = ("medianDeltaX", "medianDeltaY", "robustRmsDeltaX", "robustRmsDeltaY")


def configDefaults(module: str, cls: str) -> SimpleNamespace:
    """Return a task config's field defaults, read from its source with `ast`.

    The task modules import the stack, so the config can't be instantiated
    here: a changed default changes this. ``float("nan")`` is read as NaN; any
    other default that is not a literal is left out.
    """
    source = (ROOT / "python" / "pfs" / "drp" / "qa" / module).read_text()
    config = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef) and n.name == cls)
    defaults = {}
    for statement in config.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            for keyword in statement.value.keywords:
                if keyword.arg == "default":
                    if ast.unparse(keyword.value) == "float('nan')":
                        defaults[statement.targets[0].id] = float("nan")
                        continue
                    with contextlib.suppress(ValueError):
                        defaults[statement.targets[0].id] = ast.literal_eval(keyword.value)
    return SimpleNamespace(**defaults)


@pytest.fixture(scope="module")
def sources() -> list[CoverageSource]:
    """Return the validation visits' stored metrics with the thresholds the two tasks judge them by."""
    iq = configDefaults("imageQualityQa.py", "ImageQualityQaConfig")
    dm = configDefaults("dmLineResiduals.py", "DmLineResidualsConfig")
    return [
        CoverageSource(
            pd.read_parquet(DATA / "iqQaMetrics-validation.parquet"),
            [thresholdsPath(iq.thresholdsFile), configThresholds(iq)],
            IMAGE_QUALITY,
        ),
        CoverageSource(
            # A species with fewer than minLines lines is stored as NaN: not measured.
            pd.read_parquet(DATA / "dmDriftMetrics-validation.parquet"),
            [
                thresholdsPath(dm.thresholdsFile),
                residualThresholds(
                    dm.xWarnThreshold,
                    dm.xFailThreshold,
                    dm.yWarnThreshold,
                    dm.yFailThreshold,
                    dm.xRmsWarnThreshold,
                    dm.xRmsFailThreshold,
                    dm.yRmsWarnThreshold,
                    dm.yRmsFailThreshold,
                ),
            ],
            LINE_RESIDUALS,
        ),
    ]


def testEveryKnownBadEntryIsCaughtOrUnchecked(sources):
    """The validation set's promise: a known-bad entry's check runs and gives its verdict, or it says why not.

    An entry naming a metric that stopped being judged (nLines, pctFlagged; PIPE2D-1935, 1929)
    went unnoticed until PIPE2D-1934 because nothing checked this.
    """
    visitSet = loadValidationVisits()
    coverage = knownBadCoverage(visitSet, sources)
    confirmed = coverage[~coverage["unconfirmed"]]
    assert len(confirmed) == len(visitSet.confirmedBad)
    failing = confirmed.loc[~confirmed["ok"], ["visits", "metric", "result", "unchecked"]]
    assert failing.empty, f"known-bad entries neither caught nor unchecked:\n{failing.to_string()}"

    caught = confirmed[confirmed["result"] == CAUGHT]
    assert (caught["unchecked"] == "").all() and (caught["reached"] == caught["measured"]).all()
    assert caught.groupby("metric")["reached"].sum().to_dict() == {
        "linesPerFiber": 48,  # the obstructed Run30 frames
        "medFwhm": 82,  # SM1 defocused, Run27
        "medianDeltaX": 88,  # Run27 against a later calibration, SM3
        "medianDeltaY": 36,  # and SM2
        "robustRmsDeltaX": 4,  # 150661, the screen in the wrong position, b arm
    }
    unchecked = confirmed[confirmed["unchecked"] != ""]
    assert unchecked["result"].value_counts().to_dict() == {
        "not judged": 21,  # the b-arm pctFlagged baselines and the unlit IIS arcs
        "no metric": 1,  # the cloudy twilight: no metric measured on twilights sees cloud
    }


def testUncheckedEntriesFailWithoutTheirReason(sources):
    """Negative control: drop the reasons, and the entries naming an unjudged metric fail the check."""
    visitSet = loadValidationVisits()
    stripped = replace(
        visitSet, knownBad=tuple(replace(entry, unchecked=None) for entry in visitSet.knownBad)
    )
    coverage = knownBadCoverage(stripped, sources)
    assert (~coverage["ok"]).sum() == 22
    assert set(coverage.loc[~coverage["ok"], "metric"]) == {"pctFlagged", "nLines", "skyFluxUniformity"}


def testCaughtEntryMarkedUncheckedFails(sources):
    """Negative control: an unchecked entry that is caught has outlived its reason."""
    visitSet = loadValidationVisits()
    knownBad = tuple(
        replace(entry, unchecked="stale") if entry.metric == "linesPerFiber" else entry
        for entry in visitSet.knownBad
    )
    coverage = knownBadCoverage(replace(visitSet, knownBad=knownBad), sources)
    stale = coverage[coverage["metric"] == "linesPerFiber"]
    assert len(stale) == 4 and (stale["result"] == CAUGHT).all() and not stale["ok"].any()


def testLooserThresholdsMissTheDefocus(sources):
    """Negative control: with FAIL above every measured width, the SM1 defocus is missed, not caught."""
    iq = sources[0]
    loose = pd.DataFrame(
        [
            {
                "metric": "medFwhm",
                "warn": 9.0,
                "fail": 10.0,
                "higherIsWorse": True,
                "absolute": False,
                "provenance": "test",
            }
        ]
    )
    coverage = knownBadCoverage(loadValidationVisits(), [CoverageSource(iq.metrics, [loose], ("medFwhm",))])
    defocus = coverage[(coverage["metric"] == "medFwhm") & (coverage["measured"] > 0)]
    assert len(defocus) == 9 and (defocus["result"] == "missed").all() and not defocus["ok"].any()


def bad(**fields) -> ValidationVisitSet:
    """Return a set holding one known-bad entry on visit 1."""
    entry = ValidationVisit(
        visits=(1,), sequenceType="scienceArc", **({"expect": "FAIL", "metric": "m"} | fields)
    )
    return ValidationVisitSet(knownBad=(entry,))


def rows(*values, arm="b", spectrograph=1) -> pd.DataFrame:
    """Return metrics rows of visit 1, one per value of ``m``."""
    return pd.DataFrame([{"visit": 1, "arm": arm, "spectrograph": spectrograph, "m": v} for v in values])


def threshold(warn=1.0, fail=2.0) -> pd.DataFrame:
    """Return a thresholds table judging ``m``, higher is worse."""
    return pd.DataFrame(
        [
            {
                "metric": "m",
                "warn": warn,
                "fail": fail,
                "higherIsWorse": True,
                "absolute": False,
                "provenance": "test",
            }
        ]
    )


@pytest.mark.parametrize(
    ("frame", "fields", "result"),
    [
        (rows(3.0), {}, CAUGHT),
        (rows(1.5), {}, "missed"),  # WARN on a FAIL entry
        (rows(1.5), {"expect": "WARN"}, CAUGHT),
        (rows(3.0), {"expect": "WARN"}, CAUGHT),  # FAIL reaches a WARN entry
        (rows(np.nan), {}, "not measured"),
        (rows(3.0).assign(visit=2), {}, "no rows"),
        (rows(3.0), {"metric": "other"}, "no metric"),
        (rows(3.0), {"metric": None}, "no metric"),
        # Several rows per image (species): the image is judged at its worst.
        (rows(0.5, 3.0), {}, CAUGHT),
        # Negative control: an image whose rows are all below FAIL is missed, whatever the count.
        (rows(0.5, 1.5), {}, "missed"),
    ],
)
def testResults(frame, fields, result):
    coverage = knownBadCoverage(bad(**fields), [CoverageSource(frame, [threshold()], ("m",))])
    assert coverage["result"].tolist() == [result]


def testImageWithoutAVerdictIsNotJudged():
    """A measured image the thresholds don't judge leaves the entry not judged, even beside a caught one."""
    frame = pd.concat([rows(3.0), rows(3.0, arm="r")], ignore_index=True)
    table = threshold().assign(arm="b")
    (row,) = knownBadCoverage(bad(), [CoverageSource(frame, [table], ("m",))]).to_dict("records")
    assert (row["images"], row["measured"], row["judged"], row["reached"]) == (2, 2, 1, 1)
    assert row["result"] == "not judged" and not row["ok"]
    # With the reason recorded, the same entry is ok.
    (row,) = knownBadCoverage(
        bad(unchecked="r not judged"), [CoverageSource(frame, [table], ("m",))]
    ).to_dict("records")
    assert row["ok"]


def testUnconfirmedEntriesAreReportedNotDecided():
    """An unconfirmed entry is reported with its counts but is always ok."""
    coverage = knownBadCoverage(bad(unconfirmed=True), [CoverageSource(rows(0.5), [threshold()], ("m",))])
    assert coverage["result"].tolist() == ["missed"] and coverage["ok"].all()
