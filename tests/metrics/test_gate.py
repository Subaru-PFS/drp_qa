"""Tests for the gate: judging metrics against layered threshold tables."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.calibration import readThresholds
from pfs.drp.qa.metrics.gate import configThresholds, gate, judge, loadThresholds

ROOT = Path(__file__).parents[2]
DATA = Path(__file__).parent / "data"
RUN25 = DATA / "iqQaThresholds-run25.yaml"


def configDefaults() -> SimpleNamespace:
    """Return the threshold defaults of ``ImageQualityQaConfig``, read from its source.

    The task module imports the stack, so the defaults are read with `ast`
    rather than by instantiating the config: a changed default changes this.
    """
    source = (ROOT / "python" / "pfs" / "drp" / "qa" / "imageQualityQa.py").read_text()
    config = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "ImageQualityQaConfig"
    )
    defaults = {}
    for statement in config.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            for keyword in statement.value.keywords:
                if keyword.arg == "default":
                    defaults[statement.targets[0].id] = ast.literal_eval(keyword.value)
    return SimpleNamespace(**defaults)


@pytest.fixture(scope="module")
def stored() -> pd.DataFrame:
    """Return the validation visits' ``iqQaMetrics`` with the verdicts the task stored."""
    return pd.read_parquet(DATA / "iqQaMetrics-validation.parquet")


def image(**values) -> pd.DataFrame:
    """Return one image's metrics row."""
    base = {"visit": 1, "arm": "b", "spectrograph": 1, "obsType": "arc", "seqName": "Arc: Neon"}
    return pd.DataFrame([base | values])


def table(*entries) -> pd.DataFrame:
    """Return a thresholds table, higher-is-worse unless an entry says otherwise."""
    rows = [{"higherIsWorse": True, "absolute": False, "provenance": "test"} | entry for entry in entries]
    return pd.DataFrame(rows)


def testConfigReproducesStoredVerdicts(stored):
    """The config defaults as a table give every verdict the task stored."""
    verdicts = gate(stored, configThresholds(configDefaults()))
    assert (verdicts["qaStatus"] == stored["qaStatus"]).all()
    # The data cover every verdict and every deciding metric.
    assert set(stored["qaStatus"]) == {"PASS", "WARN", "FAIL"}
    assert set(verdicts["qaDecidedBy"]) == {"", "medFwhm", "pctFlagged", "medDxCenter"}


def testConfigRegressionHasTeeth(stored):
    """Negative controls: the comparison fails without per-arm or per-species keys."""
    blended = configDefaults()
    blended.flagRateWarnThreshold = {}
    blended.flagRateFailThreshold = {}
    assert (gate(stored, configThresholds(blended))["qaStatus"] != stored["qaStatus"]).sum() > 50

    armOnly = configDefaults()
    armOnly.flagRateWarnThreshold = {k: v for k, v in armOnly.flagRateWarnThreshold.items() if ":" not in k}
    armOnly.flagRateFailThreshold = {k: v for k, v in armOnly.flagRateFailThreshold.items() if ":" not in k}
    assert (gate(stored, configThresholds(armOnly))["qaStatus"] != stored["qaStatus"]).sum() > 0


def testReadsRun25(stored):
    """The Run25 file gates the arcs it has populations for; the config takes the rest."""
    thresholds = [RUN25, configThresholds(configDefaults())]
    judged = judge(stored, thresholds).merge(stored[["obsType", "seqName"]], left_on="row", right_index=True)
    fwhm = judged[judged["metric"] == "medFwhm"]
    assert (fwhm.loc[fwhm["obsType"] == "arc", "layer"] == 0).all()
    assert (fwhm.loc[fwhm["obsType"] != "arc", "layer"] == 1).all()
    # Run25 has no b-arm Argon flag rate: the config's b:Argon entry judges it.
    argon = judged[
        (judged["metric"] == "pctFlagged") & (judged["arm"] == "b") & (judged["seqName"] == "Arc: Argon")
    ]
    assert len(argon) and (argon["population"] == "arm=b species=Argon").all()
    assert (argon["layer"] == 1).all()
    assert judged.loc[judged["metric"] == "medDxCenter", "layer"].eq(1).all()

    # The obstructed frames fail, the traces of them included.
    verdicts = gate(stored, thresholds)
    obstructed = stored["visit"].isin([150115, 150116, 150641, 150642])
    assert (verdicts.loc[obstructed, "qaStatus"] == "FAIL").all()


def testLoadThresholds():
    """Paths are read, tables pass through, and order is kept."""
    config = configThresholds(configDefaults())
    tables = loadThresholds([RUN25, config])
    assert len(tables) == 2 and tables[1] is config
    pd.testing.assert_frame_equal(tables[0], readThresholds(RUN25)[1])
    assert len(loadThresholds(str(RUN25))) == 1
    assert loadThresholds([]) == []


def testLevels():
    """At a threshold is crossed; FAIL is tested before WARN."""
    thresholds = table({"metric": "medFwhm", "warn": 3.0, "fail": 3.5})
    for value, expected in ((2.9, "PASS"), (3.0, "WARN"), (3.49, "WARN"), (3.5, "FAIL"), (9.0, "FAIL")):
        assert gate(image(medFwhm=value), thresholds)["qaStatus"].iloc[0] == expected


def testMissingAndInfinite():
    """NaN gets no verdict; an infinite value is judged and fails."""
    thresholds = table({"metric": "medFwhm", "warn": 3.0, "fail": 3.5})
    judged = judge(image(medFwhm=np.nan), thresholds)
    assert judged["status"].iloc[0] == ""
    assert gate(image(medFwhm=np.nan), thresholds)["qaStatus"].iloc[0] == "PASS"
    assert gate(image(medFwhm=np.nan), thresholds, default="UNKNOWN")["qaStatus"].iloc[0] == "UNKNOWN"
    verdict = gate(image(medFwhm=math.inf), thresholds)
    assert verdict["qaStatus"].iloc[0] == "FAIL"
    assert verdict["qaReason"].iloc[0] == "medFWHM=infpx >= fail threshold 3.5px"


def testDirectionAndAbsolute():
    """A lower-is-worse entry fails low; an absolute one judges |value|."""
    low = table({"metric": "nLines", "warn": 100, "fail": 50, "higherIsWorse": False})
    assert gate(image(nLines=50), low)["qaStatus"].iloc[0] == "FAIL"
    assert gate(image(nLines=99), low)["qaStatus"].iloc[0] == "WARN"
    assert gate(image(nLines=1000), low)["qaStatus"].iloc[0] == "PASS"

    offset = table({"metric": "medDxCenter", "warn": 1.0, "fail": 2.0, "absolute": True})
    verdict = gate(image(medDxCenter=-2.5), offset)
    assert verdict["qaStatus"].iloc[0] == "FAIL"
    assert verdict["qaReason"].iloc[0] == "|dxCenter|=2.500px >= fail threshold 2px"
    # Negative control: the same entry without ``absolute`` passes it.
    signed = offset.assign(absolute=False)
    assert gate(image(medDxCenter=-2.5), signed)["qaStatus"].iloc[0] == "PASS"


def testMostSpecificEntryWins():
    """``arm`` + ``species`` beats ``arm``, which beats an entry naming nothing."""
    thresholds = table(
        {"metric": "pctFlagged", "warn": 15.0, "fail": 20.0},
        {"metric": "pctFlagged", "arm": "b", "warn": 50.0, "fail": 60.0},
        {"metric": "pctFlagged", "arm": "b", "species": "Neon", "warn": 80.0, "fail": 90.0},
    )
    metrics = pd.concat(
        [
            image(pctFlagged=70.0),
            image(pctFlagged=70.0, seqName="Arc: Argon"),
            image(pctFlagged=70.0, arm="r"),
        ],
        ignore_index=True,
    )
    judged = judge(metrics, thresholds)
    assert list(judged["population"]) == ["arm=b species=Neon", "arm=b", ""]
    assert list(judged["status"]) == ["PASS", "FAIL", "FAIL"]


def testEarlierTableWins():
    """A table's match, however general, beats any match in a later table."""
    first = table({"metric": "medFwhm", "warn": 2.0, "fail": 2.5})
    second = table({"metric": "medFwhm", "arm": "b", "obsType": "arc", "warn": 3.0, "fail": 3.5})
    judged = judge(image(medFwhm=2.7), [first, second])
    assert judged["layer"].iloc[0] == 0 and judged["status"].iloc[0] == "FAIL"
    judged = judge(image(medFwhm=2.7, obsType="trace", arm="r"), [second, first])
    assert judged["layer"].iloc[0] == 1


def testEntryWithoutThresholdsStopsTheSearch():
    """A matching entry with no thresholds means "not judged here", not "look further"."""
    config = configThresholds(configDefaults())
    judged = judge(image(medFwhm=9.0, traceOnly=True, pctFlagged=0.0, medDxCenter=0.0), config)
    fwhm = judged[judged["metric"] == "medFwhm"].iloc[0]
    assert fwhm["status"] == "" and fwhm["population"] == "traceOnly=True"
    judged = judge(image(medFwhm=9.0, traceOnly=False, pctFlagged=0.0, medDxCenter=0.0), config)
    assert judged.loc[judged["metric"] == "medFwhm", "status"].iloc[0] == "FAIL"


def testTiesAndMissingColumns():
    """Equal specificity and unmeasured metrics are errors; an absent column matches nothing."""
    tied = table(
        {"metric": "medFwhm", "arm": "b", "warn": 3.0, "fail": 3.5},
        {"metric": "medFwhm", "obsType": "arc", "warn": 3.0, "fail": 3.5},
    )
    with pytest.raises(ValueError, match="equally specific"):
        gate(image(medFwhm=2.0), tied)
    # Ties among entries that don't match are harmless.
    assert gate(image(medFwhm=2.0, arm="r", obsType="trace"), tied)["qaStatus"].iloc[0] == "PASS"

    with pytest.raises(KeyError, match="medFwhm"):
        gate(image(pctFlagged=1.0), tied)

    thresholds = table({"metric": "medFwhm", "camera": "b1", "warn": 3.0, "fail": 3.5})
    judged = judge(image(medFwhm=9.0), thresholds)
    assert judged["layer"].iloc[0] == -1 and judged["status"].iloc[0] == ""


def testVerdictColumns():
    """The worst verdict, the first metric reaching it, and every WARN/FAIL reason."""
    thresholds = table(
        {"metric": "medFwhm", "warn": 3.0, "fail": 3.5},
        {"metric": "pctFlagged", "warn": 15.0, "fail": 20.0},
        {"metric": "medDxCenter", "warn": 1.0, "fail": 2.0, "absolute": True},
    )
    metrics = pd.concat(
        [
            image(medFwhm=3.2, pctFlagged=25.0, medDxCenter=3.0),
            image(medFwhm=2.0, pctFlagged=5.0, medDxCenter=0.1),
        ],
        ignore_index=True,
    ).set_axis([10, 20])
    verdicts = gate(metrics, thresholds)
    assert list(verdicts.index) == [10, 20]
    assert list(verdicts["qaStatus"]) == ["FAIL", "PASS"]
    assert list(verdicts["qaDecidedBy"]) == ["pctFlagged", ""]
    assert verdicts["qaReason"].iloc[0] == (
        "medFWHM=3.20px >= warn threshold 3px; pctFlagged=25.0% >= fail threshold 20%; "
        "|dxCenter|=3.000px >= fail threshold 2px"
    )
    assert verdicts["qaReason"].iloc[1] == ""


def testConfigKeysResolvePerSide():
    """A key in one dict takes the other side from its arm, as the task's lookup did."""
    config = configDefaults()
    config.flagRateWarnThreshold = {"b": 50.0, "b:Argon": 93.0}
    config.flagRateFailThreshold = {"b": 60.0}
    entries = configThresholds(config)
    argon = entries[(entries["metric"] == "pctFlagged") & (entries["species"] == "Argon")].iloc[0]
    assert (argon["warn"], argon["fail"]) == (93.0, 60.0)
    assert (
        gate(
            image(seqName="Arc: Argon", pctFlagged=61.0, medFwhm=2.0, medDxCenter=0.0, traceOnly=False),
            entries,
        )["qaStatus"].iloc[0]
        == "FAIL"
    )
