"""Tests for the gate: judging metrics against layered threshold tables."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from pfs.drp.qa.metrics.calibration import labelRows, readThresholds
from pfs.drp.qa.metrics.gate import (
    PROVISIONAL_COLUMNS,
    VERDICT_COLUMNS,
    configThresholds,
    gate,
    judge,
    layerPipelineMismatch,
    loadThresholds,
    pipelineMismatch,
    thresholdsPath,
)
from pfs.drp.qa.metrics.validationVisits import loadValidationVisits

ROOT = Path(__file__).parents[2]
DATA = Path(__file__).parent / "data"
RUN25 = DATA / "iqQaThresholds-run25.yaml"
IIS = DATA / "iqQaThresholds-iis-run30.yaml"


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


@pytest.fixture(scope="module")
def configStored() -> pd.DataFrame:
    """Return ``iqQaMetrics`` from a run whose verdicts the task made from its config alone."""
    return pd.read_parquet(DATA / "iqQaMetrics-config.parquet")


def image(**values) -> pd.DataFrame:
    """Return one image's metrics row."""
    base = {"visit": 1, "arm": "b", "spectrograph": 1, "obsType": "arc", "seqName": "Arc: Neon"}
    return pd.DataFrame([base | values])


def table(*entries) -> pd.DataFrame:
    """Return a thresholds table, higher-is-worse unless an entry says otherwise."""
    rows = [{"higherIsWorse": True, "absolute": False, "provenance": "test"} | entry for entry in entries]
    return pd.DataFrame(rows)


def testConfigReproducesStoredVerdicts(configStored):
    """The config defaults as a table give every verdict the task stored."""
    verdicts = gate(configStored, configThresholds(configDefaults()))
    assert (verdicts["qaStatus"] == configStored["qaStatus"]).all()
    # The data cover every verdict and every deciding metric.
    assert set(configStored["qaStatus"]) == {"PASS", "WARN", "FAIL"}
    assert set(verdicts["qaDecidedBy"]) == {"", "medFwhm", "pctFlagged", "medDxCenter"}


def testConfigRegressionHasTeeth(configStored):
    """Negative controls: the comparison fails without per-arm or per-species keys."""
    stored = configStored
    blended = configDefaults()
    blended.flagRateWarnThreshold = {}
    blended.flagRateFailThreshold = {}
    assert (gate(stored, configThresholds(blended))["qaStatus"] != stored["qaStatus"]).sum() > 50

    armOnly = configDefaults()
    armOnly.flagRateWarnThreshold = {k: v for k, v in armOnly.flagRateWarnThreshold.items() if ":" not in k}
    armOnly.flagRateFailThreshold = {k: v for k, v in armOnly.flagRateFailThreshold.items() if ":" not in k}
    assert (gate(stored, configThresholds(armOnly))["qaStatus"] != stored["qaStatus"]).sum() > 0


def testReadsRun25(stored):
    """The Run25 file gates the arcs and quartz it has populations for; the config takes the rest."""
    thresholds = [RUN25, configThresholds(configDefaults())]
    judged = judge(stored, thresholds).merge(stored[["obsType", "seqName"]], left_on="row", right_index=True)
    fwhm = judged[judged["metric"] == "medFwhm"]
    derived = fwhm["obsType"].isin(["arc", "trace"])
    assert (fwhm.loc[derived, "layer"] == 0).all()
    assert (fwhm.loc[~derived, "layer"] == 1).all()
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


def testIlluminationDefaultsToFull():
    """An entry without ``illumination`` judges full-slit images only; an IIS image needs one naming it."""
    full = table({"metric": "medFwhm", "arm": "b", "warn": 3.0, "fail": 3.5})
    assert gate(image(medFwhm=4.0), full)["qaStatus"].iloc[0] == "FAIL"
    assert gate(image(medFwhm=4.0, illumination="full"), full)["qaStatus"].iloc[0] == "FAIL"
    assert (
        gate(image(medFwhm=4.0, illumination="iis"), full, default="UNKNOWN")["qaStatus"].iloc[0] == "UNKNOWN"
    )

    both = table(
        {"metric": "medFwhm", "illumination": "iis", "warn": 5.0, "fail": 6.0},
        {"metric": "medFwhm", "arm": "b", "warn": 3.0, "fail": 3.5},
    )
    images = pd.concat(
        [image(medFwhm=4.0), image(visit=2, medFwhm=4.0, illumination="iis")], ignore_index=True
    )
    judged = judge(images, both)
    assert judged["population"].tolist() == ["arm=b", "illumination=iis"]
    assert judged["status"].tolist() == ["FAIL", "PASS"]
    # Negative control: by specificity alone the arm=b entry (1 column) ties the IIS one, which
    # would be an error, and an entry naming two columns would take the IIS image.
    specific = table(
        {"metric": "medFwhm", "illumination": "iis", "warn": 5.0, "fail": 6.0},
        {"metric": "medFwhm", "arm": "b", "obsType": "arc", "warn": 3.0, "fail": 3.5},
    )
    assert judge(images, specific)["population"].tolist() == ["arm=b obsType=arc", "illumination=iis"]


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
    skip = table({"metric": "medFwhm", "obsType": "trace", "warn": None, "fail": None})
    config = configThresholds(configDefaults())
    judged = judge(image(medFwhm=9.0, obsType="trace", pctFlagged=0.0, medDxCenter=0.0), [skip, config])
    fwhm = judged[judged["metric"] == "medFwhm"].iloc[0]
    assert fwhm["status"] == "" and fwhm["population"] == "obsType=trace" and fwhm["layer"] == 0
    judged = judge(image(medFwhm=9.0, obsType="arc", pctFlagged=0.0, medDxCenter=0.0), [skip, config])
    assert judged.loc[judged["metric"] == "medFwhm", "status"].iloc[0] == "FAIL"


def testCalibrationValuesAreNotJudged():
    """A FWHM read from fiberProfiles gets no verdict from any table; a measured one does."""
    entry = table({"metric": "medFwhm", "obsType": "trace", "arm": "b", "warn": 3.0, "fail": 3.5})
    config = configThresholds(configDefaults())
    others = {"pctFlagged": 0.0, "medDxCenter": 0.0}
    for thresholds in (entry, config, [entry, config]):
        judged = judge(image(medFwhm=9.0, obsType="trace", traceOnly=True, **others), thresholds)
        assert judged.loc[judged["metric"] == "medFwhm", "status"].iloc[0] == ""
        # Negative control: the same value, measured, fails.
        judged = judge(image(medFwhm=9.0, obsType="trace", traceOnly=False, **others), thresholds)
        assert judged.loc[judged["metric"] == "medFwhm", "status"].iloc[0] == "FAIL"
    # Other metrics of the same image are still judged.
    judged = judge(image(medFwhm=9.0, traceOnly=True, pctFlagged=99.0, medDxCenter=0.0), config)
    assert judged.loc[judged["metric"] == "pctFlagged", "status"].iloc[0] == "FAIL"


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


def testThresholdsPath(tmp_path):
    """A relative name is a shipped file; an absolute path is taken as given."""
    shipped = thresholdsPath(configDefaults().thresholdsFile)
    assert shipped.parent == ROOT / "python" / "pfs" / "drp" / "qa" / "metrics" / "data"
    assert shipped.exists()
    assert thresholdsPath(str(tmp_path / "x.yaml")) == tmp_path / "x.yaml"


def testShippedFileIsTheAdoptedRun25(stored):
    """The shipped file is what its script makes from the Run25 derivation."""
    script = thresholdsPath("makeRun25Thresholds.py")
    spec = importlib.util.spec_from_file_location("makeRun25Thresholds", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    remade = module.adopt(
        yaml.safe_load(RUN25.read_text()), pd.read_parquet(DATA / "iqQaMetrics-validation.parquet")
    )
    assert remade == yaml.safe_load(thresholdsPath("iqQaThresholds-run25.yaml").read_text())


def loadScript(name: str):
    """Return the thresholds-making script ``name`` beside the shipped files, imported as a module."""
    spec = importlib.util.spec_from_file_location(name, thresholdsPath(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def iisStored() -> pd.DataFrame:
    """Return Run30's nightly IIS frames' ``iqQaMetrics``, reduced afresh (u/wtg/qa-iis/003b)."""
    return pd.read_parquet(DATA / "iqQaMetrics-iis-run30.parquet")


def testShippedIisFileIsTheAdoptedRun30(iisStored, tmp_path):
    """The IIS file is what its script derives from the Run30 IIS frames and adopts (PIPE2D-1954)."""
    module = loadScript("makeIisThresholds")
    derived = module.derive(iisStored, tmp_path / "derived.yaml")
    assert yaml.safe_load(derived.read_text()) == yaml.safe_load(IIS.read_text())
    remade = module.adopt(yaml.safe_load(IIS.read_text()), iisStored)
    assert remade == yaml.safe_load(thresholdsPath("iqQaThresholds-iis-run30.yaml").read_text())
    assert all(entry["population"]["illumination"] == "iis" for entry in remade["thresholds"])


def testShippedIisFileVerdicts(iisStored):
    """IIS frames are judged by the IIS file alone: the known-good nights pass, the two faults warn."""
    files = [thresholdsPath(configDefaults().thresholdsFile)]
    files += [thresholdsPath(name) for name in configDefaults().extraThresholdsFiles]
    thresholds = [*files, configThresholds(configDefaults())]
    metrics = iisStored.drop(columns=list(VERDICT_COLUMNS))
    verdicts = gate(metrics, thresholds, default="UNKNOWN")
    judged = judge(metrics, thresholds)
    assert (metrics["illumination"] == "iis").all()
    # Neither the Run25 file (layer 0) nor the config (layer 2) judges an IIS image.
    assert set(judged.loc[judged["status"] != "", "layer"]) == {1}
    flagged = metrics.assign(qaStatus=verdicts["qaStatus"], qaDecidedBy=verdicts["qaDecidedBy"])
    flagged = flagged[flagged["qaStatus"] != "PASS"]
    assert flagged[["visit", "arm", "spectrograph", "qaStatus", "qaDecidedBy"]].to_records(
        index=False
    ).tolist() == [
        (148829, "n", 4, "WARN", "medDxCenter"),  # IIS lamp header off: reduced against the sky line list
        (149595, "b", 2, "WARN", "medFwhm"),  # the night SM2 and SM4 moved; not a known-good night
    ]
    # Every known-good image has its FWHM judged, the n-arm Ne arcs and the m arm excepted.
    fwhm = judged[judged["metric"] == "medFwhm"].merge(metrics[["seqName"]], left_on="row", right_index=True)
    unjudged = fwhm[fwhm["status"] == ""]
    assert set(unjudged["arm"]) == {"m", "n"}
    assert set(unjudged.loc[unjudged["arm"] == "n", "seqName"]) == {"ImageQuality (Ne)"}
    # Negative control: a full-slit image is not judged by the IIS file.
    full = image(medFwhm=1.0, linesPerFiber=1.0, medDxCenter=0.0)
    assert (judge(full, [IIS])["status"] == "").all()
    # An IIS lamp with no entry of its own (an IIS Ar arc) has no FWHM reference: not judged, not failed.
    argon = image(
        seqName="ImageQuality (Ar)",
        illumination="iis",
        medFwhm=1.95,
        linesPerFiber=50.0,
        medDxCenter=0.0,
        pctFlagged=np.nan,
    )
    judgedArgon = judge(argon, thresholds).set_index("metric")
    assert judgedArgon.loc["medFwhm", "status"] == "" and judgedArgon.loc["medDxCenter", "status"] == "PASS"


def testShippedFileVerdicts(stored):
    """The adopted file: obstructed frames and defocused quartz fail, no known-good fails; flags aren't judged.

    The fixture is ``u/wtg/qa-thresholds/010`` (PIPE2D-1938): the validation visits reduced afresh,
    Run29's b2 and n4 quartz among them. The one-group Run30 arcs that PIPE2D-1935 checked are no
    longer validation visits; comparison mode judges them with every other daily arc.
    """
    thresholds = [thresholdsPath(configDefaults().thresholdsFile), configThresholds(configDefaults())]
    judged = judge(stored, thresholds)
    verdicts = gate(stored, thresholds, default="UNKNOWN")

    obstructed = stored["visit"].isin([150115, 150116, 150641, 150642])
    assert (verdicts.loc[obstructed, "qaStatus"] == "FAIL").all()
    # The validation set records them as known-bad for linesPerFiber, the metric that fails them
    # (PIPE2D-1935): each lit fiber measures fewer lines.
    lines = labelRows(stored, loadValidationVisits(), "linesPerFiber")
    perFiber = judged[judged["metric"] == "linesPerFiber"].set_index("row")
    obstructedLines = lines[lines["visit"].isin([150115, 150116, 150641, 150642])]
    assert len(obstructedLines) == obstructed.sum() and (obstructedLines["validation"] == "bad:FAIL").all()
    assert (perFiber.loc[obstructedLines.index, "status"] == "FAIL").all()
    # Every known-good arc and quartz passes it, but b Argon, under a line per fiber, isn't judged.
    goodLines = perFiber.loc[lines.index[lines["validation"] == "good"]]
    goodLines = goodLines[goodLines["value"].notna()]
    argon = stored.loc[goodLines.index, "seqName"].eq("Arc: Argon") & stored.loc[goodLines.index, "arm"].eq(
        "b"
    )
    assert (goodLines.loc[~argon, "status"] == "PASS").all() and (goodLines.loc[argon, "status"] == "").all()
    run25Quartz = (stored["seqName"] == "Trace") & stored["visit"].between(133040, 135850)
    assert (perFiber.loc[np.flatnonzero(run25Quartz), "status"] == "PASS").all()
    # medFwhm (PIPE2D-1942): every known-good arc and quartz is judged against its own detector and
    # lamp's reference median, and none fails; every known-bad one with a value fails. b-arm Argon
    # is not judged (PIPE2D-1938): under a line per fiber, its width rests on a handful of lines.
    labelled = labelRows(stored, loadValidationVisits(), "medFwhm")
    fwhm = judged[judged["metric"] == "medFwhm"].set_index("row")
    bArgon = (stored["arm"] == "b") & (stored["seqName"] == "Arc: Argon")
    good = fwhm.loc[labelled.index[labelled["validation"] == "good"]]
    good = good[good["value"].notna()]
    assert len(good) == 636 and good["population"].str.contains("spectrograph=").all()
    assert (good["layer"] == 0).all() and (good["status"] != "FAIL").all()
    assert good["status"].value_counts().to_dict() == {
        "PASS": 595,
        "": 40,
        "WARN": 1,
    }  # 135841 m1 Argon, +3.3 %
    assert bArgon[good.index[good["status"] == ""]].all() and bArgon[good.index].sum() == 40
    bad = fwhm.loc[labelled.index[labelled["validation"] == "bad:FAIL"]]
    bad = bad[bad["value"].notna()]
    assert len(bad) == 82 and (bad["status"] == "FAIL").all()
    assert not bArgon[bad.index].any()  # the set asserts nothing about b-arm Argon widths
    # The b2 and n4 quartz reference is Run29's set (PIPE2D-1938): its traces pass on those two cameras,
    # and Run25's own, narrower, pass as well.
    camera = stored["arm"].add(stored["spectrograph"].astype(str))
    run29Quartz = (stored["seqName"] == "Trace") & stored["visit"].between(144587, 147830)
    moved = run29Quartz & camera.isin(["b2", "n4"])
    assert moved.sum() == 42 and (verdicts.loc[moved, "qaStatus"] == "PASS").all()
    run25B2 = run25Quartz & (camera == "b2")
    assert run25B2.sum() == 14 and (verdicts.loc[run25B2, "qaStatus"] == "PASS").all()
    # Negative control: against a tolerance on Run25's own b2 and n4 medians, Run29's b2 quartz
    # would FAIL (5 % above) and n4's WARN (3 % above).
    run25Median = stored.loc[run25Quartz & camera.isin(["b2", "n4"])].groupby(camera)["medFwhm"].median()
    run29Median = stored.loc[moved].groupby(camera)["medFwhm"].median()
    assert run29Median["b2"] > 1.05 * run25Median["b2"] and run29Median["n4"] > 1.03 * run25Median["n4"]
    # Negative control: the per-arm entries alone fail known-good arcs and quartz.
    perArm = readThresholds(thresholdsPath(configDefaults().thresholdsFile))[1]
    perArm = perArm[perArm["method"] != "tolerance"].drop(columns=["spectrograph", "species"])
    perArmFwhm = judge(stored, [perArm, configThresholds(configDefaults())]).set_index("row")
    perArmFwhm = perArmFwhm[perArmFwhm["metric"] == "medFwhm"]
    assert (perArmFwhm.loc[good.index, "status"] == "FAIL").sum() >= 3
    # SM1 defocused (Run27): every SM1 quartz fails on medFwhm.
    defocused = np.flatnonzero(
        stored["visit"].isin([140032, 140035, *range(140139, 140149)]) & (stored["spectrograph"] == 1)
    )
    defocusedFwhm = judged[(judged["metric"] == "medFwhm") & judged["row"].isin(defocused)]
    assert len(defocusedFwhm) == len(defocused) and (defocusedFwhm["status"] == "FAIL").all()
    # Negative control: the derived (degenerate) quartz entries fail some of them.
    derived = gate(stored, [RUN25, configThresholds(configDefaults())])
    assert (derived.loc[run25Quartz, "qaStatus"] == "FAIL").any()

    # nLines is no longer judged (PIPE2D-1935), nor is a twilight frame's count.
    assert "nLines" not in set(judged["metric"])
    twilight = perFiber.loc[np.flatnonzero(stored["seqName"] == "Twilight sky")]
    assert len(twilight) and (twilight["status"] == "").all()
    # pctFlagged is not judged at all: the shipped file's entry stops the search before the config.
    flags = judged[judged["metric"] == "pctFlagged"]
    assert len(flags) and (flags["layer"] == 0).all() and (flags["status"] == "").all()
    # Negative control: the config alone judges them, some badly.
    configOnly = judge(stored, [configThresholds(configDefaults())])
    assert configOnly.loc[configOnly["metric"] == "pctFlagged", "status"].isin(["WARN", "FAIL"]).any()


PIPELINE = {"datamodel": "w.2026.40", "obs_pfs": "w.2026.40", "drp_stella": "w.2026.41-6-ge56938d2"}


def reductions(*drpStella) -> pd.DataFrame:
    """Images reduced with PIPELINE but for drp_stella's version (None: not recorded)."""
    return pd.DataFrame(
        {
            "medFwhm": 2.0,
            "versionDatamodel": [None if v is None else "w.2026.40" for v in drpStella],
            "versionObsPfs": [None if v is None else "w.2026.40" for v in drpStella],
            "versionDrpStella": list(drpStella),
        },
        index=[10 + i for i in range(len(drpStella))],
    )


def testPipelineMismatch():
    """Only the image reduced with the thresholds' pipeline gets a firm verdict."""
    metrics = reductions("w.2026.41-6-ge56938d2", "w.2026.38", None)
    result = pipelineMismatch(metrics, PIPELINE)
    assert list(result.columns) == list(PROVISIONAL_COLUMNS)
    assert result.index.tolist() == metrics.index.tolist()
    assert result["qaProvisional"].tolist() == [False, True, True]
    assert result["qaPipeline"].tolist() == [
        "",
        "drp_stella w.2026.38 (thresholds w.2026.41-6-ge56938d2)",
        "reduction's pipeline unknown",
    ]


def testPipelineMismatchWithoutAReference():
    """Thresholds that record no pipeline, or only part of it, can't vouch for any reduction."""
    metrics = reductions("w.2026.41-6-ge56938d2")
    assert pipelineMismatch(metrics, {})["qaPipeline"].tolist() == ["the thresholds record no pipeline"]
    partial = {key: value for key, value in PIPELINE.items() if key != "obs_pfs"}
    assert pipelineMismatch(metrics, partial)["qaPipeline"].tolist() == [
        "obs_pfs w.2026.40 (thresholds unknown)"
    ]
    # No version columns at all, as in iqQaMetrics read without its pfsArm.
    assert pipelineMismatch(metrics[["medFwhm"]], PIPELINE)["qaProvisional"].all()


def testLayerPipelineMismatch():
    """Each image is provisional against the pipeline of the layer that judged it (PIPE2D-1954)."""
    run25 = table({"metric": "medFwhm", "warn": 3.0, "fail": 4.0})
    iis = table({"metric": "medFwhm", "illumination": "iis", "warn": 2.0, "fail": 2.5})
    iisPipeline = PIPELINE | {"obs_pfs": "w.2026.43"}
    images = pd.concat(
        [image(medFwhm=2.0), image(visit=2, medFwhm=1.9, illumination="iis"), image(visit=3, medFwhm=np.nan)],
        ignore_index=True,
    )
    for column, value in (("versionDatamodel", "w.2026.40"), ("versionDrpStella", "w.2026.41-6-ge56938d2")):
        images[column] = value
    images["versionObsPfs"] = [
        "w.2026.40",
        "w.2026.43",
        "w.2026.43",
    ]  # each reduced on its own file's pipeline
    judged = judge(images, [run25, iis])
    assert judged.loc[judged["status"] != "", "layer"].tolist() == [0, 1]  # the third image's FWHM is NaN
    mismatch = layerPipelineMismatch(images, judged, [PIPELINE, iisPipeline])
    # The full-slit and IIS images match their own layers; the unjudged image is compared with the first.
    assert mismatch["qaProvisional"].tolist() == [False, False, True]
    assert mismatch["qaPipeline"].iloc[2] == "obs_pfs w.2026.43 (thresholds w.2026.40)"
    # Negative control: against one pipeline, as before, the IIS image reads as provisional.
    assert pipelineMismatch(images, PIPELINE)["qaProvisional"].tolist() == [False, True, True]
    # Through gate, and a layer without a pipeline (the config's) checks nothing.
    verdicts = gate(images, [run25, iis], default="UNKNOWN", pipeline=[PIPELINE, iisPipeline, None])
    assert verdicts["qaProvisional"].tolist() == [False, False, True]
    assert gate(images, [run25, iis], pipeline=[None, None])["qaProvisional"].tolist() == [
        False,
        False,
        False,
    ]


def testGateMarksProvisionalOnlyWhenAsked():
    thresholds = pd.DataFrame([{"metric": "medFwhm", "warn": 3.0, "fail": 4.0, "higherIsWorse": True}])
    metrics = reductions("w.2026.41-6-ge56938d2", "w.2026.38")
    plain = gate(metrics, thresholds)
    assert list(plain.columns) == list(VERDICT_COLUMNS)
    checked = gate(metrics, thresholds, pipeline=PIPELINE)
    assert list(checked.columns) == [*VERDICT_COLUMNS, *PROVISIONAL_COLUMNS]
    # Provisional qualifies the verdict; it doesn't change it.
    assert checked["qaStatus"].tolist() == plain["qaStatus"].tolist() == ["PASS", "PASS"]
    assert checked["qaProvisional"].tolist() == [False, True]


def testUnknownIsNotAVersion():
    """Both sides reading "unknown" is no match."""
    metrics = reductions("w.2026.41-6-ge56938d2").assign(versionObsPfs="unknown")
    result = pipelineMismatch(metrics, PIPELINE | {"obs_pfs": "unknown"})
    assert result["qaPipeline"].tolist() == ["obs_pfs unknown (thresholds unknown)"]
    assert pipelineMismatch(metrics, dict.fromkeys(PIPELINE, "unknown"))["qaPipeline"].tolist() == [
        "the thresholds record no pipeline"
    ]


def testGateKeepsOneRowPerImageWithDuplicateLabels():
    """Rows that share an index label keep their own verdicts and flags, in order."""
    thresholds = pd.DataFrame([{"metric": "medFwhm", "warn": 3.0, "fail": 4.0, "higherIsWorse": True}])
    metrics = reductions("w.2026.41-6-ge56938d2", "w.2026.38").assign(medFwhm=[2.0, 5.0])
    metrics.index = [0, 0]
    result = gate(metrics, thresholds, pipeline=PIPELINE)
    assert len(result) == 2
    assert result["qaStatus"].tolist() == ["PASS", "FAIL"]
    assert result["qaProvisional"].tolist() == [False, True]
