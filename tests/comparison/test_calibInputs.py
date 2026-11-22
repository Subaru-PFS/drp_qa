"""Tests for `pfs.drp.qa.comparison.calibInputs` on a synthetic calibration set."""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.comparison.calibInputs import (
    LAMP_SPECIES,
    Rule,
    Thresholds,
    Verdict,
    buildReport,
    coverage,
    detectorTable,
    gateNumbers,
    gateVerdicts,
    goNoGo,
    judge,
    lampOf,
    lampsOf,
    pick,
    scatterLimits,
    setVisits,
    speciesTable,
    summary,
    trends,
)
from pfs.drp.qa.metrics import gate
from pfs.drp.qa.metrics.normalValues import NORMAL_COLUMNS, candidates
from pfs.drp.qa.metrics.registry import CHECKS, IIS_ARC

ARMS = {"n": (1, 2, 3, 4)}


def _listing():
    """Two arc sequences of three (Neon, HgCd), a trace sequence of two; one Neon visit never reduced."""
    rows = []
    start = pd.Timestamp("2026-10-31 03:30")
    for k in range(3):
        rows.append(
            (
                100 + k,
                70,
                "scienceArc",
                "Arc: Neon",
                "iic scienceArc neon=5 duplicate=3",
                5.0,
                start + pd.Timedelta(minutes=k),
            )
        )
    for k in range(3):
        rows.append(
            (
                110 + k,
                71,
                "scienceArc",
                "Arc: HgCd",
                "iic scienceArc hgcd=45 duplicate=3",
                45.0,
                start + pd.Timedelta(minutes=10 + k),
            )
        )
    for k in range(2):
        rows.append(
            (
                120 + k,
                72,
                "scienceTrace",
                "Trace",
                "iic scienceTrace halogen=30 duplicate=2",
                30.0,
                start + pd.Timedelta(minutes=20 + k),
            )
        )
    rows.append((130, 73, "darks", "dark", "iic dark exptime=10", 10.0, start + pd.Timedelta(minutes=30)))
    return pd.DataFrame(
        rows,
        columns=[
            "pfs_visit_id",
            "iic_sequence_id",
            "sequence_type",
            "sequence_name",
            "cmd_str",
            "exptime",
            "time_exp_start",
        ],
    ).assign(cameras="n1,n2,n3,n4")


def _metrics(visits, bad=None, fading=None, wide=None, drift=0.0):
    """Return iqQaMetrics, dmLineResidualMetrics and reductionStats for every detector image of ``visits``: numbers, no verdicts.

    ``bad``: {(visit, spectrograph)} whose detector image is wrong (tiny FWHM, huge reserved RMS,
    few lines); ``fading``: {visit: factor} scaling the lamp's line count; ``wide``: {(visit,
    spectrograph): factor} scaling the FWHM, to within the sequence rules; ``drift``: every
    image's offset (px) from the calibration in use.
    """
    bad, fading, wide = bad or set(), fading or {}, wide or {}
    iq, dm, reduction = [], [], []
    for row in visits.itertuples():
        isTrace = row.species == ("Trace",)
        obsType = "trace" if isTrace else "arc"
        for sp in ARMS["n"]:
            isBad = (row.visit, sp) in bad
            fwhm = 0.68 if isBad else (2.90 + 0.01 * sp) * wide.get((row.visit, sp), 1.0)
            rms = 2.2 if isBad else 0.08 + 0.005 * sp
            scale = fading.get(row.visit, 1.0) * (0.3 if isBad else 1.0)
            perFiber = (3000.0 if isTrace else 12.0) * scale
            iq.append((row.visit, "n", sp, obsType, "full", row.sequence_name, fwhm, perFiber, drift))
            # The reserved-line RMS lives in reductionStats now (whole-detector, empty amplifier).
            reduction.append((row.visit, "n", sp, "", "reservedXRms", rms))
            reduction.append((row.visit, "n", sp, "", "reservedYRms", rms))
            for species in row.species:
                lines = 1000 * scale if species != "Trace" else 2e6
                for statusType, factor in (("BOTH", 1.0), ("RESERVED", 0.1)):
                    for metric, value, offset in (
                        ("nLines", lines * factor, None),
                        ("linesPerFiber", lines * factor / 600, None),
                        ("robustRmsDeltaX", rms, None),
                        ("robustRmsDeltaY", rms, None),
                        ("medianDeltaX", 0.0, drift),
                        ("medianDeltaY", 0.0, drift),
                    ):
                        key = (row.visit, "n", sp, "brn", obsType)
                        dm.append((*key, "visit", species, statusType, metric, value))
                        dm.append(
                            (*key, "calib", species, statusType, metric, value if offset is None else offset)
                        )
    iq = pd.DataFrame(
        iq,
        columns=[
            "visit",
            "arm",
            "spectrograph",
            "obsType",
            "illumination",
            "seqName",
            "medFwhm",
            "linesPerFiber",
            "medDxCenter",
        ],
    )
    dm = pd.DataFrame(
        dm,
        columns=[
            "visit",
            "arm",
            "spectrograph",
            "cameraSet",
            "obsType",
            "reference",
            "description",
            "statusType",
            "metric",
            "value",
        ],
    )
    reduction = pd.DataFrame(
        reduction, columns=["visit", "arm", "spectrograph", "amplifier", "metric", "value"]
    )
    return iq, dm, reduction


@pytest.fixture
def visits():
    return setVisits(_listing())


@pytest.fixture
def setNormals(visits):
    """Return the normal values of a set like this one with nothing wrong: what a good image measures."""
    made = candidates(*_metrics(visits)[:2])
    return made.assign(count=9, validFrom=1).reindex(columns=list(NORMAL_COLUMNS))


@pytest.mark.parametrize(
    "cmdStr, lamps",
    [
        (
            "iic scienceArc exptime=5 head='sps iis on=neon warmingTime=150' tail='sps iis off=neon'",
            ["neon (IIS)"],
        ),
        ("iic scienceArc iisNeon=30 duplicate=2", ["neon (IIS)"]),
        ("iic scienceArc hgcd=0.0 argon=10 xenon=0.0 neon=0.0 krypton=0.0 duplicate=3", ["argon"]),
        ("iic scienceTrace halogen=20 duplicate=2", ["halogen"]),
        ("iic dark exptime=300", []),
        (None, []),
    ],
)
def testLampsOf(cmdStr, lamps):
    assert lampsOf(cmdStr) == lamps


def testLampOf():
    assert lampOf("iic scienceArc hgcd=45 duplicate=3") == "hgcd"
    assert lampOf("iic scienceArc iisNeon=40 forcePfsConfig", "scienceArc") == "neon (IIS)"
    assert lampOf(None, "scienceTrace") == "halogen"
    assert lampOf(None, "scienceArc") is None


def testSetVisits(visits):
    assert list(visits["visit"]) == [100, 101, 102, 110, 111, 112, 120, 121]  # the dark is not part of a set
    assert list(visits["k"]) == [1, 2, 3, 1, 2, 3, 1, 2]
    assert list(visits["n"]) == [3, 3, 3, 3, 3, 3, 2, 2]
    assert list(visits["lamp"]) == ["neon"] * 3 + ["hgcd"] * 3 + ["halogen"] * 2
    assert visits.loc[3, "species"] == ("HgI", "CdI")
    assert visits.loc[6, "species"] == ("Trace",)
    assert list(setVisits(_listing(), visits=[100, 101])["visit"]) == [100, 101]


def _byImage(table):
    return table.set_index(["visit", "spectrograph"])


def testDetectorGoneBadIsOutOnItsOwnExposuresOnly(visits, setNormals):
    # n4 breaks from the last HgCd exposure onwards: 112 and both traces.
    bad = {(112, 4), (120, 4), (121, 4)}
    iq, dm, reduction = _metrics(visits, bad=bad)
    judged = judge(detectorTable(visits, iq, dm, reduction, setNormals))
    outs = judged[judged["verdict"] == Verdict.OUT]
    assert set(zip(outs["visit"], outs["spectrograph"], strict=True)) == bad
    rest = judged[~judged.index.isin(outs.index)]
    assert (rest["verdict"] == Verdict.USE).all() and (rest["gateStatus"] == "PASS").all()
    row = _byImage(judged).loc[(112, 4)]
    assert "gate FAIL" in row["reason"] and "FWHM" in row["reason"] and "reserved RMS" in row["reason"]
    assert row["rules"] == f"{Rule.GATE_FAIL}, {Rule.FWHM}, {Rule.RMS}"
    # Negative control: the arm's median over detectors hides the fault, so a per-arm judgement would
    # have passed these exposures; the report must be per detector.
    armFwhm = judged.groupby("visit")["medFwhm"].median()
    assert (armFwhm / armFwhm.median() > 0.9).all()
    recommendation = pick(judged, visits).set_index("detector")
    assert recommendation.loc["n4", "out"] == [112, 120, 121]
    assert recommendation.loc["n3", "out"] == []
    assert recommendation.loc["n4", "measured"] == [100, 101, 102, 110, 111, 112, 120, 121]


def testTheVerdictIsTheGates(visits, setNormals):
    """The image's gate verdict is `gate.gate`'s on its numbers, and a gate FAIL alone is a no-go."""
    # 6 % wider than the camera's normal value: beyond the gate's FAIL, within the sequence rules.
    iq, dm, reduction = _metrics(visits, wide={(101, 2): 1.06})
    table = detectorTable(visits, iq, dm, reduction, setNormals)
    expected = gate.gate(
        iq=iq.drop(columns="medDxCenter"), lines=dm[dm["reference"] == "visit"], normalValues=setNormals
    )
    assert list(table["gateStatus"]) == list(expected["qaStatus"])
    assert list(table["gateReason"]) == list(expected["qaReason"])
    row = _byImage(judge(table)).loc[(101, 2)]
    assert (row["gateStatus"], row["gateDecidedBy"], row["gateChecks"]) == ("FAIL", "width", "width")
    assert row["gateReason"] == "width 3.10 px is +6.0 % from normal 2.92 px, limit +5 %"
    assert row["verdict"] == Verdict.OUT and row["reason"] == f"gate FAIL: {row['gateReason']}"
    assert row["rules"] == Rule.GATE_FAIL  # the sequence rules say nothing: 1.06x is within fwhmCheck
    assert (table["gateStatus"] == "PASS").sum() == len(table) - 1
    # Negative control: a width between WARN and FAIL is a check, and one under WARN is used.
    for factor, status, verdict in ((1.04, "WARN", Verdict.CHECK), (1.02, "PASS", Verdict.USE)):
        iq, dm, reduction = _metrics(visits, wide={(101, 2): factor})
        row = _byImage(judge(detectorTable(visits, iq, dm, reduction, setNormals))).loc[(101, 2)]
        assert (row["gateStatus"], row["verdict"]) == (status, verdict), factor


def testWithoutNormalValuesTheGateGivesNoVerdict(visits, setNormals):
    """No table of normal values: nothing is passed by the gate, and the sequence itself still tells."""
    iq, dm, reduction = _metrics(visits, bad={(112, 4)}, wide={(101, 2): 1.06})
    table = detectorTable(visits, iq, dm, reduction)
    assert set(table["gateStatus"]) == {"UNKNOWN"} and set(table["gateReason"]) == {"no normal value"}
    assert (table["gateUnjudged"] > 0).all() and set(table["gateChecks"]) == {""}
    judged = judge(table)
    byImage = _byImage(judged)
    assert byImage.loc[(112, 4), "verdict"] == Verdict.OUT
    assert (
        "FWHM 0.68 px" in byImage.loc[(112, 4), "reason"]
        and "reserved RMS" in byImage.loc[(112, 4), "reason"]
    )
    assert not judged["reason"].str.contains("gate").any()
    assert byImage.loc[(101, 2), "verdict"] == Verdict.USE  # too wide for the gate, which was not asked
    report = buildReport(visits, judged, speciesTable(visits, dm), "test")
    assert "- **The gate gave no verdict on any detector image** (no normal value)" in report
    # Negative control: with the normal values the gate judges every image, fails the wide one, and
    # the report has nothing to say about verdicts not given.
    withNormals = judge(detectorTable(visits, iq, dm, reduction, setNormals))
    assert "UNKNOWN" not in set(withNormals["gateStatus"])
    assert _byImage(withNormals).loc[(101, 2), "verdict"] == Verdict.OUT
    assert "gave no verdict" not in buildReport(visits, withNormals, speciesTable(visits, dm), "test")
    # Negative control: loosen the sequence rules past the fault and, with no gate verdict, nothing is out.
    loose = judge(table, Thresholds(fwhmOut=10, fwhmCheck=10, rmsOut=100, rmsCheck=100))
    assert _byImage(loose).loc[(112, 4), "verdict"] == Verdict.WEAK  # few lines


def testReportCountsWhatTheGateDidNotJudge(visits, setNormals):
    iq, dm, reduction = _metrics(visits)
    species = speciesTable(visits, dm)
    # No normal value of any kind for n3: its eight images get no gate verdict.
    without = setNormals[~((setNormals["camera"] == "n3") | (setNormals["metric"] == "linesPerFiber"))]
    judged = judge(detectorTable(visits, iq, dm, reduction, without))
    assert set(judged.loc[judged["detector"] == "n3", "gateStatus"]) == {"UNKNOWN"}
    assert set(judged.loc[judged["detector"] != "n3", "gateStatus"]) == {"PASS"}
    assert (judged["verdict"] == Verdict.USE).all()  # go, by the sequence rules
    report = buildReport(visits, judged, species, "test", normalValues="normalValues-run99.parquet")
    assert "- The gate gave no verdict on 8 of 32 detector images (no normal value): " in report
    assert "| Normal values | `normalValues-run99.parquet` |" in report
    # No normal value for n3's line scatter alone: its width is judged, its scatter is not.
    noScatter = setNormals[
        ~((setNormals["camera"] == "n3") & setNormals["metric"].str.startswith("robustRms"))
    ]
    judged = judge(detectorTable(visits, iq, dm, reduction, noScatter))
    n3 = judged[judged["detector"] == "n3"].set_index("visit")
    assert set(n3["gateStatus"]) == {"PASS"}
    # Neon: NeI in x and y; HgCd: mercury and cadmium are not judged at all; quartz: the traces in x.
    assert n3["gateUnjudged"].to_dict() == {100: 2, 101: 2, 102: 2, 110: 0, 111: 0, 112: 0, 120: 1, 121: 1}
    report = buildReport(visits, judged, species, "test")
    assert "- On 5 detector image(s) with a gate verdict, the gate left some numbers without one" in report
    # Negative control: with every normal value neither line is written.
    full = buildReport(visits, judge(detectorTable(visits, iq, dm, reduction, setNormals)), species, "test")
    assert "gave no verdict" not in full and "without one" not in full and "Normal values" not in full


def testOffsetFromTheCalibrationInUseIsNotJudged(visits, setNormals):
    """The set replaces the calibration in use: however far the instrument is from it, no image is out for it."""
    iq, dm, reduction = _metrics(visits, drift=3.0)
    table = detectorTable(visits, iq, dm, reduction, setNormals)
    assert set(table["gateStatus"]) == {"PASS"}
    numbers = gateNumbers(iq, dm, setNormals)
    assert set(numbers["check"]) == {
        "width",
        "linesPerFiberArc",
        "linesPerFiberQuartz",
        "scatterX",
        "scatterY",
    }
    # Negative control: the gate given every table fails each image on that offset.
    plain = gate.gate(iq=iq, lines=dm, normalValues=setNormals)
    assert set(plain["qaStatus"]) == {"FAIL"} and set(plain["qaDecidedBy"]) == {"calibOffset"}
    # Likewise the trace centre offset of a frame lit through the engineering fibers.
    iis = iq.assign(illumination="iis")
    assert set(gate.gate(iq=iis)["qaDecidedBy"]) == {"traceCentreOffset"}
    assert "FAIL" not in set(gateVerdicts(iis)["gateStatus"])


def testGateNumbersCarryTheirLimits(visits, setNormals):
    """``warnLimit`` and ``failLimit`` are the check's limits in the number's units, where the verdict changes."""
    wide = {(100, 1): 1.029, (101, 1): 1.031, (102, 1): 1.049, (110, 1): 1.051}
    iq, dm = _metrics(visits, wide=wide)[:2]
    numbers = gateNumbers(iq, dm, setNormals)
    width = numbers[(numbers["check"] == "width") & (numbers["spectrograph"] == 1)].set_index("visit")
    check = CHECKS["width"]
    assert (width["warnLimit"] == width["normal"] * (1 + check.warn / 100)).all()
    assert (width["failLimit"] == width["normal"] * (1 + check.fail / 100)).all()
    assert width.loc[[100, 101, 102, 110], "status"].tolist() == ["PASS", "WARN", "WARN", "FAIL"]
    above = numbers[numbers["check"].isin(["width", "scatterX", "scatterY"])]
    assert ((above["value"] >= above["warnLimit"]) == above["status"].isin(["WARN", "FAIL"])).all()
    # Fewer lines are worse: the limits lie below the normal value, and a count under them fails.
    perFiber = numbers[numbers["check"] == "linesPerFiberArc"]
    arc = CHECKS["linesPerFiberArc"]
    assert np.allclose(perFiber["failLimit"], perFiber["normal"] * (1 - arc.fail / 100))
    assert (perFiber["failLimit"] < perFiber["warnLimit"]).all() and (
        perFiber["warnLimit"] < perFiber["normal"]
    ).all()
    # Negative control: taken above the normal value, as a width's are, they would pass a lamp gone dim.
    assert not np.allclose(perFiber["failLimit"], perFiber["normal"] * (1 + arc.fail / 100))
    dim = gateNumbers(*_metrics(visits, fading={102: 0.4})[:2], setNormals)
    dim = dim[(dim["check"] == "linesPerFiberArc") & (dim["visit"] == 102)]
    assert (dim["value"] < dim["failLimit"]).all() and set(dim["status"]) == {"FAIL"}
    # Without normal values there is no verdict and no limit to draw.
    bare = gateNumbers(iq, dm)
    assert set(bare["status"]) == {""} and bare[["warnLimit", "failLimit"]].isna().all().all()


def testFwhmAloneOnlySaysCheck(visits):
    # A faint lamp's FWHM can be off on one exposure with a sound fit: look at it, don't leave it out.
    iq, dm, reduction = _metrics(visits)
    iq.loc[(iq["visit"] == 110) & (iq["spectrograph"] == 1), "medFwhm"] = 2.0  # 0.69x its sequence median
    judged = _byImage(judge(detectorTable(visits, iq, dm, reduction)))
    assert judged.loc[(110, 1), "verdict"] == Verdict.CHECK
    assert "FWHM 2.00 px" in judged.loc[(110, 1), "reason"] and judged.loc[(110, 1), "rules"] == Rule.FWHM
    # Negative control: the same FWHM with the reserved RMS also off (1.6x) is out.
    reduction.loc[
        (reduction["visit"] == 110)
        & (reduction["spectrograph"] == 1)
        & reduction["metric"].isin(["reservedXRms", "reservedYRms"]),
        "value",
    ] = 0.14
    judged = _byImage(judge(detectorTable(visits, iq, dm, reduction)))
    assert judged.loc[(110, 1), "verdict"] == Verdict.OUT


def testFadingLampIsWeakNotOut(visits):
    iq, dm, reduction = _metrics(visits, fading={101: 0.9, 102: 0.75})
    judged = judge(detectorTable(visits, iq, dm, reduction))
    byVisit = judged.groupby("visit")["verdict"].agg(lambda s: set(s))
    assert byVisit[100] == {Verdict.USE}
    assert byVisit[101] == {Verdict.USE}  # 0.9 is above the weak threshold
    assert byVisit[102] == {Verdict.WEAK}
    assert set(judged.loc[judged["visit"] == 102, "rules"]) == {Rule.LINES}
    species = speciesTable(visits, dm)
    neon = species[(species["species"] == "NeI") & (species["spectrograph"] == 1)].set_index("visit")[
        "linesRatio"
    ]
    assert neon.round(2).to_dict() == {100: 1.0, 101: 0.9, 102: 0.75}
    tr = trends(judged, species)
    assert list(tr[tr["species"] == "NeI"]["linesRatio"].round(2)) == [1.0, 0.9, 0.75]
    cov = coverage(judged, species).set_index(["detector", "species"])
    assert cov.loc[("n1", "NeI"), "usable"] == 3 and cov.loc[("n1", "NeI"), "weak"] == 1


def testSpeciesWithNoUsableExposureIsFlagged(visits, setNormals):
    iq, dm, reduction = _metrics(visits, bad={(100, 2), (101, 2), (102, 2)})
    judged = judge(detectorTable(visits, iq, dm, reduction, setNormals))
    species = speciesTable(visits, dm)
    cov = coverage(judged, species).set_index(["detector", "species"])
    assert cov.loc[("n2", "NeI"), "usable"] == 0
    report = buildReport(visits, judged, species, "test")
    assert "**n2 has no go NeI exposure.**" in report
    # Negative control: a detector wrong through its whole sequence agrees with itself, so the sequence
    # rules pass it; only the gate, with the normal values, sees it.
    blind = judge(detectorTable(visits, iq, dm, reduction))
    assert coverage(blind, species).set_index(["detector", "species"]).loc[("n2", "NeI"), "usable"] == 3


def testMissingVisitIsReported(visits):
    iq, dm, reduction = _metrics(visits[visits["visit"] != 100])
    judged = judge(detectorTable(visits, iq, dm, reduction))
    assert 100 not in set(judged["visit"])
    recommendation = pick(judged, visits).set_index("detector")
    assert recommendation.loc["n1", "missing"] == [100]
    report = buildReport(visits, judged, speciesTable(visits, dm), "test")
    assert "- Not measured (exposed, but no data in the Butler): 100." in report


def testUnassessedImageIsMissingNotGo(visits, setNormals):
    """A stored row with nothing for the gate to judge and no line measured is unassessed, so missing."""
    iq, dm, reduction = _metrics(visits)
    blank = (iq["visit"] == 101) & (iq["spectrograph"] == 2)
    iq.loc[blank, ["medFwhm", "linesPerFiber"]] = float("nan")
    blankReduction = (
        (reduction["visit"] == 101)
        & (reduction["spectrograph"] == 2)
        & reduction["metric"].isin(["reservedXRms", "reservedYRms"])
    )
    reduction.loc[blankReduction, "value"] = float("nan")
    measured = dm[~((dm["visit"] == 101) & (dm["spectrograph"] == 2))]
    judged = judge(detectorTable(visits, iq, measured, reduction, setNormals))
    assert judged[(judged["visit"] == 101) & (judged["spectrograph"] == 2)].empty
    assert len(judged) == len(iq) - 1
    table = goNoGo(judged, visits)
    assert table.loc["n2", 101] == "missing"
    # Negative control: the same row with its lines measured is assessed and go.
    assert goNoGo(judge(detectorTable(visits, iq, dm, reduction, setNormals)), visits).loc["n2", 101] == "go"


def testFallbackFwhmTakesNoPartInTheSequence(visits):
    """A traceOnly image's FWHM comes from the fiberProfiles, not the image: no consistency verdict."""
    iq, dm, reduction = _metrics(visits)
    iq["traceOnly"] = False
    odd = (iq["visit"] == 101) & (iq["spectrograph"] == 2)
    iq.loc[odd, ["medFwhm", "traceOnly"]] = [5.0, True]
    judged = _byImage(judge(detectorTable(visits, iq, dm, reduction)))
    assert pd.isna(judged.loc[(101, 2), "fwhmRatio"]) and judged.loc[(101, 2), "verdict"] == Verdict.USE
    # Negative control: measured, the same width is far from its sequence and marked.
    iq.loc[odd, "traceOnly"] = False
    judged = _byImage(judge(detectorTable(visits, iq, dm, reduction)))
    assert judged.loc[(101, 2), "verdict"] == Verdict.CHECK


def testListedDetectorWithNoMetricsIsMissing(visits):
    """A camera the listing names stays in the grid even when nothing was measured on it."""
    iq, dm, reduction = _metrics(visits)
    iq = iq[iq["spectrograph"] != 3]
    dm = dm[dm["spectrograph"] != 3]
    table = goNoGo(judge(detectorTable(visits, iq, dm, reduction)), visits)
    assert list(table.index) == ["n1", "n2", "n3", "n4"]
    assert (table.loc["n3"] == "missing").all()


def testQthIsATraceLamp():
    assert lampOf("sps expose qth=10.0 cams=b1,r1") == "qth"
    assert LAMP_SPECIES["qth"] == ("Trace",)


def testGoNoGo(visits, setNormals):
    iq, dm, reduction = _metrics(visits, bad={(112, 4)})
    iq = iq[~((iq["visit"] == 100) & (iq["spectrograph"] == 2))]  # n2 not measured on 100
    table = goNoGo(judge(detectorTable(visits, iq, dm, reduction, setNormals)), visits)
    assert list(table.index) == ["n1", "n2", "n3", "n4"]
    assert list(table.columns) == [100, 101, 102, 110, 111, 112, 120, 121]
    assert table.loc["n4", 112] == "no-go" and table.loc["n4", 111] == "go"
    assert table.loc["n2", 100] == "missing" and (table.loc["n3"] == "go").all()
    # A visit that never exposed the detector is blank, not missing.
    narrow = visits.copy()
    narrow.loc[narrow["visit"] == 110, "cameras"] = "n1,n2,n3"
    iqNarrow = iq[~((iq["visit"] == 110) & (iq["spectrograph"] == 4))]
    table = goNoGo(judge(detectorTable(narrow, iqNarrow, dm, reduction, setNormals)), narrow)
    assert table.loc["n4", 110] == "" and table.loc["n2", 100] == "missing"
    # Negative control: a weak or check image is still go; only out is no-go.
    iq2, dm2, reduction2 = _metrics(visits, fading={102: 0.7})
    judged = judge(detectorTable(visits, iq2, dm2, reduction2, setNormals))
    assert set(judged.loc[judged["visit"] == 102, "verdict"]) == {
        Verdict.CHECK
    }  # a gate WARN: 30 % fewer lines
    assert (goNoGo(judged, visits).loc[:, 102] == "go").all()


def testReportIsAGoNoGo(visits, setNormals):
    iq, dm, reduction = _metrics(visits, bad={(112, 4)})
    judged = judge(detectorTable(visits, iq, dm, reduction, setNormals))
    report = buildReport(
        visits,
        judged,
        speciesTable(visits, dm),
        "Run99",
        source="u/me/comparison/run99/v1",
        figures={"Go / no-go per detector and visit": "calibInputs-run99-gonogo.png"},
    )
    assert report.startswith("# Run99: go/no-go")
    assert "- **No-go:**\n  - n4: 112\n" in report
    assert "![Go / no-go per detector and visit](calibInputs-run99-gonogo.png)" in report
    assert "| n4 | 100–102, 110–111, 120–121 | 112 |" in report
    assert "| n3 | 100–102, 110–112, 120–121 | - |" in report
    assert "## Why" not in report  # the no-go detectors' section at the end carries the numbers
    assert "## Reasons in full" not in report and "u/me/comparison/run99/v1" in report
    assert len(report.splitlines()) < 35  # a go/no-go, not a dump
    detailed = buildReport(visits, judged, speciesTable(visits, dm), "Run99", detail=True)
    assert "## Reasons in full" in detailed and "- **n4 112** (hgcd 3/3, out): gate FAIL" in detailed
    keyPoints = summary(visits, judged, speciesTable(visits, dm))
    assert keyPoints[0] == "**Leave out n4 on 112**: FWHM off, gate FAIL, reserved RMS off."
    withFigure = buildReport(
        visits, judged, speciesTable(visits, dm), "Run99", detectorFigures={"n4": "run99-n4.png"}
    )
    assert (
        "## No-go detectors in detail\n\n### n4\n" in withFigure
        and "![n4 through the set](run99-n4.png)" in withFigure
    )
    assert (
        "| 112 | hgcd 3/3 | out | 0.68 | 2.200 | gate FAIL: lines per lit fiber 3.6 is -70.0 % from normal 12.0, "
        "limit -50 %; " in withFigure
    )
    withLines = buildReport(
        visits,
        judged,
        speciesTable(visits, dm),
        "Run99",
        detectorFigures={"n4": "run99-n4.png"},
        imageFigures={("n4", 112): "run99-n4-112-lines.png"},
    )
    assert (
        "![n4 through the set](run99-n4.png)\n\n![n4 112 line residuals](run99-n4-112-lines.png)\n"
        in withLines
    )
    assert "FWHM 0.68 px (0.23x its sequence); reserved RMS 2.20 px (" in withFigure


def testNoGoMissingVisitIsReported(visits):
    iq, dm, reduction = _metrics(visits[visits["visit"] != 100])
    judged = judge(detectorTable(visits, iq, dm, reduction))
    report = buildReport(visits, judged, speciesTable(visits, dm), "test")
    assert "- Not measured (exposed, but no data in the Butler): 100." in report


X, Y = "robustRmsDeltaX", "robustRmsDeltaY"


def _normalValues(rows):
    """Return a table of normal values from (metric, camera, selector, value, count, validFrom) rows of arcs."""
    table = pd.DataFrame(rows, columns=["metric", "camera", "selector", "value", "count", "validFrom"])
    return table.assign(frameType="arc").reindex(columns=list(NORMAL_COLUMNS))


@pytest.fixture
def normalValues():
    return _normalValues(
        [
            (X, "b2", "XeI", 0.10, 9, 1000),
            (Y, "b2", "XeI", 0.20, 9, 1000),
            (X, "r2", "XeI", 0.05, 9, 1000),
            (X, "b2", "XeI", 0.30, 9, 2000),  # the next run's
            (X, "b2", "NeI", 0.10, 2, 1000),  # from too few frames to judge by
            (X, "b2", "HgI", 0.10, 9, 1000),  # mercury's scatter is stored, not judged
        ]
    )


def testScatterLimitsAreTheGatesPercentagesOfTheNormalValue(normalValues):
    limits = scatterLimits("b", 2, ["XeI", "HgI", "NeI", "Trace"], 1500, normalValues)
    assert set(limits) == {"XeI", "HgI", "NeI"}  # traces are not lamp lines
    x, y = CHECKS["scatterX"], CHECKS["scatterY"]
    assert limits["XeI"][X] == pytest.approx((0.10 * (1 + x.warn / 100), 0.10 * (1 + x.fail / 100)))
    assert limits["XeI"][Y] == pytest.approx((0.20 * (1 + y.warn / 100), 0.20 * (1 + y.fail / 100)))
    warn, fail = limits["XeI"][X]
    assert 0.10 < warn < fail
    # Negative control: y has its own percentages, not x's.
    assert limits["XeI"][Y] != pytest.approx((0.20 * (1 + x.warn / 100), 0.20 * (1 + x.fail / 100)))
    assert limits["HgI"] == {X: (None, None), Y: (None, None)}
    assert limits["NeI"][X] == (None, None)
    # Negative control: another camera has another normal value, so other limits.
    other = scatterLimits("r", 2, ["XeI"], 1500, normalValues)["XeI"][X]
    assert other == pytest.approx((0.05 * (1 + x.warn / 100), 0.05 * (1 + x.fail / 100)))
    assert other != pytest.approx((warn, fail))


def testScatterLimitsAreWhereTheGateChangesItsVerdict(normalValues):
    """A scatter just under a limit and one just over it get the gate's two verdicts."""
    limits = scatterLimits("b", 2, ["XeI"], 1500, normalValues)["XeI"]
    rows = []
    for metric, check in ((X, "scatterX"), (Y, "scatterY")):
        warn, fail = limits[metric]
        for value in (0.999 * warn, 1.001 * warn, 0.999 * fail, 1.001 * fail):
            rows.append((1500, "b", 2, "arc", check, metric, "XeI", "b2", "XeI", value, False))
    judged = gate.judge(pd.DataFrame(rows, columns=list(gate.MEASUREMENT_COLUMNS)), normalValues)
    assert list(judged["status"]) == ["PASS", "WARN", "WARN", "FAIL"] * 2


def testScatterLimitsFollowTheNormalValueInForce(normalValues):
    x = CHECKS["scatterX"]
    later = scatterLimits("b", 2, ["XeI"], 2500, normalValues)["XeI"][X]
    assert later == pytest.approx((0.30 * (1 + x.warn / 100), 0.30 * (1 + x.fail / 100)))
    # Negative control: a value is not in force before the visit it is valid from.
    assert scatterLimits("b", 2, ["XeI"], 1999, normalValues)["XeI"][X] != pytest.approx(later)
    assert scatterLimits("b", 2, ["XeI"], 999, normalValues)["XeI"][X] == (None, None)


def testNoScatterLimitsWithoutNormalValues(normalValues):
    assert scatterLimits("b", 2, ["XeI"], 1500) == {}
    assert scatterLimits("b", 2, ["XeI"], 1500, normalValues.iloc[:0]) == {}
    assert scatterLimits("b", 2, ["Trace"], 1500, normalValues) == {}
    # A frame lit through the engineering fibers is not judged on its line scatter.
    assert scatterLimits("b", 2, ["XeI"], 1500, normalValues, frameType=IIS_ARC)["XeI"] == {
        X: (None, None),
        Y: (None, None),
    }
    # Negative control: the same call for an arc has limits.
    assert scatterLimits("b", 2, ["XeI"], 1500, normalValues)["XeI"][X] != (None, None)
