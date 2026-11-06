"""Tests for `pfs.drp.qa.comparison.calibInputs` on a synthetic calibration set."""

import pandas as pd
import pytest

from pfs.drp.qa.comparison.calibInputs import (
    LAMP_SPECIES,
    Thresholds,
    Verdict,
    buildReport,
    coverage,
    detectorTable,
    goNoGo,
    judge,
    lampOf,
    lineThresholds,
    pick,
    setVisits,
    speciesTable,
    trends,
)

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


def _metrics(visits, bad=None, fading=None, failGate=None):
    """iqQaMetrics, dmLineResidualMetrics and dmLineResidualSummary for every detector image of ``visits``.

    ``bad``: {(visit, spectrograph)} whose detector image is wrong (tiny FWHM, huge reserved RMS,
    few lines); ``fading``: {visit: factor} scaling the lamp's line count; ``failGate``: {(visit,
    spectrograph)} the gate FAILed for a reason of its own.
    """
    bad, fading, failGate = bad or set(), fading or {}, failGate or set()
    iq, dm, summary = [], [], []
    for row in visits.itertuples():
        for sp in ARMS["n"]:
            isBad = (row.visit, sp) in bad
            fwhm = 0.68 if isBad else 2.90 + 0.01 * sp
            rms = 2.2 if isBad else 0.08 + 0.005 * sp
            scale = fading.get(row.visit, 1.0) * (0.3 if isBad else 1.0)
            status = "FAIL" if (row.visit, sp) in failGate else "PASS"
            reason = "linesPerFiber=5.3 <= fail threshold 11.2" if status == "FAIL" else ""
            iq.append((row.visit, "n", sp, fwhm, rms, rms, status, reason))
            for species in row.species:
                lines = 1000 * scale if species != "Trace" else 2e6
                for statusType, factor in (("BOTH", 1.0), ("RESERVED", 0.1)):
                    for metric, value in (
                        ("nLines", lines * factor),
                        ("linesPerFiber", lines * factor / 600),
                        ("robustRmsDeltaX", rms),
                        ("robustRmsDeltaY", rms),
                    ):
                        judged = statusType == "RESERVED" and metric.startswith("robustRms")
                        status = ("FAIL" if isBad else "PASS") if judged else ""
                        dm.append((row.visit, "n", sp, "visit", species, statusType, metric, value, status))
                        dm.append((row.visit, "n", sp, "calib", species, statusType, metric, value, ""))
            summary.append(
                (
                    row.visit,
                    "n",
                    sp,
                    "FAIL" if isBad else "PASS",
                    "visit NeI: robustRmsDeltaX=2.0px >= fail threshold 0.12px" if isBad else "",
                )
            )
    iq = pd.DataFrame(
        iq,
        columns=[
            "visit",
            "arm",
            "spectrograph",
            "medFwhm",
            "fitReservedXRms",
            "fitReservedYRms",
            "qaStatus",
            "qaReason",
        ],
    )
    dm = pd.DataFrame(
        dm,
        columns=[
            "visit",
            "arm",
            "spectrograph",
            "reference",
            "description",
            "statusType",
            "metric",
            "value",
            "status",
        ],
    )
    summary = pd.DataFrame(summary, columns=["visit", "arm", "spectrograph", "qaStatus", "qaReason"])
    return iq, dm, summary


@pytest.fixture
def visits():
    return setVisits(_listing())


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


def testDetectorGoneBadIsOutOnItsOwnExposuresOnly(visits):
    # n4 breaks from the last HgCd exposure onwards: 112 and both traces.
    bad = {(112, 4), (120, 4), (121, 4)}
    iq, dm, summary = _metrics(visits, bad=bad, failGate={(112, 4)})
    judged = judge(detectorTable(visits, iq, dm, summary))
    outs = judged[judged["verdict"] == Verdict.OUT]
    assert set(zip(outs["visit"], outs["spectrograph"], strict=True)) == bad
    assert (judged.loc[~judged.index.isin(outs.index), "verdict"] == Verdict.USE).all()
    reason = judged.set_index(["visit", "spectrograph"]).loc[(112, 4), "reason"]
    assert (
        "gate FAIL" in reason
        and "line residuals FAIL" in reason
        and "FWHM" in reason
        and "reserved RMS" in reason
    )
    # Negative control: the arm's median over detectors hides the fault, so a per-arm judgement would
    # have passed these exposures; the report must be per detector.
    armFwhm = judged.groupby("visit")["medFwhm"].median()
    assert (armFwhm / armFwhm.median() > 0.9).all()
    recommendation = pick(judged, visits).set_index("detector")
    assert recommendation.loc["n4", "out"] == [112, 120, 121]
    assert recommendation.loc["n3", "out"] == []
    assert recommendation.loc["n4", "measured"] == [100, 101, 102, 110, 111, 112, 120, 121]


def testConsistencyWithoutStoredVerdicts(visits):
    # The gate and the line residuals passed everything (e.g. an older thresholds file): the sequence itself tells.
    iq, dm, summary = _metrics(visits, bad={(112, 4)})
    dm["status"] = ""  # nothing judged by dmLineResiduals
    judged = judge(detectorTable(visits, iq, dm, summary))
    row = judged.set_index(["visit", "spectrograph"]).loc[(112, 4)]
    assert row["verdict"] == Verdict.OUT
    assert "FWHM 0.68 px" in row["reason"] and "reserved RMS" in row["reason"]
    # Negative control: loosen the consistency thresholds past the fault and the stored verdicts alone keep it in.
    loose = judge(
        detectorTable(visits, iq, dm, summary), Thresholds(fwhmOut=10, fwhmCheck=10, rmsOut=100, rmsCheck=100)
    )
    assert (
        loose.set_index(["visit", "spectrograph"]).loc[(112, 4), "verdict"] != Verdict.OUT
    )  # weak: few lines


def testFwhmAloneOnlySaysCheck(visits):
    # A faint lamp's FWHM can be off on one exposure with a sound fit: look at it, don't leave it out.
    iq, dm, summary = _metrics(visits)
    iq.loc[(iq["visit"] == 110) & (iq["spectrograph"] == 1), "medFwhm"] = 2.0  # 0.69x its sequence median
    judged = judge(detectorTable(visits, iq, dm, summary)).set_index(["visit", "spectrograph"])
    assert judged.loc[(110, 1), "verdict"] == Verdict.CHECK
    assert "FWHM 2.00 px" in judged.loc[(110, 1), "reason"]
    # Negative control: the same FWHM with the reserved RMS also off (1.6x) is out.
    iq.loc[(iq["visit"] == 110) & (iq["spectrograph"] == 1), ["fitReservedXRms", "fitReservedYRms"]] = 0.14
    judged = judge(detectorTable(visits, iq, dm, summary)).set_index(["visit", "spectrograph"])
    assert judged.loc[(110, 1), "verdict"] == Verdict.OUT


def testFadingLampIsWeakNotOut(visits):
    iq, dm, summary = _metrics(visits, fading={101: 0.9, 102: 0.75})
    judged = judge(detectorTable(visits, iq, dm, summary))
    byVisit = judged.groupby("visit")["verdict"].agg(lambda s: set(s))
    assert byVisit[100] == {Verdict.USE}
    assert byVisit[101] == {Verdict.USE}  # 0.9 is above the weak threshold
    assert byVisit[102] == {Verdict.WEAK}
    species = speciesTable(visits, dm)
    neon = species[(species["species"] == "NeI") & (species["spectrograph"] == 1)].set_index("visit")[
        "linesRatio"
    ]
    assert neon.round(2).to_dict() == {100: 1.0, 101: 0.9, 102: 0.75}
    tr = trends(judged, species)
    assert list(tr[tr["species"] == "NeI"]["linesRatio"].round(2)) == [1.0, 0.9, 0.75]
    cov = coverage(judged, species).set_index(["detector", "species"])
    assert cov.loc[("n1", "NeI"), "usable"] == 3 and cov.loc[("n1", "NeI"), "weak"] == 1


def testSpeciesWithNoUsableExposureIsFlagged(visits):
    iq, dm, summary = _metrics(visits, bad={(100, 2), (101, 2), (102, 2)})
    judged = judge(detectorTable(visits, iq, dm, summary))
    species = speciesTable(visits, dm)
    cov = coverage(judged, species).set_index(["detector", "species"])
    assert cov.loc[("n2", "NeI"), "usable"] == 0
    report = buildReport(visits, judged, species, "test")
    assert "**n2 has no go NeI exposure.**" in report


def testMissingVisitIsReported(visits):
    iq, dm, summary = _metrics(visits[visits["visit"] != 100])
    judged = judge(detectorTable(visits, iq, dm, summary))
    assert 100 not in set(judged["visit"])
    recommendation = pick(judged, visits).set_index("detector")
    assert recommendation.loc["n1", "missing"] == [100]
    report = buildReport(visits, judged, speciesTable(visits, dm), "test")
    assert "- Not measured (exposed, but no data in the Butler): 100." in report


def testUnassessedImageIsMissingNotGo(visits):
    """A stored row whose gate judged nothing and with no line measured is unassessed, so missing."""
    iq, dm, summary = _metrics(visits)
    blank = (iq["visit"] == 101) & (iq["spectrograph"] == 2)
    iq.loc[blank, ["medFwhm", "fitReservedXRms", "fitReservedYRms"]] = float("nan")
    iq.loc[blank, "qaStatus"] = "UNKNOWN"
    dm = dm[~((dm["visit"] == 101) & (dm["spectrograph"] == 2))]
    judged = judge(detectorTable(visits, iq, dm, summary))
    assert judged[(judged["visit"] == 101) & (judged["spectrograph"] == 2)].empty
    table = goNoGo(judged, visits)
    assert table.loc["n2", 101] == "missing"
    # Negative control: the same row with its lines measured is assessed and go.
    judgedFull = judge(detectorTable(visits, iq, *_metrics(visits)[1:]))
    assert goNoGo(judgedFull, visits).loc["n2", 101] == "go"


def testFallbackFwhmTakesNoPartInTheSequence(visits):
    """A traceOnly image's FWHM comes from the fiberProfiles, not the image: no consistency verdict."""
    iq, dm, summary = _metrics(visits)
    iq["traceOnly"] = False
    odd = (iq["visit"] == 101) & (iq["spectrograph"] == 2)
    iq.loc[odd, ["medFwhm", "traceOnly"]] = [5.0, True]
    judged = judge(detectorTable(visits, iq, dm, summary)).set_index(["visit", "spectrograph"])
    assert pd.isna(judged.loc[(101, 2), "fwhmRatio"]) and judged.loc[(101, 2), "verdict"] == Verdict.USE
    # Negative control: measured, the same width is far from its sequence and marked.
    iq.loc[odd, "traceOnly"] = False
    judged = judge(detectorTable(visits, iq, dm, summary)).set_index(["visit", "spectrograph"])
    assert judged.loc[(101, 2), "verdict"] == Verdict.CHECK


def testListedDetectorWithNoMetricsIsMissing(visits):
    """A camera the listing names stays in the grid even when nothing was measured on it."""
    iq, dm, summary = _metrics(visits)
    iq = iq[iq["spectrograph"] != 3]
    dm = dm[dm["spectrograph"] != 3]
    table = goNoGo(judge(detectorTable(visits, iq, dm, summary)), visits)
    assert list(table.index) == ["n1", "n2", "n3", "n4"]
    assert (table.loc["n3"] == "missing").all()


def testQthIsATraceLamp():
    assert lampOf("sps expose qth=10.0 cams=b1,r1") == "qth"
    assert LAMP_SPECIES["qth"] == ("Trace",)


def testGoNoGo(visits):
    iq, dm, summary = _metrics(visits, bad={(112, 4)}, failGate={(112, 4)})
    iq = iq[~((iq["visit"] == 100) & (iq["spectrograph"] == 2))]  # n2 not measured on 100
    table = goNoGo(judge(detectorTable(visits, iq, dm, summary)), visits)
    assert list(table.index) == ["n1", "n2", "n3", "n4"]
    assert list(table.columns) == [100, 101, 102, 110, 111, 112, 120, 121]
    assert table.loc["n4", 112] == "no-go" and table.loc["n4", 111] == "go"
    assert table.loc["n2", 100] == "missing" and (table.loc["n3"] == "go").all()
    # A visit that never exposed the detector is blank, not missing.
    narrow = visits.copy()
    narrow.loc[narrow["visit"] == 110, "cameras"] = "n1,n2,n3"
    iqNarrow = iq[~((iq["visit"] == 110) & (iq["spectrograph"] == 4))]
    table = goNoGo(judge(detectorTable(narrow, iqNarrow, dm, summary)), narrow)
    assert table.loc["n4", 110] == "" and table.loc["n2", 100] == "missing"
    # Negative control: a weak or check image is still go; only out is no-go.
    iq2, dm2, summary2 = _metrics(visits, fading={102: 0.5})
    weak = goNoGo(judge(detectorTable(visits, iq2, dm2, summary2)), visits)
    assert (weak.loc[:, 102] == "go").all()


def testReportIsAGoNoGo(visits):
    iq, dm, summary = _metrics(visits, bad={(112, 4)}, failGate={(112, 4)})
    judged = judge(detectorTable(visits, iq, dm, summary))
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
    withFigure = buildReport(
        visits, judged, speciesTable(visits, dm), "Run99", detectorFigures={"n4": "run99-n4.png"}
    )
    assert (
        "## No-go detectors in detail\n\n### n4\n" in withFigure
        and "![n4 through the set](run99-n4.png)" in withFigure
    )
    assert "| 112 | hgcd 3/3 | out | 0.68 | 2.200 | gate FAIL: lines/fiber 5.3 (fail <= 11.2); " in withFigure
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
    iq, dm, summary = _metrics(visits[visits["visit"] != 100])
    judged = judge(detectorTable(visits, iq, dm, summary))
    report = buildReport(visits, judged, speciesTable(visits, dm), "test")
    assert "- Not measured (exposed, but no data in the Butler): 100." in report


def testLineThresholdsFromThePackagedFile():
    limits = lineThresholds("b", 2, ["XeI", "HgI", "Trace"], "brn", "arc")
    assert set(limits) == {"XeI", "HgI"}  # traces are not lamp lines
    warn, fail = limits["XeI"]["robustRmsDeltaX"]
    assert warn is not None and fail is not None and 0 < warn < fail < 0.5
    assert limits["HgI"]["robustRmsDeltaX"] == (None, None)  # mercury's scatter is stored, not judged
    # Negative control: another population gets other limits.
    other = lineThresholds("r", 2, ["XeI"], "brn", "arc")["XeI"]["robustRmsDeltaX"]
    assert other != (warn, fail)
