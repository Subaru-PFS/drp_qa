"""Tests for `pfs.drp.qa.comparison.report`."""

import pandas as pd

from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.findings import findings, judgeImages
from pfs.drp.qa.comparison.plan import coverage, failedQuanta, summarize
from pfs.drp.qa.comparison.report import ReportInputs, buildReport, failedSummary, recurringSequences

THRESHOLDS = pd.DataFrame(
    {
        "metric": ["medFwhm"],
        "warn": [2.6],
        "fail": [2.8],
        "higherIsWorse": [True],
        "absolute": [False],
        "provenance": ["test"],
    }
)


def _period(periods, visit, listing):
    rows = []
    for night in range(4):
        day = f"2026-01-0{2 + night}"
        rows.append(
            visit(10 * night + 1, f"{day} 17:00", "scienceArc", sequence=100 + night, name="Arc: Neon")
        )
        rows.append(
            visit(
                10 * night + 2, f"{day} 21:00", "scienceObject", sequence=200 + night, name="field", design=7
            )
        )
    visits = classifyVisits(listing(*rows), periods.values(), pd.Series({7: "science"}))
    metrics = pd.DataFrame(
        [
            (v, arm, sm, 3.0 if (v == 21 and sm == 1) else 2.3)
            for v in visits["pfs_visit_id"]
            for sm in (1, 2)
            for arm in "br"
        ],
        columns=["visit", "arm", "spectrograph", "medFwhm"],
    )
    return visits, metrics


def testRecurringSequencesAreCalibrationsOnly(periods, visit, listing):
    visits, _ = _period(periods, visit, listing)
    recurring = recurringSequences(visits)
    assert recurring[["sequence_type", "sequence_name", "nights"]].values.tolist() == [
        ["scienceArc", "Arc: Neon", 4]
    ]


def testBuildReport(periods, visit, listing):
    visits, metrics = _period(periods, visit, listing)
    holdings = metrics[["visit", "arm", "spectrograph"]].assign(raw=True, reduced=True, judged=True)
    judged = judgeImages(metrics, visits, THRESHOLDS)
    found = findings(metrics, judged, visits)
    page = buildReport(
        ReportInputs(
            period="run1",
            version="w.2026.41",
            collection="u/me/comparison/run1/w.2026.41",
            readUntil="now",
            visits=visits,
            summary=summarize(visits, coverage(visits, holdings)),
            metrics=metrics,
            judged=judged,
            findings=found,
            reference=None,
        )
    )
    assert page.startswith("<!doctype html>")
    assert "32 gated images" in page
    assert "Incomplete" not in page
    assert "32 gated images · 2 problem images" in page
    assert "SM1" in page  # visit 21: b1 and r1 together, one problem
    assert "Arc: Neon" in page  # a calibration's name
    assert "field" not in page  # a science sequence's name never is
    assert "run 'run' first" not in page
    assert page.count("<svg") >= 3
    assert "<h2>n-arm darks</h2>" not in page  # no section without lastLit


def testUnvalidatedAreLabelledApart(periods, visit, listing):
    visits = classifyVisits(
        listing(
            visit(1, "2026-01-03 17:00", "scienceArc", sequence=100, name="Arc: Neon"),
            visit(2, "2026-01-03 17:10", "slitThroughFocus", sequence=101, expType="arc"),
        ),
        periods.values(),
    )
    metrics = pd.DataFrame(
        [(v, "b", sm, 2.3 if v == 1 else 3.5) for v in (1, 2) for sm in (1, 2)],
        columns=["visit", "arm", "spectrograph", "medFwhm"],
    )
    holdings = metrics[["visit", "arm", "spectrograph"]].assign(raw=True, reduced=True, judged=True)
    judged = judgeImages(metrics, visits, THRESHOLDS)
    found = findings(metrics, judged, visits)
    assert found["validated"].tolist() == [False, False]  # only the through-focus arc fails
    page = buildReport(
        ReportInputs(
            period="run1",
            version="v1",
            collection="u/me/comparison/run1/v1",
            readUntil="now",
            visits=visits,
            summary=summarize(visits, coverage(visits, holdings)),
            metrics=metrics,
            judged=judged,
            findings=found,
        )
    )
    assert "2 gated images · 0 problem images" in page  # the gate's line is clean...
    assert "2 <a href='#unvalidated'>unvalidated</a>" in page  # ...the unvalidated are counted apart
    assert "Unvalidated problems: 2 images" in page


def testDarksSection(periods, visit, listing):
    from pfs.drp.qa.comparison.persistence import lastLitBefore

    visits, metrics = _period(periods, visit, listing)
    darks = classifyVisits(
        listing(
            visit(1, "2026-01-03 18:00", "scienceTrace", cameras="n1", exptime=20, name="Trace"),
            visit(2, "2026-01-03 18:02", "darks", cameras="n1", exptime=300),
        ),
        periods.values(),
    )
    holdings = metrics[["visit", "arm", "spectrograph"]].assign(raw=True, reduced=True, judged=True)
    judged = judgeImages(metrics, visits, THRESHOLDS)
    page = buildReport(
        ReportInputs(
            period="run1",
            version="v1",
            collection="c",
            readUntil="now",
            visits=visits,
            summary=summarize(visits, coverage(visits, holdings)),
            metrics=metrics,
            judged=judged,
            findings=findings(metrics, judged, visits),
            lastLit=lastLitBefore(darks),
        )
    )
    assert "<h2>n-arm darks</h2>" in page
    assert "Darks by gap after the last lit exposure" in page


def testProblemsGroupFindings():
    from pfs.drp.qa.comparison.report import problems

    found = pd.DataFrame(
        {
            "visit": [1, 1, 2, 3],
            "arm": ["n", "n", "n", "b"],
            "night": ["2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"],
            "status": ["WARN", "FAIL", "WARN", "WARN"],
            "metrics": [
                "pctFlagged=30% >= warn",
                "pctFlagged=40% >= fail",
                "pctFlagged=31% >= warn",
                "nLines=1 <= fail",
            ],
            "extent": ["n arm", "n arm", "n arm", "detector"],
            "expected": ["", "", "", ""],
            "setup": ["scienceArc 'Arc: Neon', neon; 5 s"] * 3 + ["scienceArc 'Arc: Argon', argon; 5 s"],
        }
    )
    result = problems(found)
    assert result[["status", "metrics", "arm", "images", "visits"]].values.tolist() == [
        ["FAIL", "pctFlagged", "n", 3, 2],
        ["WARN", "nLines", "b", 1, 1],
    ]
    assert (
        result.loc[0, "setup"] == "scienceArc 'Arc: Neon', neon"
        and result.loc[0, "nights"] == "2026-01-02 to 2026-01-03"
    )


def testIncompleteReportSaysSo(periods, visit, listing):
    visits, metrics = _period(periods, visit, listing)
    holdings = metrics[["visit", "arm", "spectrograph"]].assign(raw=True, reduced=True, judged=True)
    holdings.loc[holdings.index[:3], "judged"] = False  # three images still to judge
    judged = judgeImages(metrics, visits, THRESHOLDS)
    page = buildReport(
        ReportInputs(
            period="run1",
            version="w.2026.41",
            collection="u/me/comparison/run1/w.2026.41",
            readUntil="now",
            visits=visits,
            summary=summarize(visits, coverage(visits, holdings)),
            metrics=metrics,
            judged=judged,
            findings=findings(metrics, judged, visits),
            reference=None,
        )
    )
    assert "<b>Incomplete:</b> 3 detector images are not judged yet" in page


def testFailedSummary():
    detectors = pd.DataFrame(
        {
            "visit": [1, 1, 1, 2],
            "arm": ["n", "n", "b", "n"],
            "spectrograph": [1, 2, 1, 1],
            "sequence_type": "scienceArc",
            "cadence": "set",
            "status": ["failed", "failed", "judged", "judged"],
        }
    )
    failed = failedQuanta(
        "\n".join(
            f"Execution of task 'reduceExposure' on quantum {{instrument: 'PFS', arm: 'n', spectrograph: {s},"
            f" visit: {v}}} failed. Exception ValueError: need at least one array to concatenate"
            for v, s in [(1, 1), (1, 2), (2, 1)]
        )
    )
    summary = failedSummary(detectors, failed)
    assert summary[["visit", "cameras", "task"]].values.tolist() == [[1, "n1,n2", "reduceExposure"]]
    assert summary.loc[0, "error"].startswith("ValueError: need at least one array")
