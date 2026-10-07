"""Tests for `pfs.drp.qa.comparison.report`."""

import pandas as pd

from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.findings import findings, judgeImages
from pfs.drp.qa.comparison.plan import coverage, summarize
from pfs.drp.qa.comparison.report import ReportInputs, buildReport, recurringSequences

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
    assert "Also 0 unvalidated images" in page
    assert "2 findings, 2 spreading beyond one detector" in page  # SM1 of visit 21: b1 and r1
    assert "Arc: Neon" in page  # a calibration's name
    assert "field" not in page  # a science sequence's name never is
    assert "run 'run' first" not in page
    assert page.count("<svg") >= 3
    assert "n-arm darks" not in page  # no section without lastLit


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
    assert "2 gated images" in page and "0 findings" in page  # the gate's headline is clean...
    assert "Also 2 unvalidated images (2 FAIL)" in page  # ...the unvalidated FAILs are counted apart
    assert "Unvalidated findings" in page


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
    assert "n-arm darks: the last lit exposure before each" in page
    assert "Darks within 30 minutes of a lit exposure" in page and "scienceTrace 'Trace'" in page
