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
    assert "32 images judged" in page
    assert "2 findings, 2 spreading beyond one detector" in page  # SM1 of visit 21: b1 and r1
    assert "Arc: Neon" in page  # a calibration's name
    assert "field" not in page  # a science sequence's name never is
    assert "run 'run' first" not in page
    assert page.count("<svg") >= 3
