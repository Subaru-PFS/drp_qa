"""Tests for `pfs.drp.qa.comparison.residuals`."""

import pandas as pd

from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.residuals import (
    describeCrossing,
    residualFindings,
    residualProblems,
    scatterTable,
)

KEYS = ["visit", "arm", "spectrograph"]


def _visits(periods, visit, listing):
    rows = [
        visit(1, "2026-01-02 17:00", "scienceArc", sequence=100, name="Arc: Neon"),
        visit(2, "2026-01-02 18:00", "scienceArc", sequence=100, name="Arc: Neon"),
        visit(3, "2026-01-02 21:00", "scienceObject", sequence=200, name="field", design=7),
    ]
    return classifyVisits(listing(*rows), periods.values(), pd.Series({7: "science"}))


def _summary():
    rows = [
        (1, "b", 1, "PASS", "NOMINAL"),
        (1, "b", 2, "WARN", "APPLY_SHIFT"),
        (2, "b", 1, "FAIL", "RECALIBRATE"),
        (2, "b", 2, "UNKNOWN", "INSUFFICIENT_DATA"),
        (3, "b", 1, "PASS", "NOMINAL"),
    ]
    return pd.DataFrame(rows, columns=[*KEYS, "qaStatus", "recommendedAction"]).assign(cameraSet="br")


def _metrics(rows):
    columns = [*KEYS, "reference", "description", "statusType", "metric", "value", "status"]
    return pd.DataFrame(rows, columns=columns).assign(cameraSet="br")


CROSSINGS = _metrics(
    [
        (1, "b", 2, "calib", "Trace", "BOTH", "medianDeltaX", 1.2, "WARN"),
        (1, "b", 2, "calib", "Trace", "BOTH", "robustRmsDeltaX", 0.1, ""),
        (2, "b", 1, "calib", "NeI", "BOTH", "medianDeltaY", -2.5, "FAIL"),
        (2, "b", 1, "visit", "NeI", "RESERVED", "robustRmsDeltaY", 0.3, "WARN"),
        (3, "b", 1, "visit", "NeI", "RESERVED", "robustRmsDeltaY", 0.05, "PASS"),
    ]
)


def testDescribeCrossing():
    rows = list(CROSSINGS.itertuples())
    assert describeCrossing(rows[0]) == "calib Trace medianDeltaX=+1.200 px WARN"  # offsets are signed
    assert describeCrossing(rows[3]) == "visit NeI robustRmsDeltaY=0.300 px WARN"  # scatter is not


def testResidualFindings(periods, visit, listing):
    found = residualFindings(CROSSINGS, _summary(), _visits(periods, visit, listing))
    assert list(found[[*KEYS, "status"]].itertuples(index=False, name=None)) == [
        (1, "b", 2, "WARN"),
        (2, "b", 1, "FAIL"),
    ]
    assert found.iloc[0]["metrics"] == "calib Trace medianDeltaX=+1.200 px WARN"
    assert (
        found.iloc[1]["metrics"]
        == "calib NeI medianDeltaY=-2.500 px FAIL; visit NeI robustRmsDeltaY=0.300 px WARN"
    )
    assert list(found["recommendedAction"]) == ["APPLY_SHIFT", "RECALIBRATE"]
    assert found["setup"].str.startswith("scienceArc 'Arc: Neon'").all()
    assert (found["extent"] == "detector").all()
    assert found["validated"].all() and str(found.iloc[0]["night"]) == "2026-01-02"
    assert found.columns[-1] == "recommendedAction"  # the standard findings columns first
    # Negative control: a PASS image with a PASS row is no finding, and UNKNOWN is not one either.
    assert 3 not in set(found["visit"]) and (2, "b", 2) not in set(
        found[KEYS].itertuples(index=False, name=None)
    )


def testResidualProblems(periods, visit, listing):
    found = residualFindings(CROSSINGS, _summary(), _visits(periods, visit, listing))
    problems = residualProblems(found)
    assert list(problems["status"]) == ["FAIL", "WARN"]
    assert problems.iloc[0]["crossed"] == "calib NeI medianDeltaY, visit NeI robustRmsDeltaY"
    assert problems.iloc[1]["crossed"] == "calib Trace medianDeltaX"
    assert list(problems["action"]) == ["RECALIBRATE", "APPLY_SHIFT"]
    assert problems.iloc[0]["setup"] == "scienceArc 'Arc: Neon'"  # the exposure time dropped
    assert list(problems["images"]) == [1, 1] and list(problems["nights"]) == ["2026-01-02", "2026-01-02"]
    assert residualProblems(found.iloc[0:0]).empty


def testScatterTable():
    reserved = [
        (v, "b", sm, "visit", "NeI", "RESERVED", metric, value, "")
        for v in (1, 2, 3)
        for sm in (1,)
        for metric, value in (("robustRmsDeltaX", 0.1 * v), ("robustRmsDeltaY", 0.2 * v))
    ]
    ignored = [
        (1, "b", 1, "visit", "NeI", "BOTH", "robustRmsDeltaX", 9.0, ""),  # not the held-back lines
        (1, "b", 1, "calib", "NeI", "BOTH", "robustRmsDeltaX", 9.0, ""),  # not the visit's map
        (1, "b", 1, "visit", "Trace", "RESERVED", "medianDeltaX", 9.0, ""),  # not a scatter
    ]
    metrics = _metrics([*reserved, *ignored])
    reference = _metrics([(7, "b", 1, "visit", "NeI", "RESERVED", "robustRmsDeltaX", 0.05, "")])
    table = scatterTable(metrics, reference, labels=("run2", "run1"))
    assert list(table[["arm", "description", "component"]].itertuples(index=False, name=None)) == [
        ("b", "NeI", "x"),
        ("b", "NeI", "y"),
    ]
    assert list(table["images run2"]) == [3, 3] and list(table["median run2"]) == [0.2, 0.4]
    assert table.iloc[0]["median run1"] == 0.05 and pd.isna(table.iloc[1]["median run1"])
    assert "p95 run2" in table and table.iloc[0]["p95 run2"] < 0.3
    assert scatterTable(_metrics(ignored)).empty
    # With the frame type stored, traces on a quartz and on an arc are separate rows; a reference
    # without it keeps the table by arm and species alone.
    traces = _metrics(
        [
            (v, "b", 1, "visit", "Trace", "RESERVED", "robustRmsDeltaX", value, "")
            for v, value in ((1, 0.04), (2, 0.12), (3, 0.13))
        ]
    ).assign(obsType=["trace", "arc", "arc"])
    byType = scatterTable(traces, labels=("run2", "run1"))
    assert list(
        byType[["arm", "obsType", "description", "component", "images run2"]].itertuples(
            index=False, name=None
        )
    ) == [
        ("b", "arc", "Trace", "x", 2),
        ("b", "trace", "Trace", "x", 1),
    ]
    pooled = scatterTable(traces, reference, labels=("run2", "run1"))
    assert "obsType" not in pooled.columns and list(pooled["images run2"]) == [3]
    # The b arm read with m and with r are separate rows (PIPE2D-1927), each beside the reference's
    # of the same camera set; a set the reference never read has no reference values.
    traces = traces.drop(columns="obsType").assign(cameraSet=["bm", "br", "br"])
    traceReference = _metrics([(7, "b", 1, "visit", "Trace", "RESERVED", "robustRmsDeltaX", 0.05, "")])
    bySet = scatterTable(traces, traceReference, labels=("run2", "run1"))
    assert list(bySet[["arm", "cameraSet", "images run2", "median run1"]].itertuples(index=False, name=None))[
        1
    ] == ("b", "br", 2, 0.05)
    assert bySet.iloc[0][["cameraSet", "images run2"]].tolist() == ["bm", 1]
    assert pd.isna(bySet.iloc[0]["median run1"])
    # Negative control: a reference without the column keeps the table pooled over camera sets.
    unsplit = scatterTable(traces, traceReference.drop(columns="cameraSet"), labels=("run2", "run1"))
    assert "cameraSet" not in unsplit.columns and list(unsplit["images run2"]) == [3]
