"""Tests for the run comparison (`pfs.drp.qa.metrics.runComparison`). Visits and values are synthetic."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics import runComparison
from pfs.drp.qa.metrics.registry import specFor
from pfs.drp.qa.metrics.runComparison import (
    COMPARISON_COLUMNS,
    compareRuns,
    exitCode,
    metricRows,
    populationMedians,
)
from pfs.drp.qa.metrics.visitRollup import EXIT_FAIL, EXIT_NO_DATA, EXIT_PASS, EXIT_WARN


def image(visit, camera, fwhm, lamp="Ne", illumination="pfi", status="PASS", **more):
    return {
        "visit": visit,
        "arm": camera[0],
        "spectrograph": int(camera[1:]),
        "obsType": "arc",
        "seqName": f"Arc: {lamp}",
        "illumination": illumination,
        "traceOnly": False,
        "medFwhm": fwhm,
        "linesPerFiber": 40.0,
        "pctFlagged": 2.0,
        "medDxCenter": -0.02,
        "qaStatus": status,
        **more,
    }


def iq(*rows):
    return pd.DataFrame(list(rows))


def residual(visit, camera, metric, value, status="PASS"):
    return {
        "visit": visit,
        "arm": camera[0],
        "spectrograph": int(camera[1:]),
        "obsType": "arc",
        "reference": "calib",
        "description": "NeI",
        "statusType": "USED",
        "metric": metric,
        "value": value,
        "status": status,
    }


REFERENCE = iq(image(1, "b1", 2.00), image(2, "b1", 2.02), image(3, "b1", 1.98), image(1, "r1", 3.00))


def one(comparison, **where):
    for column, value in where.items():
        comparison = comparison[comparison[column] == value]
    assert len(comparison) == 1, comparison
    return comparison.iloc[0]


def testColumnsAndMedians():
    run = iq(image(10, "b1", 2.02), image(11, "b1", 2.04), image(10, "r1", 3.03))
    comparison = compareRuns(run, {"before": REFERENCE})
    assert list(comparison.columns) == list(COMPARISON_COLUMNS)
    row = one(comparison, metric="medFwhm", camera="b1")
    assert (row["nRun"], row["nReference"]) == (2, 3)
    assert row["runMedian"] == pytest.approx(2.03) and row["referenceMedian"] == pytest.approx(2.00)
    assert row["difference"] == pytest.approx(0.03) and row["relative"] == pytest.approx(0.015)
    assert (row["warn"], row["fail"]) == specFor("medFwhm").tolerance
    assert row["status"] == "PASS" and row["species"] == "Ne" and row["illumination"] == "pfi"


@pytest.mark.parametrize(("fwhm", "status"), [(2.05, "PASS"), (2.07, "WARN"), (2.12, "FAIL")])
def testStatusByTolerance(fwhm, status):
    comparison = compareRuns(iq(image(10, "b1", fwhm)), {"before": REFERENCE})
    assert one(comparison, metric="medFwhm")["status"] == status


def testDirectionAndAbsoluteValue(monkeypatch):
    """Only the bad direction is judged, and an offset by its size."""
    narrower = one(compareRuns(iq(image(10, "b1", 1.80)), {"before": REFERENCE}), metric="medFwhm")
    assert narrower["relative"] == pytest.approx(-0.10) and narrower["status"] == "PASS"

    fewer = iq(image(10, "b1", 2.0, linesPerFiber=30.0))
    # No lower-is-worse metric declares a tolerance yet: give one to ``linesPerFiber`` for the test.
    assert not specFor("linesPerFiber").higherIsWorse and specFor("linesPerFiber").tolerance is None

    def declared(name):
        spec = specFor(name)
        return dataclasses.replace(spec, tolerance=(0.1, 0.2)) if name == "linesPerFiber" else spec

    monkeypatch.setattr(runComparison, "specFor", declared)
    row = one(compareRuns(fewer, {"before": REFERENCE}), metric="linesPerFiber")
    assert row["relative"] == pytest.approx(-0.25) and row["status"] == "FAIL"
    more = iq(image(10, "b1", 2.0, linesPerFiber=50.0))
    assert one(compareRuns(more, {"before": REFERENCE}), metric="linesPerFiber")["status"] == "PASS"

    offset = iq(image(10, "b1", 2.0, medDxCenter=0.03), image(11, "b1", 2.0, medDxCenter=-0.03))
    row = one(compareRuns(offset, {"before": REFERENCE}), metric="medDxCenter")
    assert row["runMedian"] == pytest.approx(0.03) and row["referenceMedian"] == pytest.approx(0.02)
    # Negative control: the signed median of the run's offsets is zero, which would hide the change.
    assert not np.isclose(row["runMedian"], offset["medDxCenter"].median())
    # No tolerance is declared for the offset: it is reported, not judged.
    assert row["status"] == "UNKNOWN" and np.isnan(row["warn"])


def testPopulationsAreNotPooled():
    """A lamp or an illumination the reference lacks is not judged against another."""
    reference = iq(image(1, "b1", 2.0, lamp="Ne"), image(2, "b1", 3.0, lamp="Kr"))
    run = iq(
        image(10, "b1", 3.0, lamp="Kr"),
        image(11, "b1", 3.0, lamp="Ar"),
        image(12, "b1", 3.0, lamp="Ne", illumination="iis"),
    )
    comparison = compareRuns(run, {"before": reference}, metrics=["medFwhm"])
    assert len(comparison) == 3
    assert one(comparison, species="Kr")["status"] == "PASS"
    for unmatched in (one(comparison, species="Ar"), one(comparison, illumination="iis")):
        assert unmatched["status"] == "UNKNOWN" and unmatched["nReference"] == 0
        assert np.isnan(unmatched["referenceMedian"])
    # Negative control: pooled over lamps the Kr image would sit 20 % above the reference and fail.
    pooled = reference["medFwhm"].median()
    assert 3.0 / pooled - 1 > specFor("medFwhm").tolerance[1]


def testKnownGoodLeavesOutFlaggedReferenceValues():
    reference = iq(image(1, "b1", 2.0), image(2, "b1", 2.0), image(3, "b1", 4.0, status="FAIL"))
    reference = pd.concat([reference, iq(image(4, "b1", 4.0, status="WARN"))], ignore_index=True)
    run = iq(image(10, "b1", 2.5, status="FAIL"))
    good = one(compareRuns(run, {"before": reference}), metric="medFwhm")
    assert good["referenceMedian"] == pytest.approx(2.0) and good["nReference"] == 2
    assert good["status"] == "FAIL" and good["nRun"] == 1
    # Negative control: with the flagged values the reference median moves to 3.0 and the run passes.
    everything = one(compareRuns(run, {"before": reference}, knownGood=False), metric="medFwhm")
    assert everything["referenceMedian"] == pytest.approx(3.0) and everything["status"] == "PASS"


def testNotMeasuredValuesAreDropped():
    run = iq(image(10, "b1", 2.0), image(11, "b1", 9.0, traceOnly=True))
    rows = metricRows(run, ["medFwhm", "linesPerFiber"])
    assert rows[rows["metric"] == "medFwhm"]["visit"].tolist() == [10]
    # Negative control: the same image's other metrics were measured and stay.
    assert rows[rows["metric"] == "linesPerFiber"]["visit"].tolist() == [10, 11]
    assert metricRows(iq(image(10, "b1", np.nan)), ["medFwhm"]).empty


def testLongTablesAndSeveralReferences():
    run = [
        iq(image(10, "b1", 2.0)),
        pd.DataFrame(
            [
                residual(10, "b1", "robustRmsDeltaX", 0.13),
                residual(10, "b1", "medianDeltaX", -0.05),
                residual(10, "b1", "nLines", 80),
            ]
        ),
    ]
    before = [REFERENCE, pd.DataFrame([residual(1, "b1", "robustRmsDeltaX", 0.10)])]
    night = pd.DataFrame([residual(5, "b1", "robustRmsDeltaX", 0.125)])
    comparison = compareRuns(run, {"before": before, "firstNight": night})
    assert comparison["referenceName"].unique().tolist() == ["before", "firstNight"]
    assert "nLines" not in set(comparison["metric"])
    rms = comparison[comparison["metric"] == "robustRmsDeltaX"].set_index("referenceName")
    assert rms["status"].to_dict() == {"before": "WARN", "firstNight": "PASS"}
    assert rms["description"].unique().tolist() == ["NeI"]
    assert one(comparison, metric="medianDeltaX", referenceName="before")["runMedian"] == pytest.approx(0.05)
    # The first night holds no image-quality rows: those are listed, unjudged.
    assert one(comparison, metric="medFwhm", referenceName="firstNight")["status"] == "UNKNOWN"


def testMediansPerPopulation():
    medians = populationMedians(metricRows(REFERENCE, ["medFwhm"]))
    assert medians.set_index(["arm", "spectrograph"])["median"].to_dict() == {("b", 1): 2.0, ("r", 1): 3.0}
    assert medians["n"].tolist() == [3, 1]


def testExitCode():
    def status(fwhm):
        return exitCode(compareRuns(iq(image(10, "b1", fwhm)), {"before": REFERENCE}, metrics=["medFwhm"]))

    assert (status(2.0), status(2.07), status(2.2)) == (EXIT_PASS, EXIT_WARN, EXIT_FAIL)
    unjudged = compareRuns(iq(image(10, "b1", 2.0)), {"before": REFERENCE}, metrics=["pctFlagged"])
    assert len(unjudged) == 1 and exitCode(unjudged) == EXIT_NO_DATA
    empty = compareRuns(iq(), {"before": REFERENCE})
    assert empty.empty and list(empty.columns) == list(COMPARISON_COLUMNS) and exitCode(empty) == EXIT_NO_DATA
    assert compareRuns(REFERENCE, {}).empty


def testToleranceStatusOfSingleValues():
    """A visit's own value is judged against a reference median as a run's median is."""
    metric = pd.Series(["medFwhm", "medFwhm", "medFwhm", "medDxCenter", "medFwhm"], index=[5, 6, 7, 8, 9])
    relative = pd.Series([0.0, 0.04, 0.06, 0.5, np.nan], index=metric.index)
    judged = runComparison.toleranceStatus(metric, relative)
    assert judged["status"].tolist() == ["PASS", "WARN", "FAIL", "UNKNOWN", "UNKNOWN"]
    assert judged.index.tolist() == metric.index.tolist()
    assert judged["warn"].iloc[0] == specFor("medFwhm").tolerance[0] and np.isnan(judged["warn"].iloc[3])
    # Negative control: narrower by as much is not worse.
    assert set(runComparison.toleranceStatus(metric[:3], -relative[:3])["status"]) == {"PASS"}
    assert runComparison.toleranceStatus(metric[:0], relative[:0]).empty
