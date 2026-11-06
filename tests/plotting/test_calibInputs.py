"""Smoke tests for `pfs.drp.qa.plotting.calibInputs` on a small synthetic set."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pfs.drp.qa.comparison.calibInputs import detectorTable, judge, setVisits, speciesTable, trends
from pfs.drp.qa.plotting.calibInputs import (
    VERDICT_COLORS,
    plotDetectorDetail,
    plotDetectorSeries,
    plotGoNoGoGrid,
    plotLampTrends,
    plotLineResiduals,
    plotVerdictGrid,
)


def _set():
    rows = []
    start = pd.Timestamp("2026-10-31 03:30")
    for k in range(3):
        rows.append(
            (
                100 + k,
                70,
                "scienceArc",
                "Arc: Neon",
                "iic scienceArc neon=5",
                5.0,
                start + pd.Timedelta(minutes=k),
            )
        )
    for k in range(2):
        rows.append(
            (
                110 + k,
                71,
                "scienceTrace",
                "Trace",
                "iic scienceTrace halogen=30",
                30.0,
                start + pd.Timedelta(minutes=10 + k),
            )
        )
    listing = pd.DataFrame(
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
    ).assign(cameras="n1,n2")
    visits = setVisits(listing)
    iq, dm, summary = [], [], []
    for row in visits.itertuples():
        for sp in (1, 2):
            bad = row.visit == 102 and sp == 2
            fwhm, rms = (0.7, 2.0) if bad else (2.9, 0.08)
            iq.append((row.visit, "n", sp, fwhm, rms, rms, "FAIL" if bad else "PASS", "bad" if bad else ""))
            for species in row.species:
                for statusType, factor in (("BOTH", 1.0), ("RESERVED", 0.1)):
                    for metric, value in (
                        ("nLines", 1000 * factor),
                        ("linesPerFiber", 2 * factor),
                        ("robustRmsDeltaX", rms),
                        ("robustRmsDeltaY", rms),
                    ):
                        status = (
                            "FAIL"
                            if (bad and statusType == "RESERVED" and metric.startswith("robustRms"))
                            else ""
                        )
                        dm.append((row.visit, "n", sp, "visit", species, statusType, metric, value, status))
            summary.append((row.visit, "n", sp, "PASS", ""))
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
    judged = judge(detectorTable(visits, iq, dm, summary))
    return visits, judged, speciesTable(visits, dm)


def testVerdictGrid():
    visits, judged, _ = _set()
    fig = plotVerdictGrid(judged, visits)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["n1", "n2"]
    assert len(ax.get_xticklabels()) == 5  # every set visit is a column
    assert set(VERDICT_COLORS) == {"use", "weak", "check", "out"}


def testDetectorSeries():
    _, judged, _ = _set()
    fig = plotDetectorSeries(judged)
    assert len(fig.axes) == 2  # FWHM and RMS for the one arm
    assert fig.axes[1].get_yscale() == "log"


def testLampTrends():
    _visits, judged, species = _set()
    fig = plotLampTrends(trends(judged, species))
    assert len(fig.axes) == 1
    assert fig.axes[0].get_legend() is not None


def testGoNoGoGrid():
    visits, judged, _ = _set()
    judged = judged[~((judged["visit"] == 100) & (judged["spectrograph"] == 1))]  # a missing cell
    visits.loc[visits["visit"] == 110, "cameras"] = "n1"  # and a never-exposed one
    fig = plotGoNoGoGrid(judged, visits)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["n1", "n2"]  # detectors across
    assert [t.get_text() for t in ax.get_yticklabels()][:2] == ["100", "101"]  # visits down, first at the top
    assert sum(t.get_text() == "×" for t in ax.texts) == 1  # the one no-go image is crossed


def testDetectorDetail():
    visits, judged, species = _set()
    fig = plotDetectorDetail(judged, species, visits, "n2")
    assert len(fig.axes) == 3
    assert fig.axes[1].get_yscale() == "log"
    assert fig.axes[2].get_legend() is not None


def testLineResiduals():
    rng = np.random.default_rng(1)
    n = 200
    data = pd.DataFrame(
        {
            "fiberId": rng.integers(1, 600, n),
            "wavelength": rng.uniform(950, 1250, n),
            "description": rng.choice(["NeI", "Trace"], n),
            "statusType": rng.choice(["USED", "RESERVED"], n),
            "dxVisit": rng.normal(0, 0.05, n),
            "dyVisit": rng.normal(0, 0.05, n),
        }
    )
    fig = plotLineResiduals(
        data,
        title="n4 152676",
        thresholds={"NeI": {"robustRmsDeltaX": (0.1, 0.2), "robustRmsDeltaY": (None, 0.3)}},
    )
    assert len(fig.axes) == 4
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "NeI warn RMS ±0.100" in labels and "NeI fail RMS ±0.200" in labels
    assert fig.axes[0].get_legend() is not None
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any(t.startswith("reserved lines\nNeI: median") for t in texts)


def testCalibrationComparison():
    from pfs.drp.qa.comparison.calibValidate import compareCalibrations
    from pfs.drp.qa.plotting.calibInputs import plotCalibrationComparison

    rows = []
    for label, offset in (("asis", 0.5), ("pick", 0.1)):
        for visit in (100, 101, 200):
            for reference, statusType, metric, value in (
                ("calib", "BOTH", "medianDeltaX", offset),
                ("calib", "BOTH", "robustRmsDeltaX", 0.1),
                ("calib", "BOTH", "nLines", 100.0),
                ("visit", "RESERVED", "robustRmsDeltaX", 0.05),
                ("visit", "RESERVED", "nLines", 10.0),
            ):
                rows.append((label, visit, "b", 1, reference, "NeI", statusType, metric, value))
    frames = {}
    for label in ("asis", "pick"):
        frames[label] = pd.DataFrame(
            [r[1:] for r in rows if r[0] == label],
            columns=[
                "visit",
                "arm",
                "spectrograph",
                "reference",
                "description",
                "statusType",
                "metric",
                "value",
            ],
        )
    table = compareCalibrations(frames, setVisits=[100, 101])
    fig = plotCalibrationComparison(table[table["detector"] == "b1"], "b1")
    assert len(fig.axes) == 3  # offset x, RMS x vs calib, reserved RMS x
    assert fig.axes[0].get_legend() is not None
