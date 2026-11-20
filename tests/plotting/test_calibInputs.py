"""Smoke tests for `pfs.drp.qa.plotting.calibInputs` on a small synthetic set."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pfs.drp.qa.comparison.calibInputs import (
    detectorTable,
    gateNumbers,
    judge,
    setVisits,
    speciesTable,
    trends,
)
from pfs.drp.qa.metrics.normalValues import NORMAL_COLUMNS, candidates
from pfs.drp.qa.metrics.registry import CHECKS
from pfs.drp.qa.plotting.calibInputs import (
    VERDICT_COLORS,
    plotDetectorDetail,
    plotDetectorSeries,
    plotGoNoGoGrid,
    plotLampTrends,
    plotLineResiduals,
    plotVerdictGrid,
)


def _visits():
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
    return setVisits(listing)


def _metrics(visits, bad=()):
    """Return iqQaMetrics and dmLineResidualMetrics rows of the set: numbers, no verdicts.

    ``bad`` holds the (visit, spectrograph) of images with a tiny FWHM and a huge reserved RMS.
    """
    iq, dm = [], []
    for row in visits.itertuples():
        obsType = "trace" if row.species == ("Trace",) else "arc"
        for sp in (1, 2):
            fwhm, rms = (0.7, 2.0) if (row.visit, sp) in bad else (2.9, 0.08)
            iq.append((row.visit, "n", sp, obsType, "full", row.sequence_name, fwhm, 12.0, rms, rms))
            for species in row.species:
                for statusType, factor in (("BOTH", 1.0), ("RESERVED", 0.1)):
                    for metric, value in (
                        ("nLines", 1000 * factor),
                        ("linesPerFiber", 2 * factor),
                        ("robustRmsDeltaX", rms),
                        ("robustRmsDeltaY", rms),
                    ):
                        dm.append(
                            (row.visit, "n", sp, "brn", obsType, "visit", species, statusType, metric, value)
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
            "fitReservedXRms",
            "fitReservedYRms",
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
    return iq, dm


def _set(withNormalValues=True):
    """Return the set's visits, judged images, species and the gate's numbers; n2 is wrong on 102.

    The normal values are what the same set measures with nothing wrong.
    """
    visits = _visits()
    iq, dm = _metrics(visits, bad={(102, 2)})
    normalValues = None
    if withNormalValues:
        made = candidates(*_metrics(visits))
        normalValues = made.assign(count=9, validFrom=1).reindex(columns=list(NORMAL_COLUMNS))
    judged = judge(detectorTable(visits, iq, dm, normalValues))
    return visits, judged, speciesTable(visits, dm), gateNumbers(iq, dm, normalValues)


def testVerdictGrid():
    visits, judged, _, _ = _set()
    fig = plotVerdictGrid(judged, visits)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["n1", "n2"]
    assert len(ax.get_xticklabels()) == 5  # every set visit is a column
    assert set(VERDICT_COLORS) == {"use", "weak", "check", "out"}


def testDetectorSeries():
    _, judged, _, _ = _set()
    fig = plotDetectorSeries(judged)
    assert len(fig.axes) == 2  # FWHM and RMS for the one arm
    assert fig.axes[1].get_yscale() == "log"


def testLampTrends():
    _, judged, species, _ = _set()
    fig = plotLampTrends(trends(judged, species))
    assert len(fig.axes) == 1
    assert fig.axes[0].get_legend() is not None


def testGoNoGoGrid():
    visits, judged, _, _ = _set()
    judged = judged[~((judged["visit"] == 100) & (judged["spectrograph"] == 1))]  # a missing cell
    visits.loc[visits["visit"] == 110, "cameras"] = "n1"  # and a never-exposed one
    fig = plotGoNoGoGrid(judged, visits)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["n1", "n2"]  # detectors across
    assert [t.get_text() for t in ax.get_yticklabels()][:2] == ["100", "101"]  # visits down, first at the top
    assert sum(t.get_text() == "×" for t in ax.texts) == 1  # the one no-go image is crossed


def testDetectorDetail():
    visits, judged, species, numbers = _set()
    fig = plotDetectorDetail(judged, species, visits, "n2", numbers=numbers)
    assert len(fig.axes) == 3
    assert fig.axes[1].get_yscale() == "log"
    assert fig.axes[2].get_legend() is not None
    # The gate's width limits: a warn and a fail mark at each of the five visits, above the normal 2.9 px.
    warn, fail = fig.axes[0].collections
    assert len(warn.get_segments()) == len(fail.get_segments()) == 5
    limits = CHECKS["width"].limits(2.9)
    assert {segment[0][1] for segment in warn.get_segments()} == {limits[0]}
    assert {segment[0][1] for segment in fail.get_segments()} == {limits[1]}
    assert sorted(segment[0][0] + 0.45 for segment in warn.get_segments()) == [
        0,
        1,
        2,
        3,
        4,
    ]  # in visit order
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        "gate warn limit",
        "gate fail limit",
    ]
    # Negative control: a width is worse when higher, so no limit lies below the normal value.
    assert all(segment[0][1] > 2.9 for line in (warn, fail) for segment in line.get_segments())
    # Negative control: without the gate's numbers, or without normal values behind them, no limit is drawn.
    assert not plotDetectorDetail(judged, species, visits, "n2").axes[0].collections
    visits, judged, species, unjudged = _set(withNormalValues=False)
    bare = plotDetectorDetail(judged, species, visits, "n2", numbers=unjudged)
    assert not bare.axes[0].collections and bare.axes[0].get_legend() is None


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
        limits={
            "NeI": {"robustRmsDeltaX": (0.1, 0.2), "robustRmsDeltaY": (None, 0.3)},
            "HgI": {"robustRmsDeltaX": (None, None), "robustRmsDeltaY": (None, None)},
        },
    )
    assert len(fig.axes) == 4
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "NeI warn RMS ±0.100" in labels and "NeI fail RMS ±0.200" in labels
    assert "HgI: no RMS limit" in labels
    assert fig.axes[0].get_legend() is not None
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any(t.startswith("reserved lines\nNeI: median") for t in texts)
    # The limit bands are the dashed lines: warn and fail in x, fail alone in y, either side of zero.
    dashed = [[line for line in ax.get_lines() if line.get_linestyle() == "--"] for ax in fig.axes]
    assert [len(lines) for lines in dashed] == [4, 4, 2, 2]
    # Negative control: without limits (no table of normal values) no band is drawn.
    bare = plotLineResiduals(data, title="n4 152676")
    assert not [line for ax in bare.axes for line in ax.get_lines() if line.get_linestyle() == "--"]
    assert not any("RMS" in t.get_text() for t in bare.axes[0].get_legend().get_texts())


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
