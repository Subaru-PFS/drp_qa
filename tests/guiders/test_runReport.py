"""Tests for the end-of-run AG report: its reader, plots and command line, with a fake opdb."""

import datetime
import json
import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from pfs.drp.qa.comparison.runs import Period
from pfs.drp.qa.guiders import runPlotting
from pfs.drp.qa.guiders.queries import AG_RUN_TABLES, readAgRun
from pfs.drp.qa.guiders.runAnalysis import (
    findRejectionClusters,
    fitModelErrors,
    nightPosition,
    prepareRun,
    settledJitter,
)
from pfs.drp.qa.guiders.runReport import fetchRun, loadRun, main

# The table behind each readAgRun query, as FakeOpDB looks them up.
RUN_QUERIES = {
    "grid": r"AS dx_um",  # before cameras: both join agc_data
    "cameras": r"JOIN agc_data ON",
    "exposures": r"LEFT JOIN agc_guide_offset",
    "visits": r"SELECT DISTINCT ON \(pfs_visit\.pfs_visit_id\)",
    "focus": r"FROM tel_status",
    "designStars": r"FROM pfs_design_agc",
}
PERIOD = Period("runX", "runX", datetime.date(2026, 9, 1), datetime.date(2026, 9, 2))


@pytest.fixture
def opdb(FakeOpDB, makeAgRunTables):
    return FakeOpDB(makeAgRunTables(deadFrom={0: 200}), patterns=RUN_QUERIES)


def testReadAgRun(opdb):
    start, end = datetime.datetime(2026, 9, 1, 12), datetime.datetime(2026, 9, 3, 12)
    tables = readAgRun(opdb, start, end, magLimit=19)

    assert tuple(tables) == AG_RUN_TABLES
    assert len(opdb.calls) == len(AG_RUN_TABLES)
    for call in opdb.calls:
        assert call.params["start"] == start and call.params["end"] == end
        assert not re.search(r"2026", call.sql)  # bound, not formatted
    assert opdb.calls[-1].params["magLimit"] == 19.0
    assert isinstance(opdb.calls[-1].params["magLimit"], float)


def testReadAgRunNeedsOpdb():
    with pytest.raises(TypeError):
        readAgRun(object(), datetime.datetime(2026, 9, 1), datetime.datetime(2026, 9, 2))


def testPlots(makeAgRunTables):
    run = prepareRun(makeAgRunTables(deadFrom={0: 200}, modelError={2: [[0.0, 0.0], [0.0, 0.013]]}))
    jitter = settledJitter(run)
    clusters, modelErrors = findRejectionClusters(run), fitModelErrors(run)
    figures = {
        "presence": runPlotting.plotCameraPresence(run),
        "guiding": runPlotting.plotGuiding(run, jitter),
        "focus": runPlotting.plotFocus(run),
        "acquisition": runPlotting.plotAcquisition(run),
        **{f"AG{cid + 1}": runPlotting.plotCamera(run, cid, clusters, modelErrors) for cid in range(6)},
    }
    assert all(isinstance(fig, Figure) for fig in figures.values())
    assert len(figures["focus"].axes) == 2
    assert all(len(figures[f"AG{cid + 1}"].axes) == 6 for cid in range(6))
    assert figures["AG3"]._suptitle.get_text() == "AG3 (AG1/3/5)"
    presence = figures["presence"].axes[0]
    assert [t.get_text() for t in presence.get_yticklabels()] == ["AG1", "AG3", "AG5", "AG2", "AG4", "AG6"]


def testGuidingPlotWithAnAxisAllNaN(makeAgRunTables):
    """A jitter axis with no values (here ΔInR) leaves an empty panel rather than failing."""
    tables = makeAgRunTables()
    tables["exposures"]["guide_delta_insrot"] = float("nan")
    run = prepareRun(tables)
    fig = runPlotting.plotGuiding(run, settledJitter(run))

    assert fig.axes[2].get_ylim() == (0.0, 1.0)


def testPresenceCrossesDeadCameras(makeAgRunTables):
    """A cell is crossed when most of a camera's misses detected nothing, even if most frames matched."""
    run = prepareRun(makeAgRunTables(nNights=1, nVisits=4, nExp=90, deadFrom={0: 330}))
    crosses = [line for line in runPlotting.plotCameraPresence(run).axes[0].lines if line.get_marker() == "x"]
    # AG1 dies 55 min into the night: its first hour is 8% dead, and still crossed; the hours after are all dead.
    hours = np.floor(nightPosition(run.frames.taken_at, run.plottedNights) * 12)
    assert len(crosses) == len(np.unique(hours[330:]))
    # Negative control: a camera that never died is never crossed.
    assert all(line.get_ydata()[0] < 1 for line in crosses)


def testFetchAndLoad(opdb, tmp_path):
    paths = fetchRun(opdb, PERIOD, tmp_path, now=datetime.datetime(2026, 10, 1))
    tables, stamp = loadRun(PERIOD, tmp_path)

    assert set(paths) == set(AG_RUN_TABLES)
    assert stamp["rows"]["exposures"] == 320
    assert stamp["readUntil"] == str(PERIOD.end)
    pd.testing.assert_frame_equal(tables["exposures"], opdb.tables["exposures"].reset_index(drop=True))


def testFetchStopsNow(opdb, tmp_path):
    """A period still under way is read up to now."""
    now = datetime.datetime(2026, 9, 2, 3, 0)
    fetchRun(opdb, PERIOD, tmp_path, now=now)

    assert opdb.calls[0].params["end"] == now


def testHostDefaultsToPortForward():
    """The fetch step reads through the ssh port-forward unless told otherwise: on the VPN, pfsa-db refuses public_user."""
    from pfs.drp.qa.guiders.runReport import _parser

    args = _parser().parse_args(["fetch", "--period", "run30"])
    assert (args.host, args.port, args.user) == ("127.0.0.1", 5432, "public_user")


def testLoadUnfetched(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch first"):
        loadRun(PERIOD, tmp_path)


def testReportCommand(opdb, tmp_path):
    runs = tmp_path / "runs.yaml"
    runs.write_text('runs:\n  - name: runX\n    nights: ["2026-09-01", "2026-09-02"]\n')
    fetchRun(opdb, PERIOD, tmp_path, now=datetime.datetime(2026, 10, 1))
    notes = tmp_path / "notes.md"
    notes.write_text("## Notes\n\nAG1 was unplugged.\n")

    assert (
        main(
            [
                "report",
                "--period",
                "runX",
                "--runs",
                str(runs),
                "--data-dir",
                str(tmp_path),
                "--notes",
                str(notes),
            ]
        )
        == 0
    )

    out = tmp_path / "runX"
    report = (out / "report.md").read_text()
    assert report.startswith("# runX: AG run report")
    assert "**AG1: no detections at all in 120 consecutive AG exposures**" in report
    assert "### AG1/3/5 against AG2/4/6" in report
    # One section per camera, in their groups, each with its own flags and figure.
    sections = re.findall(r"^### (AG\d)$", report, re.MULTILINE)
    assert sections == ["AG1", "AG3", "AG5", "AG2", "AG4", "AG6"]
    ag1 = report.split("### AG1\n")[1].split("###")[0]
    assert "no detections at all" in ag1 and "(camera-AG1.png)" in ag1
    assert "- No flags." in report.split("### AG2\n")[1].split("###")[0]
    assert report.rstrip().endswith("AG1 was unplugged.")
    for name in (
        "cameraPresence",
        "guiding",
        "focus",
        "acquisition",
        *(f"camera-AG{n}" for n in range(1, 7)),
    ):
        assert (out / f"{name}.png").stat().st_size > 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["dropouts"][0]["agc_camera_id"] == 0
    assert set(metrics["groups"]) == {"AG1/3/5", "AG2/4/6"}
    assert set(metrics["cameras"]) == {f"AG{n}" for n in range(1, 7)}


def testUnknownPeriod(tmp_path):
    with pytest.raises(SystemExit, match="Unknown period"):
        main(["report", "--period", "run99", "--data-dir", str(tmp_path)])
