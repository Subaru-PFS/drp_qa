"""Tests for the end-of-run AG report: its reader, plots and command line, with a fake opdb."""

import datetime
import json
import re

import pandas as pd
import pytest
from matplotlib.figure import Figure

from pfs.drp.qa.comparison.runs import Period
from pfs.drp.qa.guiders import runPlotting
from pfs.drp.qa.guiders.queries import AG_RUN_TABLES, readAgRun
from pfs.drp.qa.guiders.runAnalysis import prepareRun, settledJitter
from pfs.drp.qa.guiders.runReport import fetchRun, loadRun, main

# The table behind each readAgRun query, as FakeOpDB looks them up.
RUN_QUERIES = {
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
    run = prepareRun(makeAgRunTables(deadFrom={0: 200}))
    jitter = settledJitter(run)
    figures = {
        "presence": runPlotting.plotCameraPresence(run),
        "perCamera": runPlotting.plotPerCamera(run),
        "guiding": runPlotting.plotGuiding(run, jitter),
        "focus": runPlotting.plotFocus(run),
        "acquisition": runPlotting.plotAcquisition(run),
    }
    assert all(isinstance(fig, Figure) for fig in figures.values())
    assert len(figures["perCamera"].axes) == 12
    assert len(figures["focus"].axes) == 8
    # The per-camera panels go AG1, AG3, AG5, then AG2, AG4, AG6.
    axes = figures["perCamera"].axes  # four rows of three: stars, FWHM, stars, FWHM
    titles = [ax.get_title(loc="left") for ax in axes[0:3] + axes[6:9]]
    assert [t.split(":")[0] for t in titles] == ["AG1", "AG3", "AG5", "AG2", "AG4", "AG6"]
    presence = figures["presence"].axes[0]
    assert [t.get_text() for t in presence.get_yticklabels()] == ["AG1", "AG3", "AG5", "AG2", "AG4", "AG6"]


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
    assert report.rstrip().endswith("AG1 was unplugged.")
    for name in ("cameraPresence", "perCamera", "guiding", "focus", "acquisition"):
        assert (out / f"{name}.png").stat().st_size > 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["dropouts"][0]["agc_camera_id"] == 0
    assert set(metrics["groups"]) == {"AG1/3/5", "AG2/4/6"}


def testUnknownPeriod(tmp_path):
    with pytest.raises(SystemExit, match="Unknown period"):
        main(["report", "--period", "run99", "--data-dir", str(tmp_path)])
