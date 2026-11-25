"""Tests for the end-of-run AG report: its reader, plots and command line, with a fake opdb."""

import datetime
import json
import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from pfs.drp.qa.comparison.runs import Period
from pfs.drp.qa.guiders import runPlotting
from pfs.drp.qa.guiders.queries import AG_DETECTION_FLAGS, AG_MAGNITUDE_BIN, AG_RUN_TABLES, readAgRun
from pfs.drp.qa.guiders.runAnalysis import (
    DETECTION_FLAG_COLUMNS,
    RunConfig,
    findRejectionClusters,
    fitModelErrors,
    prepareRun,
    runSummary,
    settledJitter,
)
from pfs.drp.qa.guiders.runReport import FIGURES, fetchRun, keyNumbers, loadRun, main
from pfs.utils.datamodel.ag import SourceDetectionFlags, SourceMatchingFlags

# The table behind each readAgRun query, as FakeOpDB looks them up.
RUN_QUERIES = {
    "magnitudes": r"AS n_expected",  # before designStars: both read pfs_design_agc
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
    (designStars,) = [call for call in opdb.calls if "magLimit" in call.params]
    assert designStars.params["magLimit"] == 19.0
    assert isinstance(designStars.params["magLimit"], float)
    (magnitudes,) = [call for call in opdb.calls if "magBin" in call.params]
    assert magnitudes.params["magBin"] == AG_MAGNITUDE_BIN


def testReadAgRunBindsFlagValues(opdb):
    readAgRun(opdb, datetime.datetime(2026, 9, 1, 12), datetime.datetime(2026, 9, 3, 12))
    literal = r"flags\s*(=|<>)\s*1\b|&\s*1\b"
    withFlags = [call for call in opdb.calls if "agc_match.flags" in call.sql]
    assert len(withFlags) == 3  # cameras, grid and magnitudes
    for call in withFlags:
        assert not re.search(literal, call.sql)
        assert call.params["goodMatch"] == int(SourceMatchingFlags.GOOD_MATCH)
    for call in withFlags[:2]:
        assert call.params["right"] == int(SourceDetectionFlags.RIGHT)
    (cameras,) = [call for call in opdb.calls if "sigma_pix" in call.sql]
    assert cameras.params["badShape"] == int(SourceDetectionFlags.BAD_SHAPE)
    # A good match is GOOD_MATCH on a clean detection (NONE or RIGHT) of a camera the guider had not left out:
    # the count, the image size and the position residual all take the same ones.
    good = (
        r"FILTER \(WHERE agc_match\.flags = :goodMatch AND COALESCE\(agc_data\.flags, 0\) <= :goodDetection"
        r" AND agc_match\.camera_enabled IS NOT FALSE\) AS "
    )
    assert cameras.params["goodDetection"] == int(SourceDetectionFlags.NONE | SourceDetectionFlags.RIGHT)
    for column in ("n_good", "sigma_pix", "residual_um"):
        assert re.search(good + column, cameras.sql)
    # Negative control: a filter on the match alone isn't that definition.
    assert not re.search(good + "n_good", "FILTER (WHERE agc_match.flags = :goodMatch) AS n_good")
    # Each bad-detection flag is counted by its own bit, bound by name.
    assert set(AG_DETECTION_FLAGS) == set(DETECTION_FLAG_COLUMNS)
    for column, flag in AG_DETECTION_FLAGS.items():
        assert cameras.params[column] == int(flag) and flag in SourceDetectionFlags.BAD_DETECTION
        assert f"COALESCE(agc_data.flags, 0) & :{column} <> 0) AS {column}" in cameras.sql
    assert "FILTER (WHERE agc_match.camera_enabled IS FALSE) AS n_disabled" in cameras.sql
    # sigma_pix is the radius storedRadiusToFwhm takes, sqrt((mxx + myy) / 2) of the variance: the stored
    # moments times 2 where the exposure's agcc stored half of it. Only sigma_pix leaves BAD_SHAPE out.
    assert cameras.params["halvedMoments"] == 2.0
    assert re.search(
        r"moment_20_pix \+ agc_data\.central_image_moment_02_pix, 0\) / 2\s+\* CASE\s+"
        r"WHEN agc_exposure\.version_actor = 'git' AND COALESCE\(agc_data\.flags, 0\) & :badShape = 0\s+"
        r"THEN :halvedMoments ELSE 1\s+END\s+\)",
        cameras.sql,
    )
    assert cameras.sql.count(":badShape") == 1 and cameras.sql.count(":halvedMoments") == 1
    # Negative control: the pattern finds the literals the SQL used to hold.
    assert re.search(literal, "FILTER (WHERE agc_match.flags = 1)")
    assert re.search(literal, "AS INTEGER)) & 1 AS right_half")


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
    # Size and jitter for each of the three axes, and the guide-loop problems.
    assert len(figures["guiding"].axes) == 7
    assert all(len(figures[f"AG{cid + 1}"].axes) == 8 for cid in range(6))
    assert figures["AG3"]._suptitle.get_text() == "AG3 (AG1/3/5)"
    presence = figures["presence"].axes[0]
    assert [t.get_text() for t in presence.get_yticklabels()] == ["AG1", "AG3", "AG5", "AG2", "AG4", "AG6"]


def testGuidingPlotWithAnAxisAllNaN(makeAgRunTables):
    """An axis with no values (here ΔInR) leaves empty panels rather than failing."""
    tables = makeAgRunTables()
    tables["exposures"]["guide_delta_insrot"] = float("nan")
    run = prepareRun(tables)
    fig = runPlotting.plotGuiding(run, settledJitter(run))

    assert fig.axes[4].get_ylim() == (0.0, 1.0) and fig.axes[5].get_ylim() == (0.0, 1.0)


def _cellColours(run):
    """Return the face colours of the presence map's cells, by camera row (its y)."""
    colours = {}
    for patch in runPlotting.plotCameraPresence(run).axes[0].patches:
        if patch.get_hatch() is None:
            colours.setdefault(patch.get_y(), []).append(to_hex(patch.get_facecolor()))
    return colours


def testPresenceColoursWhatACameraDid(makeAgRunTables):
    """A cell takes the colour of what the camera mostly did when it matched nothing."""
    tables = makeAgRunTables(nNights=1, nVisits=4, nExp=90, deadFrom={0: 0}, disabledFrom={4: 0})
    cameras = tables["cameras"]
    cameras.loc[cameras.agc_camera_id == 2, "n_good"] = 0  # AG3 detects, and matches nothing
    # AG1 would be not in use: let it see one spot, so that it is a camera that went dark.
    first = tables["exposures"].agc_exposure_id.iloc[0]
    cameras.loc[len(cameras)] = [first, 0, 0, 4, 3, 2.0, 100.0, 12.0, 0, 1, 0, 0, 0, 0]
    run = prepareRun(tables)
    colours = _cellColours(run)

    # Rows: AG1 at y=0, AG3 at 1, AG5 at 2, then AG2/4/6 from 3.5. The first hour of AG1 holds its one match.
    assert set(colours[0][1:]) == {runPlotting.BAD.lower()}
    assert set(colours[1]) == {runPlotting.WARN.lower()}
    assert set(colours[2]) == {runPlotting.OFF.lower()}
    # Negative control: a camera that matched throughout has no coloured cell.
    assert set(colours[4.5]) == {runPlotting.EMPTY.lower()}


def testPresenceNamesACameraNotInUse(makeAgRunTables):
    run = prepareRun(makeAgRunTables(deadFrom={1: 0}))
    ax = runPlotting.plotCameraPresence(run).axes[0]

    assert "not in use: no detections in the run" in [text.get_text() for text in ax.texts]
    # Negative control: with every camera in use, no row says so.
    inUse = runPlotting.plotCameraPresence(prepareRun(makeAgRunTables())).axes[0]
    assert "not in use: no detections in the run" not in [text.get_text() for text in inUse.texts]
    # Its row has no cells of its own (AG2 is the fourth row, at y=3.5).
    assert 3.5 not in _cellColours(run)
    title = runPlotting.plotCamera(run, 1, findRejectionClusters(run), fitModelErrors(run))._suptitle
    assert title.get_text() == "AG2 (AG2/4/6): not in use (no detections in the run)"


def testCaptionsCoverEveryFigure():
    captions = runPlotting.captions(RunConfig(nSettle=7, minSettled=11, minDesignStars=4, maxCorrection=8))

    assert set(captions) == {*FIGURES, "camera"}
    assert "first 7 guide frames" in captions["guiding"] and "at least 11" in captions["guiding"]
    assert "fewer than 4 bright" in captions["cameraPresence"]
    assert "±8″" in captions["acquisition"]
    # Negative control: the numbers follow the configuration, they aren't written in.
    assert "first 7 guide frames" not in runPlotting.captions()["guiding"]


def testKeyNumbers(makeAgRunTables):
    run = prepareRun(makeAgRunTables(deadFrom={1: 0}))
    numbers = {what: value for what, value, _ in keyNumbers(runSummary(run, settledJitter(run)))}

    assert numbers["Cameras in use"] == "AG1, AG3, AG4, AG5, AG6"
    assert numbers["Star position, measured minus expected"].startswith("12.0 µm = 0.92 px")
    assert numbers["Matched stars per AG exposure"].startswith("30 (5%: 30); AG1 6, AG3 6")
    assert re.fullmatch(r"0\.0\d\d″ \(95%: 0\.0\d\d″\)", numbers["Guide error, ΔDec"])
    # The two axes on the sky together come first among the guide errors.
    assert list(numbers)[1] == "Guide error, ΔRA and ΔDec together"
    assert numbers["Designed guide stars matched"] == "75% (5% of visits: 75%, 95%: 75%)"
    assert numbers["Guide offsets rejected / not written"] == "0 / 0"


def testFetchAndLoad(opdb, tmp_path):
    paths = fetchRun(opdb, PERIOD, tmp_path, now=datetime.datetime(2026, 10, 1))
    tables, stamp = loadRun(PERIOD, tmp_path)

    assert set(paths) == set(AG_RUN_TABLES)
    assert stamp["rows"]["exposures"] == 320
    assert stamp["readUntil"] == str(PERIOD.end)
    assert stamp["sigmaPix"] == "variance"
    pd.testing.assert_frame_equal(tables["exposures"], opdb.tables["exposures"].reset_index(drop=True))
    pd.testing.assert_frame_equal(tables["cameras"], opdb.tables["cameras"].reset_index(drop=True))


def testLoadRescalesSigmaPixOfAnOlderFetch(opdb, tmp_path):
    """A fetch from when sigma_pix was the radius of half the variance loads as the radius of the variance."""
    fetchRun(opdb, PERIOD, tmp_path, now=datetime.datetime(2026, 10, 1))
    fetched, _ = loadRun(PERIOD, tmp_path)
    stampPath = tmp_path / "opdb" / "agRun-runX-fetched.json"
    stamp = json.loads(stampPath.read_text())
    del stamp["sigmaPix"]
    stampPath.write_text(json.dumps(stamp))

    older, _ = loadRun(PERIOD, tmp_path)

    np.testing.assert_allclose(older["cameras"].sigma_pix, np.sqrt(2) * fetched["cameras"].sigma_pix)
    pd.testing.assert_frame_equal(
        older["cameras"].drop(columns="sigma_pix"), fetched["cameras"].drop(columns="sigma_pix")
    )
    # Negative control: a fetch stamped with the variance is read as it is.
    assert not np.allclose(fetched["cameras"].sigma_pix, older["cameras"].sigma_pix)


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
    # The key numbers come first, and every figure says what it shows.
    assert report.index("## Key numbers") < report.index("## Flags")
    assert "| Guide error, ΔRA on the sky | 0.0" in report
    captions = runPlotting.captions()
    for name in FIGURES:
        assert f"({name}.png)\n\n{captions[name]}" in report
    assert captions["camera"] in report and runPlotting.GROUPS_NOTE in report
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
    assert metrics["summary"]["camerasInUse"] == [f"AG{n}" for n in range(1, 7)]
    assert set(metrics["summary"]["guideError"]) == {"ra", "dec", "inr", "sky"}
    assert set(metrics["groups"]) == {"AG1/3/5", "AG2/4/6"}
    assert set(metrics["cameras"]) == {f"AG{n}" for n in range(1, 7)}


def testReportLeavesOutTheFigureOfACameraNotInUse(FakeOpDB, makeAgRunTables, tmp_path):
    runs = tmp_path / "runs.yaml"
    runs.write_text('runs:\n  - name: runX\n    nights: ["2026-09-01", "2026-09-02"]\n')
    fetchRun(FakeOpDB(makeAgRunTables(deadFrom={1: 0}), RUN_QUERIES), PERIOD, tmp_path)

    assert main(["report", "--period", "runX", "--runs", str(runs), "--data-dir", str(tmp_path)]) == 0
    section = (tmp_path / "runX" / "report.md").read_text().split("### AG2\n")[1].split("###")[0]
    assert "AG2: not in use" in section and "camera-AG2.png" not in section
    assert not (tmp_path / "runX" / "camera-AG2.png").exists()
    # Negative control: the cameras in use keep theirs.
    assert (tmp_path / "runX" / "camera-AG4.png").exists()


def testUnknownPeriod(tmp_path):
    with pytest.raises(SystemExit, match="Unknown period"):
        main(["report", "--period", "run99", "--data-dir", str(tmp_path)])


def testReportFromAnOlderFetch(opdb, tmp_path):
    """A fetch made before the grid and magnitudes tables existed still reports, and says what it left out."""
    runs = tmp_path / "runs.yaml"
    runs.write_text('runs:\n  - name: runX\n    nights: ["2026-09-01", "2026-09-02"]\n')
    fetchRun(opdb, PERIOD, tmp_path, now=datetime.datetime(2026, 10, 1))
    (tmp_path / "opdb" / "agRun-runX-grid.parquet").unlink()
    (tmp_path / "opdb" / "agRun-runX-magnitudes.parquet").unlink()
    # Nor did its cameras hold the residual, the disabled matches or the flag counts.
    camerasPath = tmp_path / "opdb" / "agRun-runX-cameras.parquet"
    older = pd.read_parquet(camerasPath).drop(columns=["residual_um", "n_disabled", *DETECTION_FLAG_COLUMNS])
    older.to_parquet(camerasPath)
    tables, stamp = loadRun(PERIOD, tmp_path)

    assert "grid" not in tables and stamp["missing"] == ["grid", "magnitudes"]
    assert main(["report", "--period", "runX", "--runs", str(runs), "--data-dir", str(tmp_path)]) == 0
    assert "**No `grid` table was read**" in (tmp_path / "runX" / "report.md").read_text()
    # Negative control: any other table is required.
    (tmp_path / "opdb" / "agRun-runX-exposures.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="run fetch again"):
        loadRun(PERIOD, tmp_path)
