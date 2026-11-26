"""Tests for ``pfs.drp.qa.guiders.runAnalysis``, on synthetic runs."""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders.runAnalysis import (
    CAMERA_GROUPS,
    DETECTION_FLAG_COLUMNS,
    MODEL_PARTS,
    CameraState,
    RunConfig,
    _toDetectorAxes,
    cameraFocus,
    cameraSummary,
    describePeriod,
    detectionFlagShares,
    detectorCells,
    findDropouts,
    findFlags,
    findRejectionClusters,
    fitModelErrors,
    groupSummary,
    isRejectedOffset,
    magnitudeSummary,
    nightPosition,
    onSkyRa,
    prepareRun,
    runSummary,
    stateShares,
    summariseModelErrors,
    visitGuideErrors,
    visitMatchFraction,
)
from pfs.utils.datamodel.ag import GuideOffsetFlag

# makeAgRunTables' defaults: 2 nights x 4 visits x 40 exposures.
N_EXPOSURES = 2 * 4 * 40


def testPrepareRunOneRowPerExposure(makeAgRunTables):
    run = prepareRun(makeAgRunTables())

    assert len(run.frames) == N_EXPOSURES
    assert run.frames.taken_at.is_monotonic_increasing
    assert (run.frames.population == "science").all()
    assert list(run.plottedNights) == [pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02")]
    assert (run.frames.n_good == 6 * 6).all()  # 3 good matches on each half of six cameras
    assert (run.frames.n_cams == 6).all()
    # sigma_pix is the radius of the stars' variance, their sigma; an AG spot's FWHM is 2.05 times it.
    perPix = 2.05 * 13 / 94.7
    assert run.cameras.fwhm.round(3).isin([round(s * perPix, 3) for s in (2.0, 2.2)]).all()
    # Negative controls: sigma_pix read as the radius of half the variance gives a FWHM sqrt(2) larger; a
    # Gaussian of that sigma has one 15% larger.
    for wrong in (np.sqrt(2) * perPix, 2.3548 * 13 / 94.7):
        assert not run.cameras.fwhm.round(3).isin([round(s * wrong, 3) for s in (2.0, 2.2)]).any()


def testCameraStates(makeAgRunTables):
    """A camera that detects nothing is DARK, unless its design had too few bright stars to expect any."""
    tables = makeAgRunTables(deadFrom={0: 100, 2: 100}, designStars={2: 2})
    # In one exposure nothing is matched anywhere: the sky wasn't usable.
    blank = tables["exposures"].agc_exposure_id.iloc[10]
    tables["cameras"].loc[tables["cameras"].agc_exposure_id == blank, "n_good"] = 0
    # In another, AG4 detects but matches nothing.
    unmatched = tables["exposures"].agc_exposure_id.iloc[20]
    rows = (tables["cameras"].agc_exposure_id == unmatched) & (tables["cameras"].agc_camera_id == 3)
    tables["cameras"].loc[rows, "n_good"] = 0
    frames = prepareRun(tables).frames.set_index("agc_exposure_id")

    assert (frames.loc[blank, [f"state{cid}" for cid in range(6)]] == CameraState.UNUSABLE).all()
    assert frames.loc[unmatched, "state3"] == CameraState.UNMATCHED
    assert (frames.state0.iloc[100:] == CameraState.DARK).all()
    assert (frames.state0.iloc[:100].drop(blank) == CameraState.OK).all()
    # Negative control: AG3 is just as dead, but its design had 2 bright stars, below minDesignStars.
    assert (frames.state2.iloc[100:] == CameraState.FEW_STARS).all()


def testFindDropoutsDeadCamera(makeAgRunTables):
    run = prepareRun(makeAgRunTables(deadFrom={0: 100, 2: 100}, designStars={2: 2}))
    dropouts = findDropouts(run)

    assert list(dropouts.agc_camera_id) == [0]  # AG3, with too few designed stars, isn't a dropout
    dropout = dropouts.iloc[0]
    assert dropout.n_frames == N_EXPOSURES - 100
    assert dropout.dark and dropout.to_end
    assert dropout.start == run.frames.taken_at.iloc[100]
    assert dropout.last_good == run.frames.taken_at.iloc[99]


def testFindDropoutsMinimumLength(makeAgRunTables):
    """A stretch shorter than minDropout isn't a dropout; one as long is."""
    tables = makeAgRunTables()
    exposures = tables["exposures"].agc_exposure_id
    cameras = tables["cameras"]
    for first, length in ((40, 29), (200, 30)):
        gap = exposures.iloc[first : first + length]
        cameras.loc[cameras.agc_exposure_id.isin(gap) & (cameras.agc_camera_id == 4), "n_good"] = 0
    dropouts = findDropouts(prepareRun(tables, RunConfig(minDropout=30)))

    assert list(dropouts.n_frames) == [30]
    assert not dropouts.dark.iloc[0]  # it still detected spots
    assert not dropouts.to_end.iloc[0]


def testJitterIsOnTheSky(makeAgRunTables):
    """RA jitter is on the sky: guide_delta_ra is a coordinate offset, ΔRA, larger by 1/cos(Dec)."""
    dec = 60.0
    run = prepareRun(makeAgRunTables(dec=dec))
    jitter = visitGuideErrors(run)

    assert len(jitter) == 8
    assert (jitter.n_frames == 40).all()
    # The synthetic noise is 0.05" on the sky in both axes, so both jitters are about 0.05".
    assert np.allclose(jitter.ra.median(), 0.05, rtol=0.2)
    assert np.allclose(jitter.dec.median(), 0.05, rtol=0.2)
    # Negative control: the coordinate offsets' jitter is twice as large at Dec 60°.
    frames = run.frames
    coordinate = frames.groupby("pfs_visit_id").guide_delta_ra.apply(
        lambda x: np.std(np.diff(x)) / np.sqrt(2)
    )
    assert not np.allclose(coordinate.median(), 0.05, rtol=0.2)
    assert np.allclose(onSkyRa(frames), frames.guide_delta_ra * 0.5)


def testGuideErrorIsOfTheFramesWithTheShuttersOpen(makeAgRunTables):
    """A visit's guide error is measured from the accepted offsets of the AG exposures taken with the shutters open."""
    # Each visit's shutters close after its 6th AG exposure; the pointing is then 1″ off, as after a move.
    tables = makeAgRunTables(nNights=1, nVisits=2, nExp=12, openFrames=6)
    exposures = tables["exposures"]
    afterClose = exposures.groupby("pfs_visit_id").cumcount() >= 6
    exposures.loc[afterClose, "guide_delta_dec"] = 1.0
    exposures.loc[2, ["mask", "guide_delta_dec"]] = (1, -100.0)  # rejected, with the shutters open: left out
    run = prepareRun(tables)
    errors = visitGuideErrors(run).set_index("pfs_visit_id")

    assert run.frames.shutter.sum() == 2 * 6
    assert errors.n_frames.to_dict() == {150000: 5, 150001: 6}
    assert (errors.dec_size < 0.15).all() and (errors.dec < 0.15).all()
    # Negative control: measured over every frame, the 1″ after the shutters closed would be the visit's error.
    every = prepareRun(makeAgRunTables(nNights=1, nVisits=2, nExp=12))
    assert visitGuideErrors(every).n_frames.eq(12).all()
    assert exposures[exposures.pfs_visit_id == 150001].guide_delta_dec.pow(2).mean() ** 0.5 > 0.5


def testAVisitNeedsEnoughFramesWithTheShuttersOpen(makeAgRunTables):
    tables = makeAgRunTables(nNights=1, nVisits=2, nExp=12, openFrames=4)

    assert visitGuideErrors(prepareRun(tables)).empty
    assert len(visitGuideErrors(prepareRun(tables, RunConfig(minOpenFrames=4)))) == 2
    # A read with no shutter times cannot say which frames those were: no visit is measured.
    older = makeAgRunTables()
    older["visits"] = older["visits"].drop(columns=["shutter_open", "shutter_close"])
    run = prepareRun(older)
    assert not run.frames.shutter.any() and visitGuideErrors(run).empty
    assert runSummary(run, visitGuideErrors(run))["measuredVisits"] == 0
    # Negative control: with them, every visit of the same run is.
    assert len(visitGuideErrors(prepareRun(makeAgRunTables()))) == 8


def testJitterOfAShortVisitReadsAsALongOnes(makeAgRunTables):
    """The median visit's jitter is the noise put in, for visits of six frames as for long ones.

    The jitter is the sample standard deviation (n - 1) of a visit's consecutive differences. With n, the
    median visit of six frames reads 0.88 of the same noise over a long visit, in a simulation of pure
    noise and in 6-frame pieces of Run 31's long visits alike (0.87); with n - 1, 0.98 and 0.97.
    """
    short = visitGuideErrors(prepareRun(makeAgRunTables(nNights=1, nVisits=400, nExp=6)))
    long = visitGuideErrors(prepareRun(makeAgRunTables(nNights=1, nVisits=8, nExp=300)))

    assert len(short) == 400 and len(long) == 8
    assert long.dec.median() == pytest.approx(0.05, rel=0.06)
    assert short.dec.median() == pytest.approx(0.05, rel=0.06)
    # Negative control: dividing by n would put the six-frame visits' median well under the noise.
    assert short.dec.median() * np.sqrt(4 / 5) != pytest.approx(0.05, rel=0.06)


def testGroupSummaryFocus(makeAgRunTables):
    """Each group alone reads focus from its own cameras: AG1/3/5 +10 µm, AG2/4/6 -10 µm from the median of six."""
    summary = groupSummary(prepareRun(makeAgRunTables()))

    assert list(summary.columns) == list(CAMERA_GROUPS)
    focus = summary.loc["focus from this group alone minus dZ (µm)"]
    assert np.allclose(focus, [10.0, -10.0], atol=1.0)
    # Negative control: the groups don't read the same.
    assert not np.isclose(focus["AG1/3/5"], focus["AG2/4/6"], atol=1.0)
    assert (summary.loc["matched (% of judged exposures)"] == 100).all()
    assert summary.loc["FWHM, RIGHT half (″)"].gt(summary.loc["FWHM, without RIGHT (″)"]).all()


def testGroupSummaryDeadGroup(makeAgRunTables):
    summary = groupSummary(prepareRun(makeAgRunTables(deadFrom={0: 160, 2: 160, 4: 160})))
    matched = summary.loc["matched (% of judged exposures)"]

    assert matched["AG1/3/5"] == pytest.approx(50.0)
    assert matched["AG2/4/6"] == 100.0


def testFindFlags(makeAgRunTables):
    tables = makeAgRunTables(deadFrom={4: 200})
    exposures = tables["exposures"]
    exposures.loc[50:51, "mask"] = 1  # rejected
    exposures.loc[60, "mask"] = np.nan  # not written
    # Three of visit 150002's offsets were accepted from 1 star on one camera.
    lone = exposures.agc_exposure_id.iloc[100:103]
    cameras = tables["cameras"]
    cameras.loc[cameras.agc_exposure_id.isin(lone), "n_good"] = 0
    oneHalf = (cameras.agc_camera_id == 0) & (cameras.right_half == 0)
    cameras.loc[cameras.agc_exposure_id.isin(lone) & oneHalf, "n_good"] = 1
    run = prepareRun(tables)
    jitter = visitGuideErrors(run)
    flags = findFlags(run, jitter, findDropouts(run))
    text = "\n".join(flag.text for flag in flags)

    assert "AG5: no detections at all in 120 consecutive AG exposures" in text
    assert "2 guide frames rejected" in text and "150001" in text
    assert "3 guide offsets accepted from fewer than 3 matched stars" in text and "visit 150002 (3)" in text
    assert "1 of 320 AG exposures have no guide offset" in text
    assert all(flag.kind in ("camera", "guiding", "image") for flag in flags)
    # Negative control: a clean run raises no flag.
    clean = prepareRun(makeAgRunTables())
    assert findFlags(clean, visitGuideErrors(clean), findDropouts(clean)) == []


def testIsRejectedOffset():
    tooFew = int(GuideOffsetFlag.INVALID_OFFSET | GuideOffsetFlag.TOO_FEW_STARS)
    mask = pd.Series([0, 1, tooFew, np.nan])
    assert isRejectedOffset(mask).tolist() == [False, True, True, False]
    assert isRejectedOffset(mask, GuideOffsetFlag.TOO_FEW_STARS).tolist() == [False, False, True, False]
    # Negative control: testing for the value 1 would count a TOO_FEW_STARS offset (3) as used.
    assert (mask == 1).tolist() == [False, True, False, False]


def testFindFlagsCountsTooFewStars(makeAgRunTables):
    tables = makeAgRunTables()
    exposures = tables["exposures"]
    exposures.loc[50:52, "mask"] = int(GuideOffsetFlag.INVALID_OFFSET | GuideOffsetFlag.TOO_FEW_STARS)
    exposures.loc[53, "mask"] = int(GuideOffsetFlag.INVALID_OFFSET)
    run = prepareRun(tables)
    text = "\n".join(flag.text for flag in findFlags(run, visitGuideErrors(run), findDropouts(run)))
    assert (
        "4 guide frames rejected** (INVALID_OFFSET: not sent to the telescope, 3 of them TOO_FEW_STARS)"
        in text
    )
    # Negative control: out of range alone names no TOO_FEW_STARS.
    exposures.loc[50:52, "mask"] = int(GuideOffsetFlag.INVALID_OFFSET)
    run = prepareRun(tables)
    text = "\n".join(flag.text for flag in findFlags(run, visitGuideErrors(run), findDropouts(run)))
    assert "4 guide frames rejected** (INVALID_OFFSET: not sent to the telescope)" in text


def testFindFlagsJitterAndFwhm(makeAgRunTables):
    tables = makeAgRunTables()
    exposures = tables["exposures"]
    exposures.loc[exposures.pfs_visit_id == 150002, "guide_delta_dec"] *= 10
    tables["cameras"].loc[tables["cameras"].agc_camera_id == 5, "sigma_pix"] *= 1.5
    run = prepareRun(tables)
    text = "\n".join(flag.text for flag in findFlags(run, visitGuideErrors(run), findDropouts(run)))

    assert "1 of 8 science visits" in text and "150002" in text
    assert "AG6: FWHM +" in text
    assert "AG5: FWHM" not in text


def testDescribePeriodNamesNights(makeAgRunTables):
    """Nights are named by their evening, though the last exposures are taken after midnight."""
    tables = makeAgRunTables()
    tables["exposures"].loc[len(tables["exposures"]) - 1, "taken_at"] = pd.Timestamp("2026-09-03 04:00")
    text = describePeriod(prepareRun(tables))

    assert text.startswith("nights 2026-09-01 to 2026-09-02,")
    # Negative control: the last exposure's calendar date is the next day.
    assert "2026-09-03" not in text


def testNightPosition():
    nights = pd.DatetimeIndex(["2026-09-01", "2026-09-03"])
    times = pd.Series(
        pd.to_datetime(["2026-09-01 18:00", "2026-09-02 00:00", "2026-09-04 07:00", "2026-09-02 20:00"])
    )
    x = nightPosition(times, nights)

    assert np.allclose(x[:3], [0.05, 0.5, 1.95])  # 06:00 or later is put at the night's end
    assert np.isnan(x.iloc[3])  # a night that isn't plotted


def _reject(tables, exposures, cid, cells, fraction=1.0):
    """Reject a camera's paired detections in some cells of the grid, in some exposures.

    ``fraction`` of each cell's 2 pairs are rejected (1 or 0.5), at the cell's centre.
    """
    grid = tables["grid"]
    rows = grid.agc_exposure_id.isin(exposures) & (grid.agc_camera_id == cid)
    rows &= pd.Series(list(zip(grid.row_cell, grid.column_cell, strict=True)), index=grid.index).isin(cells)
    grid.loc[rows, "n_good"] = round(2 * (1 - fraction))
    grid.loc[rows, "x_rejected_pix"] = grid.loc[rows, "x_good_pix"]
    grid.loc[rows, "y_rejected_pix"] = grid.loc[rows, "y_good_pix"]
    if fraction == 1.0:
        grid.loc[rows, ["x_good_pix", "y_good_pix"]] = np.nan


TOP_HALF = [(row, column) for row in (2, 3) for column in range(4)]
CORNER = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # +x, -y: a corner, at an angle to the rows and the halves
ALL_CELLS = [(row, column) for row in range(4) for column in range(4)]


def testFindRejectionClusters(makeAgRunTables):
    """Rejected matches on one side of a detector are a cluster, whatever the side; spread out, they're not."""
    tables = makeAgRunTables()
    exposures = tables["exposures"].agc_exposure_id
    _reject(tables, exposures.iloc[:5], 2, TOP_HALF)  # AG3's top half, as in Run 29
    _reject(tables, exposures.iloc[10:13], 3, CORNER)  # AG4's corner
    _reject(tables, exposures.iloc[20:22], 1, TOP_HALF)  # AG2 and AG5 together
    _reject(tables, exposures.iloc[20:22], 4, CORNER)
    # Negative control: AG6 loses half its pairs everywhere, which is no cluster.
    _reject(tables, exposures.iloc[30:33], 5, ALL_CELLS, fraction=0.5)
    clusters = findRejectionClusters(prepareRun(tables))

    assert clusters.groupby("agc_camera_id").size().to_dict() == {1: 2, 2: 5, 3: 3, 4: 2}
    ag3, ag4 = (clusters[clusters.agc_camera_id == cid].iloc[0] for cid in (2, 3))
    assert np.isclose(ag3.angle_deg, 90, atol=1)  # toward +y
    assert 290 < ag4.angle_deg < 345  # toward +x, -y
    assert clusters[clusters.agc_camera_id.isin([2, 3])].alone.all()
    assert not clusters[clusters.agc_camera_id.isin([1, 4])].alone.any()  # two cameras at once


def testClusterFlags(makeAgRunTables):
    """A camera is flagged for clusters on it alone, far more often than the others."""
    tables = makeAgRunTables()
    exposures = tables["exposures"].agc_exposure_id
    _reject(tables, exposures.iloc[:12], 2, TOP_HALF)
    for cid in range(6):  # 12 more exposures in which every camera has a cluster at once
        _reject(tables, exposures.iloc[100:112], cid, CORNER)
    run = prepareRun(tables)
    flags = findFlags(
        run, visitGuideErrors(run), findDropouts(run), findRejectionClusters(run), fitModelErrors(run)
    )
    clustered = [flag for flag in flags if "cluster" in flag.text]

    assert [(flag.agc_camera_id, flag.severe) for flag in clustered] == [(2, True)]
    assert "on it alone, in 12 AG exposures" in clustered[0].text and "toward 90°" in clustered[0].text
    assert "consistency 1.00" in clustered[0].text
    # Negative control: the clusters on every camera at once flag none of them.
    assert not any(flag.agc_camera_id != 2 for flag in clustered)


def _inTheFocalPlane(cid, onDetector):
    """Turn a model error or a step given on a detector's axes into the focal plane's, as the opdb holds it."""
    return (np.linalg.inv(_toDetectorAxes(cid)) @ np.asarray(onDetector, dtype=float)).tolist()


def testFitModelErrors(makeAgRunTables):
    """A scale along a detector axis and a step between its halves are each found, at their size, as themselves."""
    alongY = [[0.0, 0.0], [0.0, 0.013]]  # 13 µm per 1000 rows: 6.5 µm at 500 px
    alongX = [[-0.010, 0.0], [0.0, 0.0]]  # stars 5 µm closer together at 500 px
    rotation = [[0.0, -0.01], [0.01, 0.0]]
    tables = makeAgRunTables(
        modelError={
            2: _inTheFocalPlane(2, alongY),
            3: _inTheFocalPlane(3, alongX),
            5: _inTheFocalPlane(5, rotation),
        },
        halfStep={4: _inTheFocalPlane(4, [5.0, 0.0])},
    )
    run = prepareRun(tables)
    errors = fitModelErrors(run)
    parts = summariseModelErrors(errors)

    assert list(parts.columns) == list(MODEL_PARTS)
    assert np.allclose(parts.loc[2], [0.0, 6.5, 0.0], atol=0.3)
    assert np.isclose(parts.tangential_um[3], -5.0, atol=0.3) and np.isclose(parts.step_um[4], 5.0, atol=0.3)
    # Negative control: a slope along x is not read as a step, nor a step as a slope along x.
    assert abs(parts.step_um[3]) < 0.3 and abs(parts.tangential_um[4]) < 0.3
    assert not np.isclose(parts.step_um[3], 5.0, atol=2) and not np.isclose(
        parts.tangential_um[4], -5.0, atol=2
    )
    # Negative control: a camera turned on its axis moves x with y and y with x; it has no scale and no step.
    assert np.allclose(parts.loc[5], 0.0, atol=0.3)
    # Negative control: the 20 µm pointing error moves every camera's stars together, and is no model error.
    assert np.allclose(parts.loc[[0, 1]], 0.0, atol=0.3)
    flagged = {
        flag.agc_camera_id
        for flag in findFlags(run, visitGuideErrors(run), findDropouts(run), None, errors)
        if flag.kind == "camera" and not flag.severe
    }
    assert {2, 3, 4} <= flagged and not {0, 1, 5} & flagged


def testModelErrorOnOneHalfHasNoStep(makeAgRunTables):
    """With matches on one half only, a night's slopes are still fitted and its step is left out."""
    tables = makeAgRunTables(modelError={2: _inTheFocalPlane(2, [[0.0, 0.0], [0.0, 0.013]])})
    grid = tables["grid"]
    tables["grid"] = grid[(grid.agc_camera_id != 2) | (grid.column_cell >= 2)]
    parts = summariseModelErrors(fitModelErrors(prepareRun(tables)))

    assert np.isnan(parts.step_um[2]) and np.isclose(parts.radial_um[2], 6.5, atol=0.3)
    # Negative control: a camera with matches on both halves has a step, here none to speak of.
    assert abs(parts.step_um[3]) < 0.3


def testCameraSummary(makeAgRunTables):
    tables = makeAgRunTables(
        deadFrom={4: 160}, modelError={2: _inTheFocalPlane(2, [[0.0, 0.0], [0.0, 0.013]])}
    )
    _reject(tables, tables["exposures"].agc_exposure_id.iloc[:7], 2, TOP_HALF)
    run = prepareRun(tables)
    summary = cameraSummary(run, findRejectionClusters(run), fitModelErrors(run))

    assert list(summary.columns) == [f"AG{n}" for n in range(1, 7)]
    assert summary.loc["matched (% of judged exposures)", "AG5"] == pytest.approx(50.0)
    assert summary.loc["matched (% of judged exposures)", "AG1"] == 100.0
    assert summary.loc["good matches per exposure", "AG1"] == 6
    assert summary.loc["exposures with rejected matches clustered on it alone", "AG3"] == 7
    assert summary.loc["their direction (°)", "AG3"] == pytest.approx(90, abs=1)
    assert np.isclose(summary.loc["position error along y at 500 px (µm)", "AG3"], 6.5, atol=0.3)
    assert abs(summary.loc["position error along x at 500 px (µm)", "AG3"]) < 0.3
    assert summary.loc["focus minus the cameras' median (µm)", "AG1"] == pytest.approx(10.0, abs=1.0)


def testDetectorCells(makeAgRunTables):
    """A cell holds the share of its pairs rejected and the residual left there, along the detector's axes."""
    rowScale = [[0.0, 0.0], [0.0, 0.013]]  # in the focal plane: 13 µm in y per 1000 rows
    tables = makeAgRunTables(modelError={2: rowScale})
    _reject(tables, tables["exposures"].agc_exposure_id.iloc[:80], 2, CORNER)
    cells = detectorCells(prepareRun(tables)).set_index(["agc_camera_id", "row_cell", "column_cell"])
    ag3 = cells.loc[2]

    assert set(ag3[ag3.rejected > 20].index) == set(CORNER)
    assert (ag3.drop(index=CORNER).rejected == 0).all()
    top, bottom = (ag3.loc[row, ["dx_um", "dy_um"]].mean().to_numpy() for row in (3, 0))
    onDetector = _toDetectorAxes(2) @ [0.0, 0.013 * 3 * 260]  # the rows are 780 px apart
    assert np.allclose(top - bottom, onDetector, atol=0.5)
    # Negative control: left in the focal plane's axes, the residual would point somewhere else on the detector.
    assert not np.allclose(top - bottom, [0.0, 0.013 * 3 * 260], atol=2)
    # Negative control: a camera with nothing wrong has no residual anywhere; the pointing is taken off.
    assert (np.hypot(cells.loc[4].dx_um, cells.loc[4].dy_um) < 0.5).all()


def testResidualsTurnToTheDetectorAxes(realAgcData):
    """Between two spots of one camera, the opdb's focal-plane positions turn into their pixel separation."""
    data = realAgcData("raster").dropna(subset=["centroid_x_pix", "agc_center_x_mm"])
    checked = 0
    for cid, spots in data.groupby("agc_camera_id"):
        first = spots.iloc[0]
        last = spots.iloc[
            np.hypot(
                spots.centroid_x_pix - first.centroid_x_pix, spots.centroid_y_pix - first.centroid_y_pix
            ).argmax()
        ]
        pixels = np.array(
            [last.centroid_x_pix - first.centroid_x_pix, last.centroid_y_pix - first.centroid_y_pix]
        )
        # The fixture is in hardware coordinates: the opdb's y is the other way up.
        hardware = 1000 * np.array(
            [last.agc_center_x_mm - first.agc_center_x_mm, last.agc_center_y_mm - first.agc_center_y_mm]
        )
        # To 40 µm in a few mm: the camera models have been refitted (by under a degree) since these were taken.
        assert np.allclose(_toDetectorAxes(int(cid)) @ (hardware * [1, -1]), 13 * pixels, atol=40.0)
        # Negative control: taken as they are in hardware coordinates, they don't.
        assert not np.allclose(_toDetectorAxes(int(cid)) @ hardware, 13 * pixels, atol=200.0)
        checked += 1
    assert checked >= 3


def testModelErrorsNeedMatchesSpreadAcrossTheDetector(makeAgRunTables):
    """A camera whose good matches are all in one cell can't show a model error, and isn't fitted."""
    tables = makeAgRunTables(modelError={2: [[0.0, 0.0], [0.0, 0.013]]})
    grid = tables["grid"]
    elsewhere = (grid.agc_camera_id == 4) & ~((grid.row_cell == 1) & (grid.column_cell == 2))
    grid.loc[elsewhere, ["n_good", "n_paired"]] = 0
    grid.loc[elsewhere, ["x_good_pix", "y_good_pix", "dx_um", "dy_um"]] = np.nan
    grid.loc[grid.agc_camera_id == 4, ["x_good_pix", "y_good_pix"]] += [0.37, 0.61]  # rounding noise, if any
    fitted = set(fitModelErrors(prepareRun(tables)).agc_camera_id)

    assert 4 not in fitted
    # Negative control: the other cameras, spread over all 16 cells, are fitted.
    assert fitted == {0, 1, 2, 3, 5}


def testCameraChecksUsePlottedNightsOnly(makeAgRunTables):
    """Clusters and model errors count only the nights the figures show."""
    tables = makeAgRunTables(modelError={2: [[0.0, 0.0], [0.0, 0.013]]})
    _reject(tables, tables["exposures"].agc_exposure_id.iloc[:5], 2, TOP_HALF)
    hidden = prepareRun(tables, RunConfig(minFramesNight=1000))  # no night has enough frames to be plotted

    assert findRejectionClusters(hidden).empty and fitModelErrors(hidden).empty
    # Negative control: plotted, the same run has both.
    shown = prepareRun(tables)
    assert len(findRejectionClusters(shown)) == 5 and len(fitModelErrors(shown))


def _flags(run):
    return findFlags(run, visitGuideErrors(run), findDropouts(run))


def testCameraNotInUse(makeAgRunTables):
    """A camera with no detections in the whole run is named once and left out of every comparison."""
    tables = makeAgRunTables(deadFrom={1: 0})
    tables["exposures"]["guide_delta_z2"] = np.nan  # as the guider writes for a camera with no stars
    run = prepareRun(tables)
    jitter = visitGuideErrors(run)
    flags = findFlags(run, jitter, findDropouts(run))

    assert run.inUse == (0, 2, 3, 4, 5)
    # One flag, about that camera, and not a fault.
    assert [(flag.kind, flag.severe, flag.agc_camera_id) for flag in flags] == [("camera", False, 1)]
    assert findDropouts(run).empty
    assert jitter.full.all()
    # The focus is still compared, among the five: AG1/3/5 +10 µm and AG4/6 -10 µm about a median of +10.
    assert list(cameraFocus(run).columns) == [0, 2, 3, 4, 5]
    groups = groupSummary(run)
    assert np.allclose(groups.loc["focus minus the cameras' median (µm)"], [0.0, -20.0], atol=1.5)
    assert (groups.loc["matched (% of judged exposures)"] == 100).all()
    summary = cameraSummary(run, findRejectionClusters(run), fitModelErrors(run))
    assert summary["AG2"].isna().all()
    assert summary.drop(columns="AG2").loc["matched (% of judged exposures)"].eq(100).all()
    # Negative control: a camera that stops during the run was in use, and its loss is a fault.
    lost = prepareRun(makeAgRunTables(deadFrom={1: 100}))
    assert lost.inUse == tuple(range(6))
    (dropout,) = findDropouts(lost).itertuples()
    assert (dropout.agc_camera_id, dropout.n_frames, dropout.dark) == (1, 220, True)
    assert any(flag.severe and flag.agc_camera_id == 1 for flag in _flags(lost))
    assert not visitGuideErrors(lost).full.all()


def testDisabledCamera(makeAgRunTables):
    """A camera the guider left out of its fit is DISABLED: its matches are not good ones, and it is flagged."""
    run = prepareRun(makeAgRunTables(disabledFrom={3: 200}), RunConfig(longDropout=100))
    frames = run.frames

    assert (frames.state3.iloc[200:] == CameraState.DISABLED).all()
    assert (frames.n_good3.iloc[200:] == 0).all() and (frames.n_spots3.iloc[200:] == 8).all()
    assert (frames.n_cams.iloc[200:] == 5).all()
    (dropout,) = findDropouts(run).itertuples()
    assert (dropout.agc_camera_id, dropout.n_frames, dropout.disabled, dropout.dark) == (3, 120, True, False)
    assert any(flag.severe and flag.agc_camera_id == 3 for flag in _flags(run))
    # Negative control: before then it was matched like the others.
    assert (frames.state3.iloc[:200] == CameraState.OK).all()


def testShortDisabledStretchesAreCountedApart(makeAgRunTables):
    """A short stretch in which a camera was disabled is not counted with those in which one matched nothing."""
    run = prepareRun(makeAgRunTables(disabledFrom={3: 200}))  # 120 exposures: under longDropout
    dropouts = findDropouts(run)
    unmatched = dropouts.assign(agc_camera_id=4, disabled=False)

    def notes(found):
        return [flag for flag in findFlags(run, visitGuideErrors(run), found) if not flag.severe]

    assert len(notes(pd.concat([dropouts, unmatched]))) == len(notes(dropouts)) + 1
    # Negative control: two cameras' stretches of one kind share a line.
    assert len(notes(pd.concat([dropouts, dropouts.assign(agc_camera_id=4)]))) == len(notes(dropouts))


def testSizeAndJitter(makeAgRunTables):
    """A steady pointing error shows in the size of a visit's guide offsets, not in their jitter."""
    tables = makeAgRunTables()
    exposures = tables["exposures"]
    biased = exposures.pfs_visit_id == 150002
    exposures.loc[biased, "guide_delta_dec"] += 0.3
    jitter = visitGuideErrors(prepareRun(tables)).set_index("pfs_visit_id")

    assert jitter.dec_size[150002] == pytest.approx(np.hypot(0.3, 0.05), rel=0.1)
    assert jitter.dec[150002] == pytest.approx(0.05, rel=0.35)
    assert jitter.dec_size.drop(150002).between(0.03, 0.07).all()
    assert (jitter.residual_um == 12.0).all()
    # Negative control: the jitter doesn't see the 0.3″.
    assert not np.isclose(jitter.dec[150002], jitter.dec_size[150002], rtol=0.5)


def testStateShares(makeAgRunTables):
    """Each camera's states are counted per hour of each night, among the exposures with a star somewhere."""
    tables = makeAgRunTables(nNights=1, nVisits=4, nExp=90, deadFrom={0: 330}, disabledFrom={4: 0})
    shares = stateShares(prepareRun(tables))
    ag1 = shares[shares.agc_camera_id == 0].set_index("slot")

    # 360 exposures 10 s apart from 19:00: slots 1 and 2 (19:00-21:00); AG1 dies 55 min in.
    assert ag1.n.to_dict() == {1: 359, 2: 1}
    assert (ag1.matched[1], ag1.dark[1], ag1.dark[2]) == (330, 29, 1)
    assert (ag1[["matched", "unmatched", "dark", "disabled", "few_stars"]].sum(axis=1) == ag1.n).all()
    ag5 = shares[shares.agc_camera_id == 4]
    assert (ag5.disabled == ag5.n).all()
    # Negative control: a camera that kept matching has no other state.
    ag2 = shares[shares.agc_camera_id == 1]
    assert (ag2.matched == ag2.n).all()


def testMagnitudeSummary(makeAgRunTables):
    """The share of designed guide stars matched, by camera and magnitude."""
    summary = magnitudeSummary(prepareRun(makeAgRunTables(deadFrom={2: 160})))
    fraction = summary.set_index(["agc_camera_id", "magnitude"]).fraction

    assert fraction[0, 14.0] == 1.0 and fraction[0, 19.0] == 0.5
    # AG3 detected nothing in the second half of the run.
    assert fraction[2, 14.0] == 0.5 and fraction[2, 19.0] == 0.25
    # Negative control: the faint stars aren't matched as often as the bright ones.
    assert not np.isclose(fraction[0, 19.0], fraction[0, 14.0])


def testDetectionFlagShares(makeAgRunTables):
    tables = makeAgRunTables()
    cameras = tables["cameras"]
    cameras.loc[cameras.agc_camera_id == 5, "n_saturated"] = 2
    shares = detectionFlagShares(prepareRun(tables))

    assert list(shares.columns) == [
        "night",
        "agc_camera_id",
        "EDGE",
        *list(DETECTION_FLAG_COLUMNS.values())[1:],
        "n_spots",
    ]
    assert len(shares) == 2 * 6 and (shares.EDGE == 25.0).all()
    assert (shares.SATURATED[shares.agc_camera_id == 5] == 50.0).all()
    # Negative control: the other cameras have no saturated detection.
    assert (shares.SATURATED[shares.agc_camera_id != 5] == 0.0).all()


def testRunSummary(makeAgRunTables):
    tables = makeAgRunTables()
    exposures = tables["exposures"]
    exposures.loc[50, "mask"] = 1
    exposures.loc[60, "mask"] = np.nan
    run = prepareRun(tables)
    summary = runSummary(run, visitGuideErrors(run))

    assert (summary["nights"], summary["exposures"], summary["scienceVisits"]) == (2, N_EXPOSURES, 8)
    assert summary["camerasInUse"] == [f"AG{n}" for n in range(1, 7)]
    assert summary["matchedStars"]["perExposure"] == 36 and summary["matchedStars"]["perCamera"]["AG4"] == 6
    assert summary["matchedStars"]["underMinimumPct"] == 0.0
    assert summary["residualUm"] == {"median": 12.0, "p95": 12.0}
    assert summary["guideLoop"] == {"rejected": 1, "noOffset": 1}
    # On the sky the two axes go together, in quadrature: a visit's 0.05″ in each is 0.07″.
    for axis, expected in (("ra", 0.05), ("dec", 0.05), ("inr", 1.0), ("sky", 0.05 * np.sqrt(2))):
        error = summary["guideError"][axis]
        assert error["size"] == pytest.approx(expected, rel=0.3)
        assert error["jitter"] == pytest.approx(expected, rel=0.3)
        assert error["size95"] >= error["size"] and error["jitter95"] >= error["jitter"]


def testPrepareRunFromAnOlderRead(makeAgRunTables):
    """A read without the newer columns and the magnitudes table still prepares, with those results empty."""
    tables = makeAgRunTables()
    tables["cameras"] = tables["cameras"].drop(columns=["residual_um", "n_disabled", *DETECTION_FLAG_COLUMNS])
    del tables["magnitudes"]
    run = prepareRun(tables)

    assert magnitudeSummary(run).empty and detectionFlagShares(run).empty
    assert run.frames.residual_um.isna().all()
    assert (run.frames[[f"state{cid}" for cid in range(6)]] == CameraState.OK).all().all()
    summary = cameraSummary(run, findRejectionClusters(run), fitModelErrors(run))
    assert summary.loc["detections flagged EDGE (%)"].isna().all()
    # Negative control: with them, the same run has both.
    full = prepareRun(makeAgRunTables())
    assert not magnitudeSummary(full).empty and not detectionFlagShares(full).empty


def testFewStarFlagSaysWhenAndCountsFocusSweepsApart(makeAgRunTables):
    """Offsets accepted from too few stars are dated, and a focus sweep's are not the guide loop's."""
    tables = makeAgRunTables()
    exposures, cameras, visits = tables["exposures"], tables["cameras"], tables["visits"]
    visits.loc[visits.pfs_visit_id == 150012, "sequence_type"] = "agFocusSweep"
    # One star in three exposures of a science visit on the first night, and in two of the focus sweep.
    guiding = exposures.agc_exposure_id[exposures.pfs_visit_id == 150001].iloc[10:13]
    sweep = exposures.agc_exposure_id[exposures.pfs_visit_id == 150012].iloc[5:7]
    few = cameras.agc_exposure_id.isin([*guiding, *sweep])
    cameras.loc[few, "n_good"] = 0
    cameras.loc[few & (cameras.agc_camera_id == 0) & (cameras.right_half == 0), "n_good"] = 1
    last = exposures.taken_at[exposures.agc_exposure_id == guiding.iloc[-1]].iloc[0]
    loop, focusSweep = (flag.text for flag in _flags(prepareRun(tables)) if flag.kind == "guiding")

    # The guide loop's three: their visit, and when the last was.
    assert "visit 150001 (3)" in loop and f"{last:%Y-%m-%d %H:%M}" in loop
    assert "visit 150012 (2)" in focusSweep
    # Negative control: the focus sweep's are not counted with the guide loop's.
    assert "150012" not in loop and "150001" not in focusSweep


def testVisitMatchFraction(makeAgRunTables):
    """The share of a visit's designed guide stars that were matched, on the cameras in use."""
    # AG3 stops half-way through the run; AG2 never detects anything, so its guide stars were not available.
    run = prepareRun(makeAgRunTables(deadFrom={1: 0, 2: 160}))
    fractions = visitMatchFraction(run).set_index("pfs_visit_id")

    assert len(fractions) == 8 and run.inUse == (0, 2, 3, 4, 5)
    # Each camera matches its two bright stars and one of its two faint ones: three in four.
    assert fractions.fraction[150000] == pytest.approx(0.75)
    # With one of the five cameras gone, four fifths of that.
    assert fractions.fraction[150013] == pytest.approx(0.75 * 4 / 5)
    summary = runSummary(run, visitGuideErrors(run))["matchedFraction"]
    assert summary["visits"] == 8 and summary["median"] == pytest.approx((0.75 + 0.6) / 2)
    assert summary["p5"] < summary["median"] < summary["p95"]
    # Negative control: counting AG2's stars as available would lower every visit's share.
    everyCamera = run.magnitudes.groupby("pfs_visit_id")[["n_expected", "n_good"]].sum()
    assert not np.isclose(everyCamera.n_good[150000] / everyCamera.n_expected[150000], 0.75)
    # A read without the magnitudes has none to give.
    tables = makeAgRunTables()
    del tables["magnitudes"]
    older = prepareRun(tables)
    assert visitMatchFraction(older).empty
    assert np.isnan(runSummary(older, visitGuideErrors(older))["matchedFraction"]["median"])
