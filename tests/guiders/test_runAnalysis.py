"""Tests for ``pfs.drp.qa.guiders.runAnalysis``, on synthetic runs."""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders.runAnalysis import (
    CAMERA_GROUPS,
    CameraState,
    RunConfig,
    cameraSummary,
    describePeriod,
    findDropouts,
    findFlags,
    findRejectionClusters,
    fitModelErrors,
    groupSummary,
    nightPosition,
    onSkyRa,
    prepareRun,
    settledJitter,
)

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
    assert run.cameras.fwhm.round(3).isin([round(s * 2.3548 * 0.14, 3) for s in (2.0, 2.2)]).all()


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


def testSettledJitterOnTheSky(makeAgRunTables):
    """RA jitter is on the sky: guide_delta_ra is a coordinate offset, ΔRA, larger by 1/cos(Dec)."""
    dec = 60.0
    run = prepareRun(makeAgRunTables(dec=dec))
    jitter = settledJitter(run)

    assert len(jitter) == 8
    assert (jitter.n_frames == 40 - RunConfig().nSettle).all()
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


def testSettledJitterSkipsUnsettledAndRejected(makeAgRunTables):
    tables = makeAgRunTables(nNights=1, nVisits=1)
    tables["exposures"].loc[:4, "guide_delta_dec"] = 100.0  # acquisition: left out as unsettled
    tables["exposures"].loc[20, ["mask", "guide_delta_dec"]] = (1, -100.0)  # rejected: left out
    jitter = settledJitter(prepareRun(tables))

    assert jitter.n_frames.iloc[0] == 40 - 5 - 1
    assert jitter.dec.iloc[0] < 0.1


def testGroupSummaryFocus(makeAgRunTables):
    """Each group alone reads focus from its own cameras: AG1/3/5 +10 µm, AG2/4/6 -10 µm from the median of six."""
    summary = groupSummary(prepareRun(makeAgRunTables()))

    assert list(summary.columns) == list(CAMERA_GROUPS)
    focus = summary.loc["focus from this group alone minus dZ (µm)"]
    assert np.allclose(focus, [10.0, -10.0], atol=1.0)
    # Negative control: the groups don't read the same.
    assert not np.isclose(focus["AG1/3/5"], focus["AG2/4/6"], atol=1.0)
    assert (summary.loc["matched (% of judged exposures)"] == 100).all()
    assert summary.loc["FWHM, no-glass half (″)"].gt(summary.loc["FWHM, glass half (″)"]).all()


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
    jitter = settledJitter(run)
    flags = findFlags(run, jitter, findDropouts(run))
    text = "\n".join(flag.text for flag in flags)

    assert "AG5: no detections at all in 120 consecutive AG exposures" in text
    assert "2 guide frames rejected" in text and "150001" in text
    assert "3 guide offsets accepted from fewer than 3 matched stars" in text and "visit 150002 (3)" in text
    assert "1 of 320 AG exposures have no guide offset" in text
    assert all(flag.kind in ("camera", "guiding", "image") for flag in flags)
    # Negative control: a clean run raises no flag.
    clean = prepareRun(makeAgRunTables())
    assert findFlags(clean, settledJitter(clean), findDropouts(clean)) == []


def testFindFlagsJitterAndFwhm(makeAgRunTables):
    tables = makeAgRunTables()
    exposures = tables["exposures"]
    exposures.loc[exposures.pfs_visit_id == 150002, "guide_delta_dec"] *= 10
    tables["cameras"].loc[tables["cameras"].agc_camera_id == 5, "sigma_pix"] *= 1.5
    run = prepareRun(tables)
    text = "\n".join(flag.text for flag in findFlags(run, settledJitter(run), findDropouts(run)))

    assert "1 of 8 science visits have settled jitter over 3× the median" in text and "150002" in text
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
        run, settledJitter(run), findDropouts(run), findRejectionClusters(run), fitModelErrors(run)
    )
    clustered = [flag for flag in flags if "cluster" in flag.text]

    assert [(flag.agc_camera_id, flag.severe) for flag in clustered] == [(2, True)]
    assert "on it alone, in 12 AG exposures" in clustered[0].text and "toward 90°" in clustered[0].text
    assert "consistency 1.00" in clustered[0].text
    # Negative control: the clusters on every camera at once flag none of them.
    assert not any(flag.agc_camera_id != 2 for flag in clustered)


def testFitModelErrors(makeAgRunTables):
    """A camera's model error is found at its size and direction; the pointing, common to all, isn't one."""
    rowScale = [[0.0, 0.0], [0.0, 0.013]]  # 13 µm per 1000 rows: 6.5 µm at 500 px, along y
    diagonal = (0.01 * np.outer([1, 0], [np.cos(np.radians(45)), np.sin(np.radians(45))])).tolist()
    run = prepareRun(makeAgRunTables(modelError={2: rowScale, 3: diagonal}))
    errors = fitModelErrors(run).groupby("agc_camera_id")[["error_um", "angle_deg"]].median()

    assert np.isclose(errors.error_um[2], 6.5, atol=0.3) and np.isclose(errors.angle_deg[2], 90, atol=2)
    assert np.isclose(errors.error_um[3], 5.0, atol=0.3) and np.isclose(errors.angle_deg[3], 45, atol=3)
    # Negative control: the 20 µm pointing error moves every camera's stars together, and is no model error.
    assert (errors.drop([2, 3]).error_um < 0.6).all()
    flags = findFlags(run, settledJitter(run), findDropouts(run), None, fitModelErrors(run))
    assert [flag.agc_camera_id for flag in flags if "position model" in flag.text] == [2, 3]


def testCameraSummary(makeAgRunTables):
    tables = makeAgRunTables(deadFrom={4: 160}, modelError={2: [[0.0, 0.0], [0.0, 0.013]]})
    _reject(tables, tables["exposures"].agc_exposure_id.iloc[:7], 2, TOP_HALF)
    run = prepareRun(tables)
    summary = cameraSummary(run, findRejectionClusters(run), fitModelErrors(run))

    assert list(summary.columns) == [f"AG{n}" for n in range(1, 7)]
    assert summary.loc["matched (% of judged exposures)", "AG5"] == pytest.approx(50.0)
    assert summary.loc["matched (% of judged exposures)", "AG1"] == 100.0
    assert summary.loc["good matches per exposure", "AG1"] == 6
    assert summary.loc["exposures with rejected matches clustered on it alone", "AG3"] == 7
    assert summary.loc["their direction (°)", "AG3"] == pytest.approx(90, abs=1)
    assert np.isclose(summary.loc["position-model error at 500 px (µm)", "AG3"], 6.5, atol=0.3)
    assert summary.loc["focus minus 6-camera median (µm)", "AG1"] == pytest.approx(10.0, abs=1.0)
