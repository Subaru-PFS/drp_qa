"""Tests for ``pfs.drp.qa.guiders.runAnalysis``, on synthetic runs."""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders.runAnalysis import (
    CAMERA_GROUPS,
    CameraState,
    RunConfig,
    describePeriod,
    findDropouts,
    findFlags,
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
    cameras.loc[cameras.agc_exposure_id.isin(lone) & (cameras.agc_camera_id > 0), "n_good"] = 0
    cameras.loc[cameras.agc_exposure_id.isin(lone) & (cameras.agc_camera_id == 0), "n_good"] = 1
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
