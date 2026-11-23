"""Tests for `pfs.drp.qa.metrics.slitOffsets`.

The data are synthetic: a detector's fibers laid out in slit blocks, with a smooth
along-slit trend in the spatial offset and one planted bad fiber. The negative
controls pin two conventions: that detrending within the block is what makes a
modest outlier detectable (pooling the detector into one block misses it), and
that a difference below the invariance tolerance is not counted as a change.
"""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.slitOffsets import (
    INVARIANCE_FIELDS,
    SUMMARY_FIELDS,
    flagOutliers,
    offsetInvariance,
    offsetSummary,
    robustRms,
)

#: Slit blocks (MTP ferrules), fibers per block, and the planted bad fiber.
N_BLOCKS = 5
PER_BLOCK = 20
BLOCK_STEP = 1.0  # px of along-slit trend between adjacent blocks
NOISE = 0.01  # px of healthy fiber-to-fiber scatter within a block
BAD_FIBER = 2 * PER_BLOCK + 7  # a fiber in the middle block
BAD_SPATIAL = 0.30  # px, 30x the local noise but well under the between-block trend


def makeDetector(seed: int = 1):
    """Return (fiberId, spatial, spectral, group) for a healthy detector.

    The spatial offset steps block to block (the along-slit trend); the spectral
    offset is flat. Within a block the fibers scatter by ``NOISE``.
    """
    rng = np.random.default_rng(seed)
    nFibers = N_BLOCKS * PER_BLOCK
    fiberId = np.arange(1, nFibers + 1, dtype=np.int32)
    group = np.repeat([f"MTP{b}" for b in range(N_BLOCKS)], PER_BLOCK)
    level = np.repeat(np.arange(N_BLOCKS) * BLOCK_STEP, PER_BLOCK)
    spatial = level + rng.normal(0.0, NOISE, nFibers)
    spectral = rng.normal(0.0, NOISE, nFibers)
    return fiberId, spatial, spectral, group


def residualsFor(fiberId, spatial, spectral, group, **kwargs):
    from pfs.drp.qa.metrics.slitOffsets import slitOffsetResiduals

    return slitOffsetResiduals(fiberId, spatial, spectral, group, **kwargs)


def testRobustRms():
    assert robustRms([]) != robustRms([])  # NaN is not equal to itself
    assert robustRms([1.0, 1.0, 1.0]) == 0.0
    # 0.741 * IQR for a standard normal is ~1; a wide sample sits near its sigma.
    big = np.random.default_rng(0).normal(0.0, 2.0, 100000)
    assert robustRms(big) == pytest.approx(2.0, abs=0.05)


def testHealthyDetectorHasNoOutliers():
    fiberId, spatial, spectral, group = makeDetector()
    flagged = flagOutliers(residualsFor(fiberId, spatial, spectral, group))
    assert flagged["outlier"].sum() == 0
    # The trend is removed: residuals are centred on zero within each block.
    assert flagged["spatialResid"].abs().max() < 5 * NOISE


def testPlantedOutlierIsFlagged():
    fiberId, spatial, spectral, group = makeDetector()
    spatial[BAD_FIBER - 1] += BAD_SPATIAL
    flagged = flagOutliers(residualsFor(fiberId, spatial, spectral, group))

    assert flagged["outlier"].sum() == 1
    bad = flagged[flagged["fiberId"] == BAD_FIBER].iloc[0]
    assert bool(bad["spatialOutlier"]) and not bool(bad["spectralOutlier"])

    summary = offsetSummary(flagged)
    assert set(summary) == set(SUMMARY_FIELDS)
    assert summary["nSlitOffsetOutliers"] == 1
    assert summary["worstFiberId"] == BAD_FIBER
    assert summary["maxSpatialResid"] == pytest.approx(BAD_SPATIAL, abs=0.05)


def testGroupingIsLoadBearing():
    """Negative control: the same outlier is missed without the per-block detrend.

    Grouped by its slit block, the planted fiber stands 0.30 px above a ~0.01 px
    local scatter and is flagged. Pooled into a single block, the along-slit trend
    inflates the robust scale so far that the fiber is lost in it -- which is why
    the detrend is done per block, not over the whole detector.
    """
    fiberId, spatial, spectral, group = makeDetector()
    spatial[BAD_FIBER - 1] += BAD_SPATIAL

    grouped = flagOutliers(residualsFor(fiberId, spatial, spectral, group))
    assert grouped.loc[grouped["fiberId"] == BAD_FIBER, "outlier"].iloc[0]

    oneBlock = np.full(len(fiberId), "ALL")
    pooled = flagOutliers(residualsFor(fiberId, spatial, spectral, oneBlock))
    assert not pooled.loc[pooled["fiberId"] == BAD_FIBER, "outlier"].iloc[0]
    assert pooled.attrs["spatialScale"] > 10 * NOISE  # the trend inflated the pooled scale


def testUnselectedFiberDoesNotRaiseAlarmOrSetScale():
    fiberId, spatial, spectral, group = makeDetector()
    # A broken, unlit fiber with a wild offset: placed, but neither flagged nor
    # allowed to set the robust scale.
    spatial[BAD_FIBER - 1] += 50.0
    select = np.ones(len(fiberId), dtype=bool)
    select[BAD_FIBER - 1] = False

    flagged = flagOutliers(residualsFor(fiberId, spatial, spectral, group, select=select))
    assert flagged["outlier"].sum() == 0
    assert flagged.attrs["spatialScale"] < 5 * NOISE  # the wild fiber did not inflate it


def testInvarianceZeroWhenIdentical():
    fiberId, spatial, spectral, _ = makeDetector()
    result = offsetInvariance(fiberId, spatial, spectral, spatial.copy(), spectral.copy())
    assert set(result) == set(INVARIANCE_FIELDS)
    assert result["nSlitOffsetChanged"] == 0
    assert result["maxSpatialOffsetDelta"] == 0.0
    assert result["worstFiberId"] is None


def testInvarianceCountsAChange():
    fiberId, spatial, spectral, _ = makeDetector()
    visitSpatial = spatial.copy()
    visitSpatial[BAD_FIBER - 1] += 0.4
    result = offsetInvariance(fiberId, visitSpatial, spectral, spatial, spectral)
    assert result["nSlitOffsetChanged"] == 1
    assert result["maxSpatialOffsetDelta"] == pytest.approx(0.4)
    assert result["worstFiberId"] == BAD_FIBER


def testInvarianceIgnoresSubTolerance():
    """Negative control: a difference at the tolerance floor is not a change."""
    fiberId, spatial, spectral, _ = makeDetector()
    visitSpatial = spatial + 1e-9  # numerical dust, below tol
    result = offsetInvariance(fiberId, visitSpatial, spectral, spatial, spectral, tol=1e-6)
    assert result["nSlitOffsetChanged"] == 0


def testLengthMismatchRaises():
    with pytest.raises(ValueError, match="same length"):
        residualsFor(np.array([1, 2, 3]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array(["a", "a"]))


def testNaNOffsetsArePlacedNotFlagged():
    fiberId, spatial, spectral, group = makeDetector()
    spatial[BAD_FIBER - 1] = np.nan
    flagged = flagOutliers(residualsFor(fiberId, spatial, spectral, group))
    bad = flagged[flagged["fiberId"] == BAD_FIBER].iloc[0]
    assert not bool(bad["outlier"])
    assert not bool(bad["selected"])
    assert isinstance(flagged, pd.DataFrame)


# --- cross-arm confirmation and run-to-run comparison -----------------------

from pfs.drp.qa.metrics.slitOffsets import (  # noqa: E402
    CONFIRMED_COLUMNS,
    changeSummary,
    compareRuns,
    crossArmConfirmed,
)


def flaggedFrame(records):
    """Build a `flagOutliers`-shaped frame from (spectrograph, arm, fiberId, outlier) tuples."""
    rows = []
    for spec, arm, fiberId, outlier in records:
        rows.append(
            {
                "spectrograph": spec,
                "arm": arm,
                "fiberId": fiberId,
                "outlier": outlier,
                "selected": True,
                "spatialResid": 0.3 if outlier else 0.01,
                "spectralResid": 0.0,
            }
        )
    return pd.DataFrame(rows)


def testCrossArmConfirmedNeedsTwoArms():
    # fiber 940 is an outlier in 3 arms (confirmed); fiber 999 in only 1 (not).
    records = [
        (2, "b", 940, True),
        (2, "m", 940, True),
        (2, "r", 940, True),
        (2, "n", 940, False),
        (2, "b", 999, True),  # single-arm: the negative control
        (2, "m", 999, False),
        (1, "b", 100, False),  # a clean spectrograph
    ]
    confirmed = crossArmConfirmed(flaggedFrame(records), minArms=2)
    assert list(confirmed.columns) == list(CONFIRMED_COLUMNS)
    assert confirmed["fiberId"].to_list() == [940]
    row = confirmed.iloc[0]
    assert row["nArms"] == 3
    assert row["arms"] == "b,m,r"
    # Negative control: a single-arm outlier is not confirmed.
    assert 999 not in confirmed["fiberId"].to_list()


def testCrossArmConfirmedEmpty():
    confirmed = crossArmConfirmed(flaggedFrame([(1, "b", 100, False)]), minArms=2)
    assert confirmed.empty
    assert list(confirmed.columns) == list(CONFIRMED_COLUMNS)


def confirmedFrame(pairs):
    """Return a minimal `crossArmConfirmed`-shaped frame from (spectrograph, fiberId) pairs."""
    return pd.DataFrame(
        {
            "spectrograph": [p[0] for p in pairs],
            "fiberId": [p[1] for p in pairs],
            "nArms": [4] * len(pairs),
            "arms": ["b,m,n,r"] * len(pairs),
            "maxSpatialResid": [0.3] * len(pairs),
            "maxSpectralResid": [0.0] * len(pairs),
        }
    ).astype(CONFIRMED_COLUMNS)


def testCompareRunsClassifiesChange():
    current = confirmedFrame([(2, 940), (2, 802)])  # 940 persists, 802 is new
    previous = confirmedFrame([(2, 940), (3, 1358)])  # 1358 cleared
    comparison = compareRuns(current, previous)
    keys = zip(comparison["spectrograph"], comparison["fiberId"], strict=True)
    status = dict(zip(keys, comparison["status"], strict=True))
    assert status[(2, 940)] == "persisted"
    assert status[(2, 802)] == "new"
    assert status[(3, 1358)] == "cleared"

    summary = changeSummary(comparison)
    assert set(summary) == {"nConfirmed", "nNew", "nCleared", "nPersisted"}
    assert summary == {"nConfirmed": 2, "nNew": 1, "nCleared": 1, "nPersisted": 1}


def testCompareRunsAllNewWhenNoPrevious():
    current = confirmedFrame([(2, 940), (3, 1358)])
    previous = confirmedFrame([])
    summary = changeSummary(compareRuns(current, previous))
    assert summary["nNew"] == 2
    assert summary["nCleared"] == 0
    assert summary["nPersisted"] == 0


def testOffsetTrajectory():
    from pfs.drp.qa.metrics.slitOffsets import offsetTrajectory

    r1 = confirmedFrame([(2, 940), (3, 1358)])
    r2 = confirmedFrame([(2, 940), (2, 802)])
    traj = offsetTrajectory({"run1": r1, "run2": r2})
    assert list(traj.columns) == ["spectrograph", "fiberId", "run1", "run2"]
    # 940 chronic (both runs), 1358 cleared after run1, 802 new in run2; nArms is 4 in confirmedFrame.
    row940 = traj[(traj["spectrograph"] == 2) & (traj["fiberId"] == 940)].iloc[0]
    assert (row940["run1"], row940["run2"]) == (4, 4)
    row1358 = traj[traj["fiberId"] == 1358].iloc[0]
    assert (row1358["run1"], row1358["run2"]) == (4, 0)
    row802 = traj[traj["fiberId"] == 802].iloc[0]
    assert (row802["run1"], row802["run2"]) == (0, 4)
    # A fiber confirmed in no run never appears.
    assert len(traj) == 3
