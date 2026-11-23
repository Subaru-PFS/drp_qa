"""slit_offset_confirmed: the write replaces a run's rows, and the trajectory reads across runs.

On SQLite, and on PostgreSQL when PFS_QADB_TEST_DSN is set. The rows use ``u/test/`` collections and
``testRun`` run names, so a copy of the real qadb keeps its own rows.
"""

import pandas as pd

from pfs.drp.qa.metrics.slitOffsets import CONFIRMED_COLUMNS, offsetTrajectory
from pfs.drp.qa.qadb.reader import readSlitOffsetConfirmed
from pfs.drp.qa.qadb.writer import writeSlitOffsetConfirmed


def conf(pairs):
    """Return a `crossArmConfirmed`-shaped frame from (spectrograph, fiberId) pairs."""
    return pd.DataFrame(
        {
            "spectrograph": [s for s, _ in pairs],
            "fiberId": [f for _, f in pairs],
            "nArms": [4] * len(pairs),
            "arms": ["b,m,n,r"] * len(pairs),
            "maxSpatialResid": [0.30] * len(pairs),
            "maxSpectralResid": [0.10] * len(pairs),
        }
    ).astype(CONFIRMED_COLUMNS)


def testRoundTripAndReplace(qadb):
    coll = "u/test/run31/detectorMaps"
    assert writeSlitOffsetConfirmed(qadb, "testRun31", coll, conf([(2, 940), (2, 941), (3, 1358)])) == 3

    back = readSlitOffsetConfirmed(qadb, run="testRun31")
    assert sorted(zip(back["spectrograph"], back["fiberId"], strict=True)) == [(2, 940), (2, 941), (3, 1358)]
    assert list(back.columns) == [
        "run",
        "spectrograph",
        "fiberId",
        "nArms",
        "arms",
        "maxSpatialResid",
        "maxSpectralResid",
    ]

    # Re-writing the run replaces its rows: 1358 is no longer confirmed, so it goes.
    writeSlitOffsetConfirmed(qadb, "testRun31", coll, conf([(2, 940), (2, 941)]))
    back = readSlitOffsetConfirmed(qadb, run="testRun31")
    assert sorted(zip(back["spectrograph"], back["fiberId"], strict=True)) == [(2, 940), (2, 941)]


def testEmptyRunClearsIt(qadb):
    coll = "u/test/run31/detectorMaps"
    writeSlitOffsetConfirmed(qadb, "testRun31", coll, conf([(2, 940)]))
    # A run with no confirmed outlier writes nothing and clears the run's rows.
    assert writeSlitOffsetConfirmed(qadb, "testRun31", coll, conf([])) == 0
    assert readSlitOffsetConfirmed(qadb, run="testRun31").empty


def testTrajectoryAcrossRuns(qadb):
    writeSlitOffsetConfirmed(qadb, "testRun30", "u/test/run30/dm", conf([(2, 940), (3, 1358)]))
    writeSlitOffsetConfirmed(qadb, "testRun31", "u/test/run31/dm", conf([(2, 940), (2, 802)]))
    runs = ["testRun30", "testRun31"]
    byRun = {r: readSlitOffsetConfirmed(qadb, run=r).drop(columns="run") for r in runs}
    matrix = offsetTrajectory(byRun)

    assert list(matrix.columns) == ["spectrograph", "fiberId", *runs]
    row940 = matrix[(matrix["spectrograph"] == 2) & (matrix["fiberId"] == 940)].iloc[0]
    assert (row940["testRun30"], row940["testRun31"]) == (4, 4)  # chronic
    row1358 = matrix[matrix["fiberId"] == 1358].iloc[0]
    assert (row1358["testRun30"], row1358["testRun31"]) == (4, 0)  # cleared
    row802 = matrix[matrix["fiberId"] == 802].iloc[0]
    assert (row802["testRun30"], row802["testRun31"]) == (0, 4)  # new
