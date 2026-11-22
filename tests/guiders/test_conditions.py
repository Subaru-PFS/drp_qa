"""Per-AG-exposure seeing and transparency from good-match spots (`pfs.drp.qa.guiders.conditions`).

The opdb read lives in `queries.readConditionSpots`; the aggregation in `exposureConditions` is pure
and is most of what this file tests, against a direct `pfs.utils.agMetrics` computation on the same
spots. Negative controls cover the sub-PSF seeing drop, the brighter-star transparency selection
(matching `agMetrics.summarizeTransparency`, the ag actor log's), and that the moments are taken
already corrected (the ``"git"`` doubling is done in the SQL, not here). `writeConditions` is checked
end-to-end against an in-memory SQLite with the real qadb table columns, so a column drift is caught.
"""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from pfs.drp.qa.guiders.conditions import exposureConditions, writeConditions
from pfs.utils import agMetrics

COLUMNS = [
    "agc_exposure_id",
    "altitude",
    "mxx",
    "myy",
    "estimated_magnitude",
    "guide_star_magnitude",
    "guide_star_color",
]


def makeSpots(rows):
    """Build a spots DataFrame from (agc_exposure_id, altitude, mxx, myy, est, cat, color) rows."""
    spots = pd.DataFrame(rows, columns=COLUMNS)
    spots["pfs_visit_id"] = 42
    spots["taken_at"] = pd.Timestamp("2026-11-20 03:00:00")
    return spots


def perSpotSeeing(spots):
    """Return each spot's calibrated seeing, straight from agMetrics (moments taken as corrected)."""
    fwhm = agMetrics.momentsToFwhm(spots.mxx.to_numpy(float), spots.myy.to_numpy(float))
    return np.asarray(agMetrics.calibratedSeeing(fwhm, spots.altitude.to_numpy(float)), float)


def perSpotTransparency(spots):
    """Return each spot's raw transparency (no selection), straight from agMetrics."""
    return np.asarray(
        agMetrics.magnitudesToTransparency(
            spots.estimated_magnitude.to_numpy(float),
            spots.guide_star_magnitude.to_numpy(float),
            spots.guide_star_color.to_numpy(float),
        ),
        float,
    )


def test_seeing_aggregates_match_agMetrics():
    """One exposure's seeing mean, median and sigma match agMetrics over every good match."""
    spots = makeSpots(
        [
            (100, 60.0, 9.0, 8.0, 15.0, 15.1, 0.6),
            (100, 60.0, 12.0, 11.0, 14.6, 14.7, 0.5),
            (100, 60.0, 7.0, 8.0, 16.2, 16.0, 0.7),
        ]
    )
    out = exposureConditions(spots)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.agc_exposure_id == 100 and row.pfs_visit_id == 42
    seeing = pd.Series(perSpotSeeing(spots))
    assert np.all(np.isfinite(seeing))  # a clean exposure
    assert row.seeing_mean == pytest.approx(seeing.mean())
    assert row.seeing_median == pytest.approx(seeing.median())
    assert row.seeing_sigma == pytest.approx(seeing.std())  # pandas ddof=1, as the code uses


def test_moments_taken_as_corrected():
    """ExposureConditions does not double the moments; the ``"git"`` correction is the SQL's job."""
    spots = makeSpots([(1, 90.0, 9.0, 9.0, 15.0, 15.0, 0.5)])
    got = exposureConditions(spots).iloc[0].seeing_median
    assert got == pytest.approx(perSpotSeeing(spots)[0])  # variancePerMoment = 1
    # Negative control: it must not be the doubled-moment ("git") seeing.
    doubled = agMetrics.momentsToFwhm(
        spots.mxx.to_numpy(float),
        spots.myy.to_numpy(float),
        variancePerMoment=agMetrics.VARIANCE_PER_AGC_MOMENT,
    )
    doubledSeeing = float(agMetrics.calibratedSeeing(doubled, 90.0)[0])
    assert not np.isclose(got, doubledSeeing)


def test_transparency_matches_summarizeTransparency():
    """The transparency median is the ag actor log's: agMetrics.summarizeTransparency, brighter stars."""
    # A bright and a faint half (the faint quartile is dropped); all transparencies in range.
    spots = makeSpots(
        [
            (300, 90.0, 10.0, 10.0, 14.0, 14.0, 0.5),
            (300, 90.0, 10.0, 10.0, 14.3, 14.2, 0.5),
            (300, 90.0, 10.0, 10.0, 15.0, 15.0, 0.5),
            (300, 90.0, 10.0, 10.0, 17.5, 15.0, 0.5),  # the faintest: excluded, and 2.5 mag off (dim)
        ]
    )
    raw = perSpotTransparency(spots)
    expected = agMetrics.summarizeTransparency(raw, spots.estimated_magnitude.to_numpy(float))
    row = exposureConditions(spots).iloc[0]
    assert row.transparency_median == pytest.approx(expected.transparency)
    # The mean and sigma are over the same brighter-star selection.
    faintest = np.nanquantile(spots.estimated_magnitude.to_numpy(float), 0.75)
    sel = (spots.estimated_magnitude.to_numpy(float) <= faintest) & (raw > 0) & (raw < 2)
    assert row.transparency_mean == pytest.approx(pd.Series(raw[sel]).mean())
    assert row.transparency_sigma == pytest.approx(pd.Series(raw[sel]).std())
    # Negative control: the faint spot is excluded, so moving its (dim) transparency leaves the median.
    darker = spots.copy()
    darker.loc[3, "estimated_magnitude"] = 20.0
    assert exposureConditions(darker).iloc[0].transparency_median == pytest.approx(expected.transparency)


def test_sub_psf_spot_drops_from_seeing():
    """A spot the AG cannot resolve (NaN seeing) leaves the exposure's seeing but not its other spots."""
    spots = makeSpots(
        [
            (200, 90.0, 10.0, 10.0, 15.0, 15.0, 0.5),  # resolved
            (200, 90.0, 0.3, 0.3, 15.0, 15.0, 0.5),  # below the instrument PSF
        ]
    )
    seeing = perSpotSeeing(spots)
    assert np.isfinite(seeing[0]) and np.isnan(seeing[1])  # the scenario: one resolved, one not
    row = exposureConditions(spots).iloc[0]
    # The NaN spot is dropped, not propagated or counted as zero: the mean is the one good spot's seeing.
    assert row.seeing_mean == pytest.approx(seeing[0])
    # Negative control: counting the sub-PSF spot as zero would halve the mean.
    assert row.seeing_mean > seeing[0] / 2 + 1e-6


def test_out_of_range_transparency_is_cut():
    """A star far from its catalogue magnitude (a mismatch, transparency outside 0-2) is left out."""
    # Equal estimated magnitudes so both survive the brighter-star cut; only the range cut acts.
    spots = makeSpots(
        [
            (400, 90.0, 10.0, 10.0, 15.0, 15.0, 0.5),  # observed == catalogue -> transparency ~1
            (400, 90.0, 10.0, 10.0, 15.0, 17.5, 0.5),  # 2.5 mag brighter than catalogue -> > 2
        ]
    )
    raw = perSpotTransparency(spots)
    assert raw[1] > 2.0  # the scenario: second star out of range
    row = exposureConditions(spots).iloc[0]
    assert row.transparency_mean == pytest.approx(raw[0])
    # Negative control: the out-of-range value would pull the mean above 2 if it were kept.
    assert row.transparency_mean < 2.0


def test_altitude_is_degrees_airmass_correction():
    """Altitude is degrees: at the same measured FWHM, lower altitude means more airmass, less seeing."""
    moments = (10.0, 10.0)
    high = exposureConditions(makeSpots([(1, 85.0, *moments, 15.0, 15.0, 0.5)])).iloc[0].seeing_median
    low = exposureConditions(makeSpots([(2, 35.0, *moments, 15.0, 15.0, 0.5)])).iloc[0].seeing_median
    assert np.isfinite(high) and np.isfinite(low)
    assert low < high  # more airmass at 35 deg than 85 deg -> a larger correction down to zenith
    # Negative control: altitude must matter (it is not ignored, e.g. treated as a constant).
    assert not np.isclose(low, high)


def test_empty_input_gives_empty_frame():
    """No spots returns an empty frame with the output columns, so a write of it is a no-op."""
    out = exposureConditions(makeSpots([]))
    assert len(out) == 0
    for column in ("agc_exposure_id", "pfs_visit_id", "seeing_median", "transparency_sigma"):
        assert column in out.columns


# The qadb columns writeConditions targets (spt_qa_database's seeing_agc_exposure /
# transparency_agc_exposure). Kept in step with that schema, so a rename or typo in the writer's
# column lists fails here instead of silently against the summit database.
_QADB_DDL = """
CREATE TABLE seeing_agc_exposure (
    pfs_visit_id INTEGER, agc_exposure_id INTEGER,
    seeing_mean REAL, seeing_median REAL, seeing_sigma REAL,
    wavelength_ref REAL, taken_at TIMESTAMP,
    PRIMARY KEY (pfs_visit_id, agc_exposure_id)
);
CREATE TABLE transparency_agc_exposure (
    pfs_visit_id INTEGER, agc_exposure_id INTEGER,
    transparency_mean REAL, transparency_median REAL, transparency_sigma REAL,
    wavelength_ref REAL, taken_at TIMESTAMP,
    PRIMARY KEY (pfs_visit_id, agc_exposure_id)
);
"""


class _QadbStub:
    """A SQLite stand-in exposing the ``commit`` / ``query_dataframe`` that writer.upsert uses."""

    def __init__(self):
        self.engine = create_engine("sqlite://")
        with self.engine.begin() as conn:
            for statement in filter(str.strip, _QADB_DDL.split(";")):
                conn.execute(text(statement))

    def commit(self, sql, params=None):
        with self.engine.begin() as conn:
            conn.execute(text(sql), params or {})

    def query_dataframe(self, sql):
        return pd.read_sql(sql, self.engine)


def test_writeConditions_upserts_both_tables(monkeypatch):
    """WriteConditions writes the seeing and transparency rows under the real qadb columns, idempotently."""
    spots = makeSpots(
        [
            (100, 60.0, 9.0, 8.0, 15.0, 15.1, 0.6),
            (100, 60.0, 12.0, 11.0, 14.6, 14.7, 0.5),
            (101, 70.0, 10.0, 10.0, 15.0, 15.0, 0.5),
        ]
    )
    monkeypatch.setattr("pfs.drp.qa.guiders.conditions.readConditionSpots", lambda opdb, visits: spots)
    qadb = _QadbStub()
    counts = writeConditions([42], opdb=object(), qadb=qadb)
    assert counts == (2, 2)  # two AG exposures, both tables

    expected = exposureConditions(spots).set_index("agc_exposure_id")
    seeing = qadb.query_dataframe("SELECT * FROM seeing_agc_exposure").set_index("agc_exposure_id")
    assert set(seeing.columns) == {
        "pfs_visit_id",
        "seeing_mean",
        "seeing_median",
        "seeing_sigma",
        "wavelength_ref",
        "taken_at",
    }
    assert seeing.loc[100].seeing_median == pytest.approx(expected.loc[100].seeing_median)
    assert pd.isna(seeing.loc[100].wavelength_ref)  # the agMetrics method stores no reference wavelength
    transp = qadb.query_dataframe("SELECT * FROM transparency_agc_exposure")
    assert len(transp) == 2 and "transparency_sigma" in transp.columns

    # Idempotent: a second write replaces the rows, it does not duplicate them.
    writeConditions([42], opdb=object(), qadb=qadb)
    assert len(qadb.query_dataframe("SELECT * FROM seeing_agc_exposure")) == 2


def test_describeConditions_summarizes_per_visit():
    """The shadow lines carry the totals and each visit's median seeing and transparency."""
    from pfs.drp.qa.guiders.conditions import describeConditions

    spots = makeSpots(
        [
            (100, 60.0, 9.0, 8.0, 15.0, 15.1, 0.6),
            (101, 70.0, 10.0, 10.0, 15.0, 15.0, 0.5),
        ]
    )
    spots.loc[spots.agc_exposure_id == 101, "pfs_visit_id"] = 43  # the two exposures are in two visits
    cond = exposureConditions(spots)
    lines = describeConditions(cond)
    assert "shadow, not written" in lines[0]
    assert "2 AG exposure(s) over 2 visit(s)" in lines[0]
    assert any("visit=42:" in line for line in lines) and any("visit=43:" in line for line in lines)
    # Negative control: it never claims a database write.
    assert not any("wrote" in line.lower() for line in lines)
    assert describeConditions(exposureConditions(makeSpots([])))[0].endswith("no good-match AG exposures")


def test_logConditions_logs_and_writes_nothing(monkeypatch):
    """LogConditions computes, logs the summary through the callable and returns the exposure count."""
    from pfs.drp.qa.guiders import conditions

    spots = makeSpots(
        [
            (100, 60.0, 9.0, 8.0, 15.0, 15.1, 0.6),
            (100, 60.0, 12.0, 11.0, 14.6, 14.7, 0.5),
            (101, 70.0, 10.0, 10.0, 15.0, 15.0, 0.5),
        ]
    )
    monkeypatch.setattr(conditions, "readConditionSpots", lambda opdb, visits: spots)
    captured = []
    n = conditions.logConditions([42], opdb=object(), log=captured.append)
    assert n == 2  # two AG exposures (the two mag-varied rows of exposure 100 are one exposure)
    assert captured and "shadow" in captured[0]
    # Negative control: no qadb was touched (none was passed, and a write path would have needed it).
    assert any("visit=42:" in line for line in captured)
