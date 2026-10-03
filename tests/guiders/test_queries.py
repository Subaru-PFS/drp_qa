"""Tests for ``pfs.drp.qa.guiders.queries``, with a fake opdb and a stub butler."""

import logging
import re

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders import queries
from pfs.drp.qa.guiders.coordinates import OPDB_Y_COLUMNS, opdbToHardware
from pfs.drp.qa.guiders.queries import (
    AGC_DATA_COLUMNS,
    _pairTelStatus,
    find_W_M2OFF3,
    readAgcData,
    readAGCStars,
    readInstPa,
    readPfsDesign,
    readRawMetadata,
    readSpSInfo,
    readTelStatus,
)

VISITS = [120000, 120001, 120002]


def checkBound(call, *values):
    """Check that values were bound as Python ints, not formatted into the SQL."""
    for value in values:
        assert not re.search(rf"\b{value}\b", call.sql), f"{value} is in the SQL"
    for value in call.params.values():
        for v in value if isinstance(value, list) else [value]:
            assert not isinstance(v, np.generic), f"{v!r} is a NumPy scalar; psycopg can't bind it"


# readAgcData


def testReadAgcDataColumns(makeOpdb, makeAgcData):
    opdb, agcData = makeOpdb()
    data = readAgcData(opdb, VISITS)

    assert list(data.columns) == list(AGC_DATA_COLUMNS)
    assert len(data) == len(agcData)
    pd.testing.assert_index_equal(data.index, pd.RangeIndex(len(data)))  # fresh, though FakeOpDB shuffles
    assert data.agc_exposure_id.is_monotonic_increasing
    assert not data.duplicated(["agc_exposure_id", "agc_camera_id", "spot_id"]).any()
    assert data.agc_camera_id.dtype == np.int64

    # The test fixture describes what readAgcData returns.
    assert set(makeAgcData(stars=True).columns) <= set(AGC_DATA_COLUMNS)


def testReadAgcDataOneQueryForAllVisits(makeOpdb):
    """Every visit is read in one query per table, with bound parameters."""
    opdb, _ = makeOpdb()
    readAgcData(opdb, (np.int64(v) for v in reversed(VISITS)))

    assert len(opdb.calls) == 3
    for call in opdb.calls:
        assert call.params == {"visits": VISITS}
        checkBound(call, *VISITS)


def testReadAgcDataHardwareFrame(makeOpdb):
    """Positions are in hardware coordinates: y is negated once, x isn't."""
    opdb, agcData = makeOpdb(nVisit=1)
    data = readAgcData(opdb, VISITS[0])

    expected = agcData.sort_values(["agc_exposure_id", "guide_star_id"], ignore_index=True)
    for column in ("agc_center_x_mm", "agc_nominal_x_mm", *OPDB_Y_COLUMNS):
        np.testing.assert_allclose(data[column], expected[column], err_msg=column)
    with pytest.raises(ValueError, match="already in hardware"):
        opdbToHardware(data)

    # Negative control: the opdb's own y doesn't match.
    raw = opdb.tables["agc_data"].sort_values(["agc_exposure_id", "guide_star_id"])
    assert not np.allclose(raw.agc_center_y_mm, expected.agc_center_y_mm)


def testReadAgcDataTelStatus(makeOpdb):
    """m2_off3, tel_ra and tel_dec are paired with the right exposures."""
    opdb, agcData = makeOpdb()
    data = readAgcData(opdb, VISITS)

    expected = agcData.sort_values(["agc_exposure_id", "guide_star_id"], ignore_index=True)
    np.testing.assert_allclose(data.m2_off3, expected.m2_off3)
    np.testing.assert_allclose(data.tel_ra, 150.0 + 1e-4 * data.agc_exposure_id)
    assert data.m2_off3.nunique() > 1  # so the check can tell exposures apart


def testReadAgcDataMissingTelStatus(makeOpdb, caplog):
    """A star whose exposure has no tel_status row is kept, with NaN m2_off3."""
    opdb, agcData = makeOpdb()
    telStatus = opdb.tables["agc_tel_status"]
    last = telStatus[telStatus.pfs_visit_id == VISITS[0]].status_sequence_id.max()
    opdb.tables["agc_tel_status"] = telStatus[telStatus.status_sequence_id != last]

    with caplog.at_level(logging.WARNING, logger=queries.__name__):
        data = readAgcData(opdb, VISITS)

    assert len(data) == len(agcData)  # an inner merge would drop the exposure's stars
    lastExposure = data[data.pfs_visit_id == VISITS[0]].agc_exposure_id.max()
    assert data[data.agc_exposure_id == lastExposure].m2_off3.isna().all()
    assert data[data.agc_exposure_id != lastExposure].m2_off3.notna().all()
    assert f"pfs_visit_id {VISITS[0]}: 8 AG exposures but 7 AG rows" in caplog.text


def testReadAgcDataNullTelStatus(makeOpdb, monkeypatch):
    """All-NULL tel_status columns are float NaN, and pandas options are left alone."""
    opdb, _ = makeOpdb(nVisit=1)
    telStatus = opdb.tables["agc_tel_status"].astype(object)
    telStatus[["m2_off3", "tel_ra", "tel_dec"]] = None
    opdb.tables["agc_tel_status"] = telStatus

    def setOption(*args, **kwargs):
        raise AssertionError(f"pd.set_option{args} changes the caller's pandas options")

    monkeypatch.setattr(pd, "set_option", setOption)
    data = readAgcData(opdb, VISITS[0])

    for column in ("m2_off3", "tel_ra", "tel_dec"):
        assert data[column].dtype == np.float64
        assert data[column].isna().all()


def testReadAgcDataGuideStarFlag(makeOpdb):
    """A star missing from pfs_design_agc gets guide_star_flag 0."""
    opdb, _ = makeOpdb(nVisit=1)
    rows = opdb.tables["agc_data"].astype({"guide_star_flag": float})
    rows.loc[rows.guide_star_id == 1, "guide_star_flag"] = np.nan
    opdb.tables["agc_data"] = rows

    data = readAgcData(opdb, VISITS[0])

    assert data.guide_star_flag.dtype == np.int64
    assert (data[data.guide_star_id == 1].guide_star_flag == 0).all()
    assert (data[data.guide_star_id != 1].guide_star_flag > 0).all()


def testReadAgcDataEmpty(makeOpdb, caplog):
    opdb, _ = makeOpdb(nVisit=1)

    with pytest.raises(ValueError, match="No visits"):
        readAgcData(opdb, [])
    assert opdb.calls == []

    with caplog.at_level(logging.WARNING, logger=queries.__name__):
        data = readAgcData(opdb, [1, 2])
    assert data.empty
    assert list(data.columns) == list(AGC_DATA_COLUMNS)
    assert "No AG data for pfs_visit_id 1, 2" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=queries.__name__):
        data = readAgcData(opdb, [VISITS[0], 1])
    assert not data.empty
    assert "No AG data for pfs_visit_id 1" in caplog.text


def testReadAgcDataRejectsConnection():
    """A psycopg2-style connection (no query_dataframe) gets a clear error."""
    with pytest.raises(TypeError, match=r"must be a pfs\.utils\.database\.opdb\.OpDB"):
        readAgcData(object(), VISITS)


def testReadAgcDataButler(makeOpdb, StubButler):
    """With a butler, inst_pa and the missing m2_off3 come from the raw headers."""
    opdb, agcData = makeOpdb()
    telStatus = opdb.tables["agc_tel_status"]
    noM2Off3 = VISITS[1]  # e.g. before 2025-03-21
    telStatus.loc[telStatus.pfs_visit_id == noM2Off3, "m2_off3"] = np.nan
    rows = opdb.tables["agc_data"]
    rows["m2_pos3"] = 0.01 * (rows.agc_exposure_id - 1000)  # moves during each visit
    # Newest first, so the first row isn't the first exposure.
    opdb.tables["agc_data"] = lambda params: rows.sort_values("agc_exposure_id", ascending=False)
    butler = StubButler([{"visit": v, "arm": arm, "spectrograph": 1} for v in VISITS for arm in "br"])

    data = readAgcData(opdb, VISITS, butler=butler)

    np.testing.assert_array_equal(data.inst_pa, 10.0 * data.pfs_visit_id + 1)  # the r arm

    visit = data.pfs_visit_id == noM2Off3
    first = data[visit].agc_exposure_id.min()
    m2Pos3First = data[data.agc_exposure_id == first].m2_pos3.iloc[0]
    np.testing.assert_allclose(data[visit].m2_off3, -noM2Off3 / 1e6 + data[visit].m2_pos3 - m2Pos3First)
    assert data[visit].m2_off3.nunique() > 1

    # The other visits keep tel_status's m2_off3.
    expected = agcData.sort_values(["agc_exposure_id", "guide_star_id"], ignore_index=True)
    np.testing.assert_allclose(data[~visit].m2_off3, expected[~visit].m2_off3)


def testReadAgcDataNoButler(makeOpdb):
    opdb, _ = makeOpdb(nVisit=1)
    data = readAgcData(opdb, VISITS[0])
    assert data.inst_pa.isna().all()


# _pairTelStatus


def makeExposures(nByVisit):
    rows = []
    aid = 1000
    for visit, n in nByVisit.items():
        for _ in range(n):
            aid += 1
            rows.append({"pfs_visit_id": visit, "agc_exposure_id": aid})
    return pd.DataFrame(rows)


def makeTelStatus(nByVisit):
    rows = []
    for visit, n in nByVisit.items():
        for i in range(n):
            rows.append(
                {
                    "pfs_visit_id": visit,
                    "status_sequence_id": 100 + i,
                    "m2_off3": visit + i / 10,
                    "tel_ra": 0.0,
                }
            )
    return pd.DataFrame(rows).assign(tel_dec=0.0)


def testPairTelStatus():
    exposures = makeExposures({1: 3, 2: 2})
    telStatus = makeTelStatus({1: 3, 2: 2})
    paired = _pairTelStatus(
        exposures.sample(frac=1, random_state=1), telStatus.sample(frac=1, random_state=2)
    )

    assert list(paired.agc_exposure_id) == [1001, 1002, 1003, 1004, 1005]
    np.testing.assert_allclose(paired.m2_off3, [1.0, 1.1, 1.2, 2.0, 2.1])


def testPairTelStatusPerVisit(caplog):
    """An extra row in one visit doesn't shift the pairs in another."""
    exposures = makeExposures({1: 2, 2: 2})
    telStatus = makeTelStatus({1: 3, 2: 2})
    with caplog.at_level(logging.WARNING, logger=queries.__name__):
        paired = _pairTelStatus(exposures, telStatus)

    np.testing.assert_allclose(paired.m2_off3, [1.0, 1.1, 2.0, 2.1])
    assert "pfs_visit_id 1: 2 AG exposures but 3 AG rows in tel_status; dropping the last 1 tel_status" in (
        caplog.text
    )

    # Negative control: pairing across visits takes visit 1's extra row for visit 2.
    acrossVisits = telStatus.m2_off3.iloc[: len(exposures)].to_numpy()
    assert not np.allclose(acrossVisits, paired.m2_off3)


def testPairTelStatusMoreExposures(caplog):
    exposures = makeExposures({1: 3})
    with caplog.at_level(logging.WARNING, logger=queries.__name__):
        paired = _pairTelStatus(exposures, makeTelStatus({1: 2}))

    assert list(paired.agc_exposure_id) == [1001, 1002]
    assert "dropping the last 1 AG exposures" in caplog.text


# Butler readers


def testReadRawMetadata(StubButler):
    """The Gen3 API is used (StubButler has no Gen2 raw_md)."""
    butler = StubButler([{"visit": 100, "arm": "b", "spectrograph": 1}])
    assert readRawMetadata(butler, np.int64(100))["INST-PA"] == 1000
    assert readRawMetadata(butler, 100, arm="r") is None
    assert readRawMetadata(butler, 101) is None


def testReadInstPa(StubButler, caplog):
    butler = StubButler(
        [
            {"visit": 100, "arm": "b", "spectrograph": 1},
            {"visit": 100, "arm": "r", "spectrograph": 1},
            {"visit": 101, "arm": "b", "spectrograph": 1},
        ]
    )
    with caplog.at_level(logging.WARNING, logger=queries.__name__):
        instPa = readInstPa(butler, [102, 101, 100])

    assert instPa[100] == 1001  # r1, as requested
    assert instPa[101] == 1010  # no r1, so b1
    assert np.isnan(instPa[102])  # no raws
    assert "pfs_visit_id 102: no raws" in caplog.text

    assert readInstPa(butler, [100], {"arm": "b"})[100] == 1000


def testFindW_M2OFF3(StubButler):
    butler = StubButler([{"visit": 100, "arm": "r", "spectrograph": 1}])
    assert find_W_M2OFF3(butler, 102) == -100 / 1e6  # searches back to 100

    with pytest.raises(RuntimeError, match=r"in visit 99 or the 2 visits before it"):
        find_W_M2OFF3(butler, 99, nTry=3)
    with pytest.raises(RuntimeError, match="butler is needed"):
        find_W_M2OFF3(None, 100)


# Other opdb readers


def testReadAGCStars(FakeOpDB):
    stars = pd.DataFrame({"guide_star_id": [1, 2], "guide_star_parallax": [0.5, 0.1]})
    opdb = FakeOpDB({"stars": stars}, patterns={"stars": "pfs_design_agc"})

    pd.testing.assert_frame_equal(readAGCStars(opdb, np.int64(123456789012)), stars)
    pd.testing.assert_frame_equal(readAGCStars(opdb, 123456789012, pfs_visit_id=120000), stars)

    design, config = opdb.calls
    assert design.params == {"pfs_design_id": 123456789012}
    assert config.params == {"pfs_design_id": 123456789012, "pfs_visit_id": 120000}
    for call in opdb.calls:
        checkBound(call, 123456789012, 120000)
        # The missing comma read guide_star_parallax as ra_center_design.
        assert re.search(r"guide_star_parallax,\s", call.sql)
    assert "ra_center_designed" in design.sql
    # pfs_config_agc also has guide_star_ra; an unqualified name is ambiguous.
    assert not re.search(r"(?<!\.)guide_star_ra\b", config.sql)
    # Only the visit's own config, not every earlier one too.
    assert "pfs_config.visit0 = pfs_config_sps.visit0" in config.sql


def testReadPfsDesign(FakeOpDB):
    design = pd.DataFrame({"pfs_design_id": [1], "design_name": ["test"]})
    opdb = FakeOpDB({"design": design}, patterns={"design": "FROM pfs_visit"})

    pd.testing.assert_frame_equal(readPfsDesign(opdb, 120000), design)
    assert opdb.calls[0].params == {"pfs_visit_id": 120000}
    checkBound(opdb.calls[0], 120000)


def testReadTelStatus(FakeOpDB):
    telStatus = pd.DataFrame({"pfs_visit_id": [1, 1, 2], "status_sequence_id": [1, 2, 1]})
    opdb = FakeOpDB({"tel_status": telStatus}, patterns={"tel_status": "FROM tel_status"})

    assert len(readTelStatus(opdb, 1)) == 2
    assert len(readTelStatus(opdb, np.array([1, 2]))) == 3
    assert [call.params for call in opdb.calls] == [{"visits": [1]}, {"visits": [1, 2]}]


def testReadSpSInfo(FakeOpDB):
    # Two cameras per visit, whose start times differ slightly.
    t0 = pd.Timestamp("2025-05-01T10:00:00")
    rows = pd.DataFrame(
        {
            "pfs_visit_id": [1, 1, 2, 2],
            "taken_at": [t0, t0 + pd.Timedelta(2, "s"), t0 + pd.Timedelta(1, "h"), t0 + pd.Timedelta(1, "h")],
            "exptime": [900.0, 900.0, 450.0, 450.0],
            "exp_type": "object",
            "altitude": [60.0, 62.0, 70.0, 70.0],
            "azimuth": 10.0,
            "insrot": -30.0,
            "design_name": "test",
            "group_id": 7,
            "group_name": "group",
        }
    )
    opdb = FakeOpDB({"sps": lambda params: rows}, patterns={"sps": "FROM sps_exposure"})

    visits = readSpSInfo(opdb)
    assert list(visits.pfs_visit_id) == [1, 2]
    assert visits.taken_at[0] == t0 + pd.Timedelta(1, "s")
    np.testing.assert_allclose(visits.altitude, [61.0, 70.0])

    call = opdb.calls[0]
    assert call.params == {"exp_type": "object"}
    assert "LIMIT" not in call.sql
    assert "time_exp_start >" not in call.sql

    readSpSInfo(opdb, taken_after="2025-05-01", min_exptime=100, exp_type=None, limit=10, windowed=True)
    call = opdb.calls[1]
    assert call.params == {"taken_after": "2025-05-01", "min_exptime": 100.0, "limit": 10}
    assert "LIMIT :limit" in call.sql
    assert "exp_type =" not in call.sql
    assert "LIKE '%windowed'" in call.sql
    checkBound(call, 100, 10)
