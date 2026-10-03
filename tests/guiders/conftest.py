"""Shared fixtures for the ``pfs.drp.qa.guiders`` tests.

These tests need numpy, pandas, matplotlib and pfs_utils but not the LSST
stack. CI runs them in their own job (``guiders`` in
``.github/workflows/tests.yml``); the standard-library job ignores this
directory.

Test modules can't import each other (``--import-mode=importlib``), so the
test doubles here (`FakeOpDB`, `StubButler`) are handed out by fixtures, as is
the real AG data in `data/` (`realAgcData`, `realAgcStars`).
"""

import functools
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders.coordinates import AGC_CAMERA_CENTERS_MM, OPDB_Y_COLUMNS
from pfs.utils.datamodel.ag import SourceCatalogFlags

GAIA_ISOLATED = SourceCatalogFlags.GAIA | SourceCatalogFlags.NON_BINARY
NOISE = 2e-3  # Centroiding noise (mm)
SEED = 12345


def makeAgcData(
    rng: np.random.RandomState,
    nVisit: int = 3,
    nExp: int = 8,
    nStar: int = 5,
    closedFrac: float = 0.2,
    stars: bool = False,
) -> pd.DataFrame:
    """Make synthetic AG data.

    Positions are in hardware coordinates, as the readers return them. Each
    agc_exposure has a random offset (~30 microns) and rotation (~1 arcmin)
    of the guide stars' centroids relative to their nominal positions, plus
    centroiding noise. The rows are shuffled (the opdb doesn't guarantee
    an order) and the per-visit frames are concatenated without resetting the
    index, as older versions of the readers did.

    Ported from ``makeAgcData`` in drp_stella's ``tests/test_guiders.py``
    (branch ``tickets/guiders-cleanup-reference``). The random draws are made
    in the same order, so the same seed gives the same data. Unlike the
    original it doesn't add the derived ``rms``, ``FWHM`` and ``left``
    columns when ``stars=True``: `pfs.drp.qa.guiders.queries.readAgcData`
    doesn't return them. Its columns are all in
    `pfs.drp.qa.guiders.queries.AGC_DATA_COLUMNS`.

    Parameters
    ----------
    rng : `numpy.random.RandomState`
        Random number generator.
    nVisit, nExp, nStar : `int`
        Number of visits, agc_exposures per visit, and guide stars per camera.
    closedFrac : `float`
        Probability that the spectrograph shutters are closed for an exposure.
    stars : `bool`
        Add the columns of ``readAGCStarsForVisitSetByPfsVisitId`` (used for
        focus plots) to those of ``readAgcDataFromOpdb``.

    Returns
    -------
    agcData : `pandas.DataFrame`
        One row per (agc_exposure, guide star), with duplicate index labels
        when ``nVisit > 1``.
    """
    nominal = {}
    for cid in range(6):
        xc, yc = AGC_CAMERA_CENTERS_MM[cid]
        for ss in range(nStar):
            nominal[1000 * cid + ss] = (cid, xc + rng.uniform(-5, 5), yc + rng.uniform(-5, 5))

    frames = []
    aid = 1000
    t0 = pd.Timestamp("2025-05-01T10:00:00")
    for vv in range(nVisit):
        rows = []
        for ee in range(nExp):
            aid += 1
            theta = np.deg2rad(rng.normal(0, 1 / 60))
            x0, y0 = rng.normal(0, 0.03, 2)
            shutter = int(rng.uniform() > closedFrac)
            cc, ss = np.cos(theta), np.sin(theta)
            for gsid, (cid, xn, yn) in nominal.items():
                row = {
                    "pfs_visit_id": 120000 + vv,
                    "agc_exposure_id": aid,
                    "guide_star_id": gsid,
                    "taken_at": t0 + np.timedelta64(10 * aid, "s"),
                    "altitude": 60.0 + vv,
                    "azimuth": 10.0 + ee,
                    "insrot": -30.0 + vv,
                    "exptime": 900.0,
                    "m2_pos3": 0.1 * vv,
                    "agc_camera_id": cid,
                    "agc_nominal_x_mm": xn,
                    "agc_nominal_y_mm": yn,
                    "agc_center_x_mm": x0 + cc * xn - ss * yn + rng.normal(0, NOISE),
                    "agc_center_y_mm": y0 + ss * xn + cc * yn + rng.normal(0, NOISE),
                    "agc_match_flags": 1,
                    "agc_data_flags": int(rng.uniform() < 0.5),
                    "shutter_open": shutter,
                    "guide_delta_insrot": rng.normal(0, 3),
                    "guide_delta_azimuth": rng.normal(0, 0.5),
                    "guide_delta_altitude": rng.normal(0, 0.5),
                }
                if stars:
                    mxx = rng.uniform(2, 4)
                    row.update(
                        guide_star_flag=int(GAIA_ISOLATED) if gsid % 5 else int(SourceCatalogFlags.HSC),
                        m2_off3=-0.3 + 0.05 * (ee % 4),
                        centroid_y_pix=rng.uniform(0, 1000),
                        mxx=mxx,
                        myy=mxx * rng.uniform(0.9, 1.1),
                        mxy=0.1,
                        estimated_magnitude=15.0,
                        guide_delta_z=rng.normal(0, 0.02),
                        **{f"guide_delta_z{ii}": rng.normal(0, 0.02) for ii in range(1, 7)},
                    )
                rows.append(row)
        frames.append(pd.DataFrame(rows).sample(frac=1, random_state=rng))

    return pd.concat(frames)


@pytest.fixture
def rng() -> np.random.RandomState:
    """Return a freshly seeded random number generator."""
    return np.random.RandomState(SEED)


@pytest.fixture(name="makeAgcData")
def makeAgcDataFixture(rng):
    """Return `makeAgcData` bound to the test's ``rng``.

    Call it with `makeAgcData`'s keyword arguments, e.g.
    ``makeAgcData(nVisit=1, stars=True)``.
    """

    def make(**kwargs) -> pd.DataFrame:
        return makeAgcData(rng, **kwargs)

    return make


# The opdb tables behind each readAgcData query, as FakeOpDB looks them up.
AGC_QUERIES = {
    "agc_data": r"FROM agc_exposure\s+JOIN agc_data",
    "agc_exposure": r"FROM agc_exposure\s+WHERE",
    "agc_tel_status": r"FROM tel_status\s+WHERE.*caller = 'agcc'",
}


class FakeOpDB:
    """Stand-in for `pfs.utils.database.opdb.OpDB`.

    Answers each query from the first table whose pattern matches the SQL.
    When the query binds ``visits``, only the table's rows with those
    ``pfs_visit_id`` values are returned, in a random order (SQL guarantees
    none without ORDER BY). Each call is recorded in ``calls``.

    Parameters
    ----------
    tables : `dict` [`str`, `pandas.DataFrame` or callable]
        Tables (or functions of the bound parameters returning one), by name.
    patterns : `dict` [`str`, `str`]
        A regular expression for each name, matched against the SQL.
    """

    def __init__(self, tables, patterns=AGC_QUERIES):
        self.tables = dict(tables)
        self.patterns = patterns
        self.calls = []
        self._rng = np.random.RandomState(0)

    def query_dataframe(self, sql, /, *, params=None, conn=None):
        self.calls.append(SimpleNamespace(sql=sql, params=params))
        for name, pattern in self.patterns.items():
            if re.search(pattern, sql, re.DOTALL):
                table = self.tables[name]
                if callable(table):
                    return table(params)
                if params and "visits" in params:
                    table = table[table.pfs_visit_id.isin(params["visits"])]
                    table = table.sample(frac=1, random_state=self._rng)
                return table.reset_index(drop=True)

        raise AssertionError(f"Unexpected query:\n{sql}")


def makeOpdbTables(agcData: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Make the opdb's answers to readAgcData's queries for some AG data.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        Synthetic AG data from ``makeAgcData(stars=True)``, in hardware
        coordinates.

    Returns
    -------
    tables : `dict` [`str`, `pandas.DataFrame`]
        For each of `AGC_QUERIES`, the rows the opdb would return: positions
        in the opdb's frame, and ``m2_off3`` in tel_status, not agc_data.
    """
    agcData = agcData.sort_values(["agc_exposure_id", "guide_star_id"], ignore_index=True)
    n = len(agcData)

    rows = agcData.drop(columns="m2_off3")
    for column in OPDB_Y_COLUMNS:
        rows[column] = -rows[column]
    rows["spot_id"] = rows.guide_star_id % 1000
    rows["agc_exptime"] = 2.0
    rows["adc_pa"] = 0.5
    rows["image_moment_00_pix"] = 1e4
    rows["centroid_x_pix"] = np.linspace(0, 1000, n)
    rows["peak_pixel_x_pix"] = rows.centroid_x_pix.round().astype(int)
    rows["peak_pixel_y_pix"] = rows.centroid_y_pix.round().astype(int)
    rows["peak_intensity"] = 500.0
    rows["background"] = 10.0
    rows["guide_ra"] = 150.0
    rows["guide_dec"] = 2.0
    rows["guide_pa"] = -90.0
    rows["guide_delta_ra"] = 0.1
    rows["guide_delta_dec"] = -0.1
    rows["guide_delta_scale"] = 0.0

    exposures = agcData.groupby("agc_exposure_id", as_index=False).agg(
        pfs_visit_id=("pfs_visit_id", "first"), m2_pos3=("m2_pos3", "first"), m2_off3=("m2_off3", "first")
    )
    telStatus = pd.DataFrame(
        {
            "pfs_visit_id": exposures.pfs_visit_id,
            "status_sequence_id": 10 * exposures.agc_exposure_id,
            "m2_off3": exposures.m2_off3,
            "tel_ra": 150.0 + 1e-4 * exposures.agc_exposure_id,
            "tel_dec": 2.0,
        }
    )

    return {
        "agc_data": rows,
        "agc_exposure": exposures[["pfs_visit_id", "agc_exposure_id", "m2_pos3"]],
        "agc_tel_status": telStatus,
    }


@pytest.fixture(name="makeOpdb")
def makeOpdbFixture(makeAgcData):
    """Return a function making a `FakeOpDB` with synthetic AG data.

    It takes `makeAgcData`'s keyword arguments (``stars`` is always set) and
    returns the `FakeOpDB` and the AG data it holds, in hardware
    coordinates. The tables are in ``opdb.tables``, to edit.
    """

    def make(**kwargs) -> tuple[FakeOpDB, pd.DataFrame]:
        agcData = makeAgcData(stars=True, **kwargs)
        return FakeOpDB(makeOpdbTables(agcData)), agcData

    return make


@pytest.fixture(name="FakeOpDB")
def fakeOpDBFixture():
    """Return the `FakeOpDB` class."""
    return FakeOpDB


class StubButler:
    """Stand-in for a Gen3 butler holding some raws.

    Only the Gen3 calls the readers make are implemented, so a Gen2 call
    (``butler.get("raw_md", ...)``) fails. Each raw's header has
    ``INST-PA = 10 * visit``, plus 1 for the r arm, and
    ``W_M2OFF3 = -visit / 1e6``.

    Parameters
    ----------
    raws : `list` [`dict`]
        Data IDs of the raws.
    """

    def __init__(self, raws):
        self.raws = raws
        self.registry = self

    def queryDatasets(self, datasetType, **where):
        assert datasetType == "raw"
        return [
            SimpleNamespace(dataId=dataId)
            for dataId in self.raws
            if all(dataId.get(key) == value for key, value in where.items())
        ]

    def get(self, datasetType, dataId):
        assert datasetType == "raw.metadata"
        return {
            "INST-PA": 10.0 * dataId["visit"] + (dataId["arm"] == "r"),
            "W_M2OFF3": -dataId["visit"] / 1e6,
        }


@pytest.fixture(name="StubButler")
def stubButlerFixture():
    """Return the `StubButler` class."""
    return StubButler


# Real AG data, made by data/makeGuiderFixtures.py.
DATA_DIR = Path(__file__).parent / "data"


@functools.cache
def _readParquet(filename: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / filename)


@pytest.fixture(name="realAgcData")
def realAgcDataFixture():
    """Return a function reading real AG data from ``data/``.

    ``realAgcData(name)`` returns a fresh copy of ``agcData-<name>.parquet``:
    the output of `pfs.drp.qa.guiders.queries.readAgcData`, in hardware
    coordinates, for a few guide stars per camera. The names are
    ``focusSweep``, ``raster`` and ``allSky``; ``data/makeGuiderFixtures.py``
    says what each holds.
    """

    def read(name: str) -> pd.DataFrame:
        return _readParquet(f"agcData-{name}.parquet").copy()

    return read


@pytest.fixture(name="realAgcStars")
def realAgcStarsFixture():
    """Return a function reading a visit's guide stars from ``data/``.

    ``realAgcStars(visit)`` returns `pfs.drp.qa.guiders.queries.readAGCStars`
    for the visit (148258 or 148291), for the stars in `realAgcData`.
    """

    def read(visit: int) -> pd.DataFrame:
        agcStars = _readParquet("agcStars.parquet")
        return agcStars[agcStars.pfs_visit_id == visit].reset_index(drop=True)

    return read
