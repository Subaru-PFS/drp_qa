#!/usr/bin/env python
"""Make the real AG data that tests/guiders reads.

The fixtures are AG data from engineering visits of Run 30 (2026-08-31), read
with ``pfs.drp.qa.guiders.queries`` and trimmed to a few guide stars per
camera. Science visits are proprietary and must not be used.

Two steps:

``dump``
    Read the AG data of every visit in `DUMP_VISITS`, and the guide stars of
    `STAR_VISITS`, from the opdb; write them to ``--dumps``.
``trim``
    Trim those dumps to the fixtures in this directory (see `FIXTURES`).

Run from the top of drp_qa with ``python`` and pfs_utils on ``PYTHONPATH``, e.g.::

    python tests/guiders/data/makeGuiderFixtures.py dump --dumps ~/Projects/Subaru/PFS/data/opdb
    python tests/guiders/data/makeGuiderFixtures.py trim --dumps ~/Projects/Subaru/PFS/data/opdb

The dump reads about 180,000 rows (22 MB): the clusters observed have hundreds
of guide stars.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

# Every engineering visit of Run 30 that might make a fixture: the all-sky
# exposures (148258-148259, 900 s), a focus sweep (148264, 148266-148282, 60 s),
# two raster scans (148284-148290, 148292-148299, 60 s), each followed by a
# flux uniformity exposure (148291, 148300, 200 s).
DUMP_VISITS = [148258, 148259, 148264, *range(148266, 148283), *range(148284, 148301)]

# One visit of each design, for readAGCStars.
STAR_VISITS = [148258, 148264, 148266, 148291]

AGC_DATA_DUMP = "agcData-run30-eng-148258-148300.parquet"
AGC_STARS_DUMP = "agcStars-run30-eng-148258-148291.parquet"

# The fixtures, agcData-<name>.parquet: their visits, and which guide stars
# they keep: on each camera (or each half of each camera's detector, if
# ``byHalf``), the ``nStars`` guide stars seen in the most AG exposures; and
# the ``nBad`` others with the most invalid matches.
FIXTURES = {
    # M2_OFF3 from 3.25 to 2.725 mm in 148266, then 2.95 to 3.55 mm in 0.075 mm steps, changing within
    # each visit: estimateFocusErrors, the AG actor's guide_delta_z, and the pairing of tel_status rows.
    "focusSweep": {"visits": [148266, *range(148270, 148278)], "nStars": 1, "byHalf": True, "nBad": 0},
    # A raster scan of seven positions (the telescope moves to the next one after the shutters close),
    # the flux uniformity exposure, and the start of the second scan, which has no SpS exposure
    # (shutter_open 2): the boresight reference and fitGuiderModel.
    "raster": {"visits": list(range(148284, 148293)), "nStars": 2, "byHalf": False, "nBad": 2},
    # One all-sky exposure, 900 s: drift, and comparePfsUtilsPositions.
    "allSky": {"visits": [148258], "nStars": 2, "byHalf": False, "nBad": 0},
}

# The guide stars of these visits' designs, from readAGCStars, for the stars in the fixtures.
AGC_STARS_VISITS = {"allSky": 148258, "raster": 148291}


def dump(dumps: Path, host: str, user: str) -> None:
    """Read the full dumps from the opdb."""
    from pfs.drp.qa.guiders.queries import readAgcData, readAGCStars, readPfsDesign
    from pfs.utils.database.opdb import OpDB

    opdb = OpDB(host=host, user=user)
    agcData = readAgcData(opdb, DUMP_VISITS)
    agcData.to_parquet(dumps / AGC_DATA_DUMP)
    print(f"{AGC_DATA_DUMP}: {len(agcData)} rows of {agcData.pfs_visit_id.nunique()} visits")

    stars = []
    for visit in STAR_VISITS:
        design = readPfsDesign(opdb, visit)
        if design.empty:
            print(f"No design for pfs_visit_id {visit}")
            continue
        stars.append(readAGCStars(opdb, int(design.pfs_design_id.iloc[0]), visit))
    agcStars = pd.concat(stars, ignore_index=True)
    agcStars.to_parquet(dumps / AGC_STARS_DUMP)
    print(f"{AGC_STARS_DUMP}: {len(agcStars)} rows of {agcStars.pfs_visit_id.nunique()} visits")


def pickGuideStars(agcData: pd.DataFrame, nStars: int, byHalf: bool, nBad: int) -> np.ndarray:
    """Return the IDs of the guide stars a fixture keeps; see `FIXTURES`."""
    data = agcData.assign(
        right=(agcData.agc_data_flags & 1) != 0,  # SourceDetectionFlags.RIGHT
        bad=agcData.agc_match_flags != 1,  # not GOOD_MATCH alone
    )
    keys = ["agc_camera_id", "right"] if byHalf else ["agc_camera_id"]
    counts = data.groupby([*keys, "guide_star_id"], as_index=False).agg(
        nExp=("agc_exposure_id", "nunique"), nRow=("agc_exposure_id", "size"), nBad=("bad", "sum")
    )
    # Ties go to fewer rows (fewer spots matched to one star), then to the lower ID.
    counts = counts.sort_values(["nExp", "nRow", "guide_star_id"], ascending=[False, True, True])
    keep = counts.groupby(keys).head(nStars)
    rest = counts.drop(keep.index).sort_values(["nBad", "guide_star_id"], ascending=[False, True])
    bad = rest[rest.nBad > 0].head(nBad)

    return np.union1d(keep.guide_star_id, bad.guide_star_id)


def trim(dumps: Path) -> None:
    """Trim the full dumps to the fixtures."""
    agcData = pd.read_parquet(dumps / AGC_DATA_DUMP)
    agcStars = pd.read_parquet(dumps / AGC_STARS_DUMP)

    stars = []
    for name, spec in FIXTURES.items():
        data = agcData[agcData.pfs_visit_id.isin(spec["visits"])]
        guideStars = pickGuideStars(data, spec["nStars"], spec["byHalf"], spec["nBad"])
        fixture = data[data.guide_star_id.isin(guideStars)].reset_index(drop=True)
        fixture.attrs = dict(agcData.attrs)  # the frame
        path = HERE / f"agcData-{name}.parquet"
        fixture.to_parquet(path, compression="zstd")
        nValid = (fixture.agc_match_flags == 1).sum()
        print(
            f"{path.name}: {len(fixture)} rows ({nValid} valid matches), {len(guideStars)} guide stars, "
            f"{fixture.agc_exposure_id.nunique()} AG exposures, {path.stat().st_size / 1e3:.0f} kB"
        )
        if name in AGC_STARS_VISITS:
            visit = AGC_STARS_VISITS[name]
            stars.append(agcStars[(agcStars.pfs_visit_id == visit) & agcStars.guide_star_id.isin(guideStars)])

    path = HERE / "agcStars.parquet"
    pd.concat(stars, ignore_index=True).to_parquet(path, compression="zstd")
    print(f"{path.name}: {sum(map(len, stars))} rows, {path.stat().st_size / 1e3:.0f} kB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["dump", "trim"])
    parser.add_argument("--dumps", type=Path, required=True, help="Directory of the full dumps")
    parser.add_argument("--host", default="pfsa-db", help="opdb host (dump)")
    parser.add_argument("--user", default="public_user", help="opdb user (dump); a read-only one")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.step == "dump":
        dump(args.dumps.expanduser(), args.host, args.user)
    else:
        trim(args.dumps.expanduser())


if __name__ == "__main__":
    main()
