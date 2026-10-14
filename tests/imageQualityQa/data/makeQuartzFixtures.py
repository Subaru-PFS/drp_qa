#!/usr/bin/env python
r"""Make the real quartz rows that tests/imageQualityQa/test_crossDispersion.py reads.

Each fixture is a few rows of one detector's calexp from an engineering quartz
(``scienceTrace``) visit, with what the cross-dispersion width measurement needs
beside them: the visit's detector-map centers and wavelengths, and the trace
widths and a few oversampled profiles of the ``fiberProfiles`` calibration the
reduction used, read from the calexp's Butler provenance
(`pfs.drp.qa.utils.provenance`) with its run, dataset ID and the quartz visit it
was built from. See `FIXTURES`.

Reads the Butler only (``writeable=False``). Needs the LSST stack with
drp_stella and drp_qa set up. Run from the top of drp_qa, e.g.::

    python tests/imageQualityQa/data/makeQuartzFixtures.py \
        --butler /work/datastore --collections u/$USER/qa-thresholds/004,PFS/defaults

The collections must hold the visits' ``calexp`` and ``detectorMap`` (a
``docs/qa-thresholds.ipynb`` run does) and the calibrations. Each file is about
0.3 MB.
"""

import argparse
from pathlib import Path

import numpy as np

from pfs.drp.qa.utils.provenance import calibrationsUsed

HERE = Path(__file__).parent

#: Rows of the detector to keep: every 11th row of a 50-row grid starting at row
#: 25, so eight rows. The grid is the fixtures' own, fixed so that remaking them
#: keeps the same rows; it does not follow ``imageQualityQa``'s ``profileYStride``.
ROW_STRIDE = 50
ROW_STEP = 11

#: Mask planes ``imageQualityQa`` ignores in the calexp.
BAD_PLANES = ["BAD", "SAT", "CR", "NO_DATA"]

#: The fixtures, quartz-<visit>-<arm><spectrograph>.npz.
FIXTURES = [
    # Run25 known-good: b2 and r2 carry the earlier comparison (r2 reads wider than its calib, b2 does not),
    # r1 is the same detector as the defocused one below.
    {"visit": 133040, "arm": "b", "spectrograph": 2},
    {"visit": 133040, "arm": "r", "spectrograph": 2},
    {"visit": 133040, "arm": "r", "spectrograph": 1},
    # Run27 known-bad: SM1 defocused, expect medFwhm FAIL.
    {"visit": 140032, "arm": "r", "spectrograph": 1},
]


def calibWidths(fiberProfiles, fiberIds: np.ndarray) -> dict[str, np.ndarray]:
    """Return each fiber's calibration trace width, median over swaths.

    Two readings of the same profiles: drp_stella's second moment
    (``calculateStatistics().width``, a sigma) and the FWHM at half maximum
    (`pfs.drp.qa.crossDispersion.profileFwhm`). Masked swaths are left out.
    Fibers without a profile are NaN.
    """
    from pfs.drp.qa.crossDispersion import profileFwhm

    moment = np.full(len(fiberIds), np.nan)
    halfMax = np.full(len(fiberIds), np.nan)
    for ii, fiberId in enumerate(fiberIds):
        if fiberId not in fiberProfiles:
            continue
        profile = fiberProfiles[fiberId]
        width = np.ma.masked_invalid(np.ma.asarray(profile.calculateStatistics().width))
        if width.count() > 0:
            moment[ii] = float(np.ma.median(width))
        fwhm = np.ma.masked_invalid(profileFwhm(profile.index, profile.profiles))
        if fwhm.count() > 0:
            halfMax[ii] = float(np.ma.median(fwhm))
    return {"calibMomentSigma": moment, "calibHalfMaxFwhm": halfMax}


#: Fibers whose calibration profile is kept, evenly spaced across the detector.
N_PROFILE_FIBERS = 40


def calibProfiles(fiberProfiles, fiberIds: np.ndarray, height: int) -> dict[str, np.ndarray]:
    """Return the calibration's oversampled profiles of a few fibers, in the swath nearest mid-detector.

    The profile shape itself, so the width estimators can be run on synthetic
    rows built from it. Masked samples are NaN. Fibers the calibration had no
    light for carry drp_stella's placeholder Gaussian (FWHM 1.5 px).
    """
    present = np.array([ff for ff in fiberIds if ff in fiberProfiles])
    chosen = present[np.linspace(0, len(present) - 1, N_PROFILE_FIBERS).round().astype(int)]
    profiles, rows = [], []
    for fiberId in chosen:
        profile = fiberProfiles[fiberId]
        swath = int(np.argmin(np.abs(np.asarray(profile.rows) - height / 2)))
        profiles.append(np.ma.filled(np.ma.asarray(profile.profiles[swath], dtype=np.float32), np.nan))
        rows.append(profile.rows[swath])
    return {
        "calibProfileIndex": np.asarray(fiberProfiles[chosen[0]].index, dtype=np.float32),
        "calibProfiles": np.array(profiles),
        "calibProfileFiberIds": chosen.astype(np.int32),
        "calibProfileRows": np.array(rows, dtype=np.float32),
    }


def export(butler, visit: int, arm: str, spectrograph: int) -> Path:
    """Write one fixture and return its path."""
    dataId = {"instrument": "PFS", "visit": visit, "arm": arm, "spectrograph": spectrograph}
    calexp = butler.get("calexp", dataId)
    detectorMap = butler.get("detectorMap", dataId)
    calibs = calibrationsUsed(butler, dataId)
    calib = calibs[calibs.datasetType == "fiberProfiles"].iloc[0]
    fiberProfiles = butler.get(butler.get_dataset(calib.id))

    height = calexp.getHeight()
    rows = np.arange(ROW_STRIDE // 2, height, ROW_STRIDE)[::ROW_STEP]
    fiberIds = np.asarray(detectorMap.fiberId, dtype=np.int32)
    xCenters = np.empty((len(rows), len(fiberIds)))
    wavelengths = np.empty((len(rows), len(fiberIds)), dtype=np.float32)
    for ii, row in enumerate(rows):
        yy = np.full(len(fiberIds), float(row))
        xCenters[ii] = detectorMap.getXCenter(fiberIds, yy)
        wavelengths[ii] = detectorMap.findWavelength(fiberIds, yy)

    badBits = calexp.mask.getPlaneBitMask(BAD_PLANES)
    path = HERE / f"quartz-{visit}-{arm}{spectrograph}.npz"
    np.savez_compressed(
        path,
        image=calexp.image.array[rows].astype(np.float32),
        variance=calexp.variance.array[rows].astype(np.float32),
        bad=(calexp.mask.array[rows] & badBits) != 0,
        rows=rows,
        fiberIds=fiberIds,
        xCenters=xCenters,
        wavelengths=wavelengths,
        **calibWidths(fiberProfiles, fiberIds),
        **calibProfiles(fiberProfiles, fiberIds, height),
        calibRun=calib.run,
        calibId=calib.id,
        calibSource=calib.source,  # "provenance", or "lookup" when the calexp recorded none
        calibVisit0=fiberProfiles.identity.visit0,
        calibObsDate=str(fiberProfiles.identity.obsDate),
        visit=visit,
        arm=arm,
        spectrograph=spectrograph,
    )
    print(
        f"{path.name}: {len(rows)} rows, {len(fiberIds)} fibers, {path.stat().st_size / 1e3:.0f} kB;"
        f" fiberProfiles {calib.run} {calib.id} ({calib.source})"
    )
    return path


def main() -> None:
    from lsst.daf.butler import Butler

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument("--collections", required=True, help="Comma-separated input collections")
    args = parser.parse_args()

    butler = Butler(args.butler, collections=args.collections.split(","), writeable=False)
    for fixture in FIXTURES:
        export(butler, **fixture)


if __name__ == "__main__":
    main()
