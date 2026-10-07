#!/usr/bin/env python
r"""Make the real quartz rows that tests/imageQualityQa/test_crossDispersion.py reads.

Each fixture is a few rows of one detector's calexp from an engineering quartz
(``scienceTrace``) visit, with what the cross-dispersion width measurement needs
beside them: the visit's detector-map centers and wavelengths, and the trace
widths of the ``fiberProfiles`` calibration used for that visit. See `FIXTURES`.

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

HERE = Path(__file__).parent

#: Rows of the detector to keep: every 11th row of the grid ``imageQualityQa``
#: samples at its default ``profileYStride`` of 50, so eight rows.
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
    (``calculateStatistics().width``) and the Gaussian fit of
    `pfs.drp.qa.crossDispersion.profileWidth`. Masked swaths are left out.
    Fibers without a profile are NaN.
    """
    from pfs.drp.qa.crossDispersion import profileWidth

    moment = np.full(len(fiberIds), np.nan)
    fit = np.full(len(fiberIds), np.nan)
    for ii, fiberId in enumerate(fiberIds):
        if fiberId not in fiberProfiles:
            continue
        profile = fiberProfiles[fiberId]
        width = np.ma.masked_invalid(np.ma.asarray(profile.calculateStatistics().width))
        if width.count() > 0:
            moment[ii] = float(np.ma.median(width))
        sigma = np.ma.masked_invalid(profileWidth(profile.index, profile.profiles)["sigma"])
        if sigma.count() > 0:
            fit[ii] = float(np.ma.median(sigma))
    return {"calibMomentSigma": moment, "calibFitSigma": fit}


def export(butler, visit: int, arm: str, spectrograph: int) -> Path:
    """Write one fixture and return its path."""
    dataId = {"instrument": "PFS", "visit": visit, "arm": arm, "spectrograph": spectrograph}
    calexp = butler.get("calexp", dataId)
    detectorMap = butler.get("detectorMap", dataId)
    fiberProfiles = butler.get("fiberProfiles", dataId)
    profilesRef = butler.find_dataset("fiberProfiles", dataId)

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
        calibRun=str(profilesRef.run if profilesRef is not None else ""),
        visit=visit,
        arm=arm,
        spectrograph=spectrograph,
    )
    print(f"{path.name}: {len(rows)} rows, {len(fiberIds)} fibers, {path.stat().st_size / 1e3:.0f} kB")
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
