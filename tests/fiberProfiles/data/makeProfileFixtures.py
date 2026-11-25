"""Export the real ``fiberProfiles`` fixtures of ``tests/fiberProfiles``.

Needs the LSST stack and the Butler on pfsa-usr02; run it there and copy the
files back:

    python tests/fiberProfiles/data/makeProfileFixtures.py /work/datastore

Each fixture is one detector's calibration as ``PFS/defaults`` resolves it for
an engineering visit of the run, reduced to what the tests need: every fiber's
profile averaged over its swaths (float32), the sample offsets and the fiber
identifiers. A file is about 130 kB.
"""

import sys
from pathlib import Path

import numpy as np

#: The fixtures, profiles-<run>-<arm><spectrograph>.npz, with the visit the calibration is resolved for.
FIXTURES = [
    # Run31: light of fibers that cannot be hidden left in their neighbours' profiles, and copied into
    # replaced ones.
    {"run": "run31", "visit": 156021, "arm": "r", "spectrograph": 4},
    # Run30, the same detector, without the fault.
    {"run": "run30", "visit": 150146, "arm": "r", "spectrograph": 4},
]


def export(butler, run: str, visit: int, arm: str, spectrograph: int) -> Path:
    """Write one fixture and return its path."""
    profiles = butler.get("fiberProfiles", instrument="PFS", visit=visit, arm=arm, spectrograph=spectrograph)
    fiberId = np.array(sorted(profiles.fiberId), dtype=np.int32)
    mean = [np.ma.masked_invalid(profiles[ff].profiles).mean(axis=0).filled(np.nan) for ff in fiberId]
    path = Path(__file__).parent / f"profiles-{run}-{arm}{spectrograph}.npz"
    np.savez_compressed(
        path,
        fiberId=fiberId,
        index=np.asarray(profiles[fiberId[0]].index, dtype=np.float32),
        profile=np.array(mean, dtype=np.float32),
        oversample=np.float32(profiles[fiberId[0]].oversample),
        visit0=np.int32(profiles.identity.visit0),
    )
    return path


if __name__ == "__main__":
    from lsst.daf.butler import Butler

    butler = Butler(sys.argv[1], collections=["PFS/defaults"])
    for fixture in FIXTURES:
        print(export(butler, **fixture), flush=True)
