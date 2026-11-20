r"""Remake the fixture of frames the gate could not type: stored numbers of Run31 engineering visits.

``untypedFrames-run31.parquet`` holds the ``image_quality_qa`` rows, and ``untypedLines-run31.parquet`` the
trace offsets from the calibration (``dm_line_residual_qa_metric``), of a few visits of each kind of frame
that had no frame type in Run31: darks, a fiber-profile sequence, a dot-roach sequence and the engineering
exposures taken with ``iic expose``. Each row has its sequence type from the opdb (``seqType``), which the
tasks did not store then. Every visit is a calibration or engineering exposure.

The inputs are one read of the summit qadb for Run31 (``run31Iq.csv.gz`` and ``run31Lines.csv.gz``, the
columns of ``reports/PIPE2D-2012/readRun31.sh``) and the opdb's ``visit_set`` and ``iic_sequence``.

    PYTHONPATH=<pfs_utils>/python uv run --no-project --with pandas --with pyarrow --with sqlalchemy \
        --with 'psycopg[binary]' python makeUntypedFixture.py [DIR] [OPDB_PORT]

``DIR`` holds the two files, default ``~/Projects/Subaru/PFS/reports/PIPE2D-2012``; the opdb is read on
127.0.0.1 at ``OPDB_PORT``, default 5435 (a local copy), as ``public_user``.
"""

import sys
from pathlib import Path

import pandas as pd

from pfs.utils.database.opdb import OpDB

#: The visits kept: sequence type -> visits. One sequence of each kind that has several exposures, so what
#: changes within it by design is in the fixture; the darks and engineering exposures include ones the
#: limits in the task configs flagged.
VISITS = {
    "darks": [153140, 153972, 155362],
    "fiberProfiles": list(range(153005, 153012)),
    "dotRoachInit": [152975],
    "dotRoach": [152977, 152989, 153001],
    "arcs": [153160, 155208],
    "arcs_binned": [155364, 155382],
    "flats": [154575, 154577],
    "flats_binned": [155361],
}

source = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Projects/Subaru/PFS/reports/PIPE2D-2012").expanduser()
port = int(sys.argv[2]) if len(sys.argv) > 2 else 5435
here = Path(__file__).parent
visits = sorted(visit for group in VISITS.values() for visit in group)
key = ["pfs_visit_id", "camera_name"]

opdb = OpDB(host="127.0.0.1", port=port, dbname="opdb", user="public_user")
sequences = opdb.query_dataframe(
    "SELECT vs.pfs_visit_id, s.sequence_type FROM visit_set vs JOIN iic_sequence s USING (iic_sequence_id)"
    " WHERE vs.pfs_visit_id = ANY(:visits)",
    params={"visits": visits},
)
wanted = {visit: kind for kind, group in VISITS.items() for visit in group}
found = dict(zip(sequences["pfs_visit_id"], sequences["sequence_type"], strict=True))
if found != wanted:
    raise SystemExit(f"The opdb's sequence types differ from VISITS: {found}")


def latest(table: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of each image's latest processing."""
    table = table[table["pfs_visit_id"].isin(visits)]
    return table[table["processing_id"] == table.groupby(key)["processing_id"].transform("max")]


iq = latest(pd.read_csv(source / "run31Iq.csv.gz")).drop_duplicates(key).merge(sequences, on="pfs_visit_id")
frames = pd.DataFrame(
    {
        "visit": iq["pfs_visit_id"].astype("int64"),
        "arm": iq["camera_name"].str[0],
        "spectrograph": iq["camera_name"].str[1:].astype("int64"),
        "obsType": iq["obs_type"],
        "seqType": iq["sequence_type"],
        "seqName": iq["seq_name"],
        "illumination": iq["illumination"],
        "traceOnly": iq["trace_only"].isin([True, "t"]),
        "medFwhm": iq["med_fwhm"].astype("float64"),
        "medDxCenter": iq["med_dx_center"].astype("float64"),
        "linesPerFiber": iq["lines_per_fiber"].astype("float64"),
        "cameraSet": iq["camera_set"],
    }
).sort_values(["visit", "arm", "spectrograph"], ignore_index=True)
frames.to_parquet(here / "untypedFrames-run31.parquet", index=False)

lines = latest(pd.read_csv(source / "run31Lines.csv.gz"))
lines = lines[
    (lines["reference"] == "calib") & (lines["description"] == "Trace") & (lines["metric"] == "medianDeltaX")
].merge(iq[[*key, "obs_type", "sequence_type", "camera_set"]], on=key)
offsets = pd.DataFrame(
    {
        "visit": lines["pfs_visit_id"].astype("int64"),
        "arm": lines["camera_name"].str[0],
        "spectrograph": lines["camera_name"].str[1:].astype("int64"),
        "cameraSet": lines["camera_set"],
        "obsType": lines["obs_type"],
        "seqType": lines["sequence_type"],
        "reference": lines["reference"],
        "description": lines["description"],
        "statusType": lines["status_type"],
        "metric": lines["metric"],
        "value": lines["value"].astype("float64"),
    }
).sort_values(["visit", "arm", "spectrograph"], ignore_index=True)
offsets.to_parquet(here / "untypedLines-run31.parquet", index=False)
print(f"untypedFrames-run31.parquet: {len(frames)} images of {frames['visit'].nunique()} visits")
print(frames.groupby(["seqType", "illumination"]).size().to_string())
print(f"untypedLines-run31.parquet: {len(offsets)} trace offsets")
