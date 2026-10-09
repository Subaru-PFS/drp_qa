"""Readers for the opdb: what was taken in a period, and what describes it.

This is the only module in `pfs.drp.qa.comparison` that touches the opdb. Each reader takes a
`pfs.utils.database.opdb.OpDB` as its first argument and binds its parameters rather than
formatting them into the SQL; use a read-only user (``OpDB(host=..., user="public_user")``).
Readers return a DataFrame with a fresh index. Times are HST, as the opdb stores them.
"""

import re
from collections.abc import Iterable

import pandas as pd

__all__ = [
    "LISTING_COLUMNS",
    "proposalCategory",
    "readDesignCategories",
    "readNotes",
    "readTelStatus",
    "readVisitListing",
]

#: The columns of `readVisitListing`.
LISTING_COLUMNS = {
    "pfs_visit_id": "PFS visit",
    "time_exp_start": "start of the earliest SpS exposure of the visit (HST)",
    "exptime": "longest SpS exposure time of the visit (s)",
    "exp_type": "sps_visit.exp_type: arc, flat, object, dark, bias, test, ...",
    "iic_sequence_id": "the IIC sequence the visit belongs to; NA if none",
    "sequence_type": "iic_sequence.sequence_type: scienceArc, scienceTrace, scienceObject, ...",
    "sequence_name": "iic_sequence.name, free text",
    "group_id": "sequence_group.group_id",
    "group_name": "sequence_group.group_name",
    "cmd_str": "the IIC command that made the sequence: lamps, cameras, windows, ...",
    "sequence_comments": "iic_sequence.comments",
    "pfs_design_id": "pfs_visit.pfs_design_id",
    "cameras": "the SpS cameras that took an exposure, comma-separated (b1,b2,...)",
}

# One row per SpS visit with an exposure starting in [start, end). The exposures are
# aggregated per visit, and the sequence is joined left so that a visit outside any
# sequence is still listed.
_LISTING_SQL = """
SELECT
    pfs_visit.pfs_visit_id,
    min(sps_exposure.time_exp_start) AS time_exp_start,
    max(sps_exposure.exptime) AS exptime,
    sps_visit.exp_type,
    visit_set.iic_sequence_id,
    iic_sequence.sequence_type,
    iic_sequence.name AS sequence_name,
    iic_sequence.group_id,
    sequence_group.group_name,
    iic_sequence.cmd_str,
    iic_sequence.comments AS sequence_comments,
    pfs_visit.pfs_design_id,
    string_agg(DISTINCT sps_camera.sps_camera_name, ',' ORDER BY sps_camera.sps_camera_name) AS cameras
FROM pfs_visit
JOIN sps_visit ON sps_visit.pfs_visit_id = pfs_visit.pfs_visit_id
JOIN sps_exposure ON sps_exposure.pfs_visit_id = pfs_visit.pfs_visit_id
JOIN sps_camera ON sps_camera.sps_camera_id = sps_exposure.sps_camera_id
LEFT JOIN visit_set ON visit_set.pfs_visit_id = pfs_visit.pfs_visit_id
LEFT JOIN iic_sequence ON iic_sequence.iic_sequence_id = visit_set.iic_sequence_id
LEFT JOIN sequence_group ON sequence_group.group_id = iic_sequence.group_id
WHERE sps_exposure.time_exp_start >= :start AND sps_exposure.time_exp_start < :end
GROUP BY pfs_visit.pfs_visit_id, sps_visit.exp_type, visit_set.iic_sequence_id,
    iic_sequence.iic_sequence_id, sequence_group.group_name
ORDER BY pfs_visit.pfs_visit_id
"""

_SEQUENCE_NOTES_SQL = """
SELECT iic_sequence_id, body AS note
FROM obslog_visit_set_note
WHERE iic_sequence_id = ANY(:sequences)
"""

_VISIT_NOTES_SQL = """
SELECT obslog_visit_note.pfs_visit_id, NULL AS camera, NULL AS data_flag,
    obslog_visit_note.body AS note, 'obslog' AS source
FROM obslog_visit_note
WHERE obslog_visit_note.pfs_visit_id = ANY(:visits)
UNION ALL
SELECT sps_annotation.pfs_visit_id, sps_camera.sps_camera_name, sps_annotation.data_flag,
    sps_annotation.notes, 'sps_annotation'
FROM sps_annotation
JOIN sps_camera ON sps_camera.sps_camera_id = sps_annotation.sps_camera_id
WHERE sps_annotation.pfs_visit_id = ANY(:visits)
"""

# tel_status has a row per status update, about one every 10 s during an exposure: summarized
# per visit. The range of the focus offset is kept, because the update that opens an exposure
# can predate the focus move made for it.
_TEL_STATUS_SQL = """
SELECT pfs_visit_id, count(*) AS n_status,
    min(m2_off3) AS focus_offset_min, max(m2_off3) AS focus_offset_max,
    max(abs(dither_ra)) AS dither_ra_max, max(abs(dither_dec)) AS dither_dec_max
FROM tel_status
WHERE pfs_visit_id = ANY(:visits)
GROUP BY pfs_visit_id
"""

# Science fibers only (targetType SCIENCE = 1): their proposals say whose design it is.
_DESIGN_PROPOSALS_SQL = """
SELECT pfs_design_id, proposal_id, count(*) AS n_fibers
FROM pfs_design_fiber
WHERE pfs_design_id = ANY(:designs) AND target_type = 1
GROUP BY pfs_design_id, proposal_id
"""

_PROPOSAL_RE = re.compile(r"^S\d\d[AB]-([A-Z]*)\d*([A-Z]*)")


def readVisitListing(opdb, start, end) -> pd.DataFrame:
    """Read every SpS visit with an exposure starting in ``[start, end)``.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The opdb, read-only.
    start, end : `datetime.datetime`
        The interval (HST), e.g. a `pfs.drp.qa.comparison.runs.Period`'s
        ``start`` and ``end``.

    Returns
    -------
    `pandas.DataFrame`
        One row per visit, with the `LISTING_COLUMNS`, sorted by visit.
    """
    listing = opdb.query_dataframe(_LISTING_SQL, params={"start": start, "end": end})
    listing = listing.reindex(columns=list(LISTING_COLUMNS))
    listing["iic_sequence_id"] = listing["iic_sequence_id"].astype("Int64")
    listing["group_id"] = listing["group_id"].astype("Int64")
    listing["pfs_design_id"] = listing["pfs_design_id"].astype("Int64")
    listing["time_exp_start"] = pd.to_datetime(listing["time_exp_start"])
    return listing.sort_values("pfs_visit_id").reset_index(drop=True)


def readNotes(opdb, visits: Iterable[int], sequences: Iterable[int]) -> pd.DataFrame:
    """Read the notes on some visits and sequences.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The opdb, read-only.
    visits : iterable of `int`
        PFS visits.
    sequences : iterable of `int`
        IIC sequences.

    Returns
    -------
    `pandas.DataFrame`
        One row per note: ``pfs_visit_id`` and ``iic_sequence_id`` (one of them
        NA), ``camera`` and ``data_flag`` (for an ``sps_annotation``), ``note``
        and ``source``: ``obslog`` (on a visit), ``obslog_sequence`` or
        ``sps_annotation`` (on one camera of a visit).
    """
    visits = sorted({int(visit) for visit in visits})
    sequences = sorted({int(sequence) for sequence in sequences})
    frames = []
    if visits:
        frames.append(opdb.query_dataframe(_VISIT_NOTES_SQL, params={"visits": visits}))
    if sequences:
        sequenceNotes = opdb.query_dataframe(_SEQUENCE_NOTES_SQL, params={"sequences": sequences})
        frames.append(sequenceNotes.assign(source="obslog_sequence"))
    columns = ["pfs_visit_id", "iic_sequence_id", "camera", "data_flag", "note", "source"]
    notes = pd.concat([frame.reindex(columns=columns) for frame in frames] or [pd.DataFrame(columns=columns)])
    for column in ("pfs_visit_id", "iic_sequence_id", "data_flag"):
        notes[column] = pd.to_numeric(notes[column]).astype("Int64")
    return notes.reset_index(drop=True)


def readTelStatus(opdb, visits: Iterable[int]) -> pd.DataFrame:
    """Read the telescope focus offset and dithers of some visits.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The opdb, read-only.
    visits : iterable of `int`
        PFS visits, normally those on sky.

    Returns
    -------
    `pandas.DataFrame`
        One row per visit with any ``tel_status``: ``pfs_visit_id``,
        ``n_status``, ``focus_offset_min`` and ``focus_offset_max`` (hexapod
        focus offset, mm), ``dither_ra_max`` and ``dither_dec_max`` (largest
        absolute dither, arcsec).
    """
    visits = sorted({int(visit) for visit in visits})
    columns = [
        "pfs_visit_id",
        "n_status",
        "focus_offset_min",
        "focus_offset_max",
        "dither_ra_max",
        "dither_dec_max",
    ]
    if not visits:
        return pd.DataFrame(columns=columns)
    return opdb.query_dataframe(_TEL_STATUS_SQL, params={"visits": visits}).reindex(columns=columns)


def readDesignCategories(opdb, designs: Iterable[int]) -> pd.DataFrame:
    """Read the proposal categories of the science fibers of some designs.

    Only the category is kept (``EN`` for engineering, ``QF``, ``OT``, ``UH``,
    ...), not the proposal: proposal IDs identify programs and stay out of
    anything this package writes.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The opdb, read-only.
    designs : iterable of `int`
        ``pfs_design_id`` values.

    Returns
    -------
    `pandas.DataFrame`
        ``pfs_design_id``, ``category`` (see `proposalCategory`) and
        ``n_fibers``, one row per design and category. A design with no
        science fibers has no rows.
    """
    designs = sorted({int(design) for design in designs})
    columns = ["pfs_design_id", "category", "n_fibers"]
    if not designs:
        return pd.DataFrame(columns=columns)
    proposals = opdb.query_dataframe(_DESIGN_PROPOSALS_SQL, params={"designs": designs})
    proposals["category"] = proposals["proposal_id"].map(proposalCategory)
    categories = proposals.groupby(["pfs_design_id", "category"], as_index=False)["n_fibers"].sum()
    return categories.reindex(columns=columns)


def proposalCategory(proposalId: str | None) -> str:
    """Return the category of a Subaru proposal ID.

    Parameters
    ----------
    proposalId : `str` or `None`
        E.g. ``S25B-EN16``, ``S25A-123QF``.

    Returns
    -------
    `str`
        The letters of the ID without semester and number (``EN``, ``QF``),
        ``none`` for a fiber with no proposal (``N/A``, empty or NULL), or
        ``other`` for an ID that doesn't follow the pattern.
    """
    if proposalId is None or pd.isna(proposalId) or str(proposalId).strip() in ("", "N/A"):
        return "none"
    match = _PROPOSAL_RE.match(str(proposalId).strip())
    if not match or not "".join(match.groups()):
        return "other"
    return "".join(match.groups())
