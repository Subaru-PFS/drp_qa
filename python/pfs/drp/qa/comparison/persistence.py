"""The last lit exposure before each dark, per camera: where to look for persistence.

The n-arm detectors keep an image of a bright exposure for a while, so a dark taken soon after
an arc or a quartz can still show its traces. Until darks are measured (PIPE2D-1925), this lists
for each dark the last exposure that lit the same camera and how long before it ended, from the
opdb listing alone: the darks most at risk are those with the shortest gaps after the brightest
exposures.
"""

import pandas as pd

from pfs.drp.qa.comparison.findings import lampsOf
from pfs.drp.qa.comparison.plan import expectedDetectors

__all__ = ["DARK_TYPES", "UNLIT_EXP_TYPES", "gapSummary", "lastLitBefore"]

#: Sequence types of darks.
DARK_TYPES = ("darks", "masterDarks")

#: Exposure types that light nothing.
UNLIT_EXP_TYPES = ("dark", "bias", "test")

#: Gap bounds (minutes) for `gapSummary`.
GAP_BINS = (0, 5, 30, 120, float("inf"))


def lastLitBefore(visits: pd.DataFrame, arm: str = "n") -> pd.DataFrame:
    """Return, for each dark on an arm, the last exposure that lit the same camera.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`; the whole period,
        so that the lit exposures are there too.
    arm : `str`, optional
        The arm. Default ``n``.

    Returns
    -------
    `pandas.DataFrame`
        One row per dark and set of cameras sharing its last lit exposure:
        ``visit``, ``night``, ``sequence_type``, ``exptime``, ``cameras``
        (``n1,n2``), ``litVisit``, ``litType`` (sequence type, and its name
        for a calibration), ``litExptime`` (s), ``lamps`` and
        ``minutesSince`` (from the end of the lit exposure to the start of the
        dark). The lit columns are empty when nothing lit the camera earlier
        in the listing. Sorted by ``minutesSince``, shortest first.
    """
    columns = ["visit", "night", "sequence_type", "exptime", "cameras", "litVisit", "litType"]
    columns += ["litExptime", "lamps", "minutesSince"]
    detectors = expectedDetectors(visits)
    detectors = detectors[detectors["arm"] == arm]
    info = visits.set_index("pfs_visit_id")
    detectors = detectors.assign(
        start=detectors["visit"].map(info["time_exp_start"]),
        isDark=detectors["visit"].map(info["sequence_type"]).isin(DARK_TYPES),
        isLit=~detectors["visit"].map(info["exp_type"]).isin(UNLIT_EXP_TYPES),
    )
    darks = detectors[detectors["isDark"]].sort_values("start")
    lit = detectors[detectors["isLit"] & ~detectors["isDark"]].sort_values("start")
    if darks.empty:
        return pd.DataFrame(columns=columns)

    matched = pd.merge_asof(
        darks[["visit", "spectrograph", "start"]],
        lit[["visit", "spectrograph", "start"]].rename(columns={"visit": "litVisit", "start": "litStart"}),
        left_on="start",
        right_on="litStart",
        by="spectrograph",
        direction="backward",
        allow_exact_matches=False,
    )
    litInfo = info.reindex(matched["litVisit"])
    litEnd = matched["litStart"] + pd.to_timedelta(litInfo["exptime"].to_numpy(), unit="s")
    matched["minutesSince"] = ((matched["start"] - litEnd).dt.total_seconds() / 60).round(1)
    matched["camera"] = arm + matched["spectrograph"].astype(str)

    rows = []
    for (visit, litVisit), group in matched.groupby(["visit", "litVisit"], dropna=False, sort=False):
        dark = info.loc[visit]
        row = {
            "visit": visit,
            "night": dark["night"],
            "sequence_type": dark["sequence_type"],
            "exptime": dark["exptime"],
            "cameras": ",".join(sorted(group["camera"])),
            "litVisit": litVisit,
            "litType": "",
            "litExptime": float("nan"),
            "lamps": "",
            "minutesSince": group["minutesSince"].min(),
        }
        if not pd.isna(litVisit):
            source = info.loc[int(litVisit)]
            name = source.get("sequence_name")
            row["litType"] = str(source["sequence_type"]) + (
                f" {name.strip()!r}" if isinstance(name, str) and source["category"] != "science" else ""
            )
            row["litExptime"] = source["exptime"]
            row["lamps"] = ", ".join(lampsOf(source.get("cmd_str")))
        rows.append(row)
    result = pd.DataFrame(rows, columns=columns)
    result["litVisit"] = result["litVisit"].astype("Int64")
    return result.sort_values(["minutesSince", "visit"], na_position="last", ignore_index=True)


def gapSummary(lastLit: pd.DataFrame) -> pd.DataFrame:
    """Count darks by how soon after a lit exposure they were taken.

    Parameters
    ----------
    lastLit : `pandas.DataFrame`
        From `lastLitBefore`.

    Returns
    -------
    `pandas.DataFrame`
        ``after the last lit exposure`` (``< 5 min``, ``5-30 min``, ...,
        ``none earlier``) and ``darks``.
    """
    labels = ["< 5 min", "5-30 min", "30-120 min", "> 120 min"]
    perDark = lastLit.groupby("visit")["minutesSince"].min()
    binned = pd.cut(perDark, list(GAP_BINS), labels=labels, right=False)
    counts = binned.value_counts().reindex(labels, fill_value=0)
    counts["none earlier"] = int(perDark.isna().sum())
    return counts.rename("darks").rename_axis("after the last lit exposure").reset_index()
