"""Judge an image's numbers: the one judging path.

The tasks measure and store numbers; nothing but `gate` gives a verdict. It
takes the numbers, the rules (`pfs.drp.qa.metrics.registry.CHECKS`) and the
table of each camera's normal values (`pfs.drp.qa.metrics.normalValues`), and
returns ``PASS``, ``WARN``, ``FAIL`` or ``UNKNOWN`` per image. An image takes
its worst check; one with no check judged is ``UNKNOWN``, never ``PASS``.

The numbers come as up to three tables, as the Butler or qadb holds them:

``iq``
    ``iqQaMetrics``: one row per image, with ``obsType``, ``illumination``
    and ``seqName``.
``lines``
    ``dmLineResidualMetrics``: one row per image, reference, species,
    ``statusType`` and metric.
``twilight``
    The twilight metrics (`pfs.drp.qa.metrics.twilight.sequenceConsistency`):
    one row per image of a twilight sequence.

`measurements` says which numbers each check reads, as one row per image,
check and number. `judge` gives each a verdict, and `gate` the image's.

What each check reads:

- ``skyOffset`` and ``calibOffset``: against the calibration, over every good
  line (``BOTH``, or ``UNFITTED`` when the visit's fit marked none). In x the
  median offset of the trace centroids; in y the species furthest off, sky
  species on a science or all-sky frame and lamp species on an arc.
- ``skyLineResidual`` and ``skyLineResidualBlue``: against the visit's
  detectorMap, the median of the held-back (``RESERVED``) lines in km/s: OH on
  r, n and m, [O I] on b.
- ``scatterX`` and ``scatterY``: against the visit's detectorMap, the robust
  RMS of the held-back lines: each lamp species of an arc but mercury and
  cadmium, and in x the trace centroids of a quartz.
- ``width``, ``linesPerFiberArc``, ``linesPerFiberQuartz`` and
  ``traceCentreOffset``: ``iqQaMetrics`` columns.
- the three twilight checks: the twilight metrics' columns.

On a visit read with the m arm, the b arm's standing x offset and the n arm's
y offset (`pfs.drp.qa.metrics.registry.M_ARM_OFFSETS` rows of the normal
values) are subtracted before ``skyOffset`` is judged, and the reason says so.

A frame's type needs its illumination, which only ``iqQaMetrics`` records. An
arc or quartz with line residuals and no ``iqQaMetrics`` row is not judged on
them: lit through the engineering fibers, it would be judged as a full slit.

Nothing here imports the LSST stack.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.registry import (
    ALLSKY,
    ARC,
    CHECKS,
    FIXED,
    IIS_ARC,
    M_ARM_OFFSETS,
    MIN_LINES_PER_FIBER,
    MIN_NORMAL_FRAMES,
    QUARTZ,
    SCIENCE,
    TWILIGHT,
    UNJUDGED_SPECIES,
    Check,
    frameType,
    lampOf,
    specFor,
)

__all__ = [
    "IMAGE_KEYS",
    "MEASUREMENT_COLUMNS",
    "NORMAL_KEYS",
    "SKY_LINE",
    "STATUS_ORDER",
    "VERDICT_COLUMNS",
    "gate",
    "judge",
    "longLines",
    "measurements",
    "normalKey",
    "worst",
]

#: Verdicts from best to worst. ``UNKNOWN`` is no verdict: nothing was judged.
STATUS_ORDER = ("UNKNOWN", "PASS", "WARN", "FAIL")

#: Columns identifying an image.
IMAGE_KEYS = ("visit", "arm", "spectrograph")

#: Columns of `measurements`.
MEASUREMENT_COLUMNS = (
    *IMAGE_KEYS,
    "frameType",
    "check",
    "metric",
    "part",
    "camera",
    "selector",
    "value",
    "withM",
)

#: Columns of `gate`: the verdict, the check that decided it, and why.
VERDICT_COLUMNS = (*IMAGE_KEYS, "frameType", "qaStatus", "qaDecidedBy", "qaReason")

#: What identifies a normal value, with the visit it is valid from.
NORMAL_KEYS = ("metric", "camera", "frameType", "selector")

#: The sky species whose held-back lines give an arm's residual in km/s.
SKY_LINE = {"b": "OI", "r": "OH", "n": "OH", "m": "OH"}

#: Line species of the night sky, and the description of trace centroids, in ``dmLineResidualMetrics``.
_SKY_SPECIES = ("OH", "O2", "OI", "NaI")
_TRACE = "Trace"

#: The residual statistics of a wide line-residual table (`longLines`).
_LINE_METRICS = ("medianDeltaX", "robustRmsDeltaX", "medianDeltaY", "robustRmsDeltaY", "medianDeltaYKms")


def worst(statuses: Sequence[str]) -> str:
    """Return the worst of ``statuses``; ``UNKNOWN`` when there is none."""
    known = [status for status in statuses if status in STATUS_ORDER]
    return max(known, key=STATUS_ORDER.index) if known else "UNKNOWN"


def normalKey(check: Check, arm: pd.Series, spectrograph: pd.Series) -> pd.Series:
    """Return what a check's normal value is kept per: the camera (``b1``) or the arm (``b``)."""
    if "camera" in check.by:
        return arm.astype(str) + spectrograph.astype(int).astype(str)
    return arm.astype(str)


def longLines(wide: pd.DataFrame) -> pd.DataFrame:
    """Return a wide line-residual table (a column per statistic) as ``dmLineResidualMetrics`` rows."""
    metrics = [name for name in _LINE_METRICS if name in wide.columns]
    keys = [column for column in wide.columns if column not in _LINE_METRICS]
    return wide.melt(id_vars=keys, value_vars=metrics, var_name="metric", value_name="value")


def _empty() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in MEASUREMENT_COLUMNS}).astype(
        {"visit": "int64", "spectrograph": "int64", "value": "float64", "withM": bool}
    )


def _rows(frame: pd.DataFrame, check: str, metric: str, part, selector, value, withM=False) -> pd.DataFrame:
    """Return `measurements` rows for the images of ``frame``."""
    spec = CHECKS[check]
    keep = frame["frameType"].isin(spec.frames)
    if spec.arms:
        keep &= frame["arm"].isin(spec.arms)
    frame = frame[keep]
    return pd.DataFrame(
        {
            "visit": frame["visit"].astype("int64").to_numpy(),
            "arm": frame["arm"].astype(str).to_numpy(),
            "spectrograph": frame["spectrograph"].astype("int64").to_numpy(),
            "frameType": frame["frameType"].to_numpy(),
            "check": check,
            "metric": metric,
            "part": pd.Series(part, index=keep.index)[keep].to_numpy()
            if isinstance(part, pd.Series)
            else part,
            "camera": normalKey(spec, frame["arm"], frame["spectrograph"]).to_numpy(),
            "selector": pd.Series(selector, index=keep.index)[keep].to_numpy()
            if isinstance(selector, pd.Series)
            else selector,
            "value": pd.Series(value, index=keep.index)[keep].astype(float).to_numpy(),
            "withM": pd.Series(withM, index=keep.index)[keep].astype(bool).to_numpy()
            if isinstance(withM, pd.Series)
            else withM,
        },
        columns=list(MEASUREMENT_COLUMNS),
    )


def _iqRows(iq: pd.DataFrame) -> list[pd.DataFrame]:
    """Return the measurements ``iqQaMetrics`` gives: width, lines per lit fiber, trace centre offset."""
    iq = iq.reset_index(drop=True)
    illumination = iq["illumination"] if "illumination" in iq else pd.Series(None, index=iq.index)
    iq = iq.assign(
        frameType=[frameType(o, i) for o, i in zip(iq["obsType"], illumination, strict=True)],
        lamp=iq["seqName"].map(lampOf) if "seqName" in iq else "",
    )
    frames = []

    def column(name: str) -> pd.Series:
        values = iq[name].astype(float) if name in iq else pd.Series(np.nan, index=iq.index)
        return values.mask(specFor(name).notMeasured(iq))

    perFiber = column("linesPerFiber")
    # A width resting on a handful of lines is not judged: an arc under the floor of lines per lit fiber.
    sparse = iq["frameType"].isin([ARC, IIS_ARC]) & (perFiber < MIN_LINES_PER_FIBER)
    frames.append(_rows(iq, "width", "medFwhm", "", iq["lamp"], column("medFwhm").mask(sparse)))
    for check in ("linesPerFiberArc", "linesPerFiberQuartz"):
        frames.append(_rows(iq, check, "linesPerFiber", "", iq["lamp"], perFiber))
    frames.append(_rows(iq, "traceCentreOffset", "medDxCenter", "", "", column("medDxCenter")))
    return frames


def _lineRows(lines: pd.DataFrame, iq: pd.DataFrame | None) -> list[pd.DataFrame]:
    """Return the measurements ``dmLineResidualMetrics`` gives: offsets, scatter, sky-line residuals."""
    lines = lines.reset_index(drop=True)
    keys = list(IMAGE_KEYS)
    kind = lines["obsType"].map(lambda obsType: frameType(obsType))
    # Lamp frames need their illumination, which only iqQaMetrics records.
    lamp = kind.isin([ARC, QUARTZ])
    if iq is not None and len(iq):
        illumination = iq["illumination"] if "illumination" in iq else pd.Series(None, index=iq.index)
        known = pd.DataFrame(
            {
                "visit": iq["visit"].to_numpy(),
                "arm": iq["arm"].to_numpy(),
                "spectrograph": iq["spectrograph"].to_numpy(),
                "_iqType": [frameType(o, i) for o, i in zip(iq["obsType"], illumination, strict=True)],
                "_inIq": True,
            }
        ).drop_duplicates(keys)
        merged = lines[keys].merge(known, on=keys, how="left")
        inIq = merged["_inIq"].fillna(False).astype(bool).to_numpy()
        kind = kind.where(~(lamp & inIq), merged["_iqType"].where(pd.Series(inIq), None))
        kind = kind.where(~lamp | inIq, None)
    else:
        kind = kind.where(~lamp, None)
    cameraSet = lines["cameraSet"] if "cameraSet" in lines else pd.Series("", index=lines.index)
    lines = lines.assign(frameType=kind, withM=cameraSet.fillna("").astype(str).str.contains("m"))
    lines = lines[lines["frameType"].notna() & np.isfinite(lines["value"].astype(float))]

    description = lines["description"].astype(str)
    isTrace = description == _TRACE
    isSky = description.isin(_SKY_SPECIES)
    calib = (lines["reference"] == "calib") & lines["statusType"].isin(["BOTH", "UNFITTED"])
    heldBack = (lines["reference"] == "visit") & (lines["statusType"] == "RESERVED")
    onSky = lines["frameType"].isin([SCIENCE, ALLSKY])
    metric = lines["metric"]

    frames = []
    for check, species in (("skyOffset", isSky), ("calibOffset", ~isSky)):
        x = lines[calib & isTrace & (metric == "medianDeltaX")]
        frames.append(_rows(x, check, "medianDeltaX", "x", "", x["value"], x["withM"]))
        y = lines[calib & ~isTrace & species & (metric == "medianDeltaY")]
        frames.append(
            _rows(y, check, "medianDeltaY", "y " + y["description"].astype(str), "", y["value"], y["withM"])
        )
    for check in ("skyLineResidual", "skyLineResidualBlue"):
        best = description == lines["arm"].map(SKY_LINE)
        kms = lines[heldBack & onSky & best & (metric == "medianDeltaYKms")]
        frames.append(_rows(kms, check, "medianDeltaYKms", kms["description"].astype(str), "", kms["value"]))
    judgedSpecies = np.where(
        lines["frameType"] == QUARTZ, isTrace, ~isTrace & ~isSky & ~description.isin(UNJUDGED_SPECIES)
    )
    for check, name in (("scatterX", "robustRmsDeltaX"), ("scatterY", "robustRmsDeltaY")):
        scatter = lines[heldBack & judgedSpecies & (metric == name)]
        species = scatter["description"].astype(str)
        frames.append(_rows(scatter, check, name, species, species, scatter["value"]))
    return frames


def _twilightRows(twilight: pd.DataFrame) -> list[pd.DataFrame]:
    """Return the measurements the twilight metrics give."""
    twilight = twilight.reset_index(drop=True).assign(frameType=TWILIGHT)
    frames = []
    for check in ("twilightLevel", "twilightScatter", "twilightPatch"):
        (name,) = CHECKS[check].metrics
        if name in twilight:
            value = twilight[name].astype(float).mask(specFor(name).notMeasured(twilight))
            frames.append(_rows(twilight, check, name, "", "", value))
    return frames


def measurements(
    iq: pd.DataFrame | None = None, lines: pd.DataFrame | None = None, twilight: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the numbers each check reads: one row per image, check and number.

    Parameters
    ----------
    iq : `pandas.DataFrame`, optional
        ``iqQaMetrics`` rows.
    lines : `pandas.DataFrame`, optional
        ``dmLineResidualMetrics`` rows (`longLines` of a wide table).
    twilight : `pandas.DataFrame`, optional
        Twilight metrics rows.

    Returns
    -------
    `pandas.DataFrame`
        `MEASUREMENT_COLUMNS`: the image and its ``frameType``, the ``check``
        and the ``metric`` read, ``part`` (which of a check's numbers: ``x``,
        ``y OH``, a species), ``camera`` and ``selector`` (what a normal value
        is kept per: the camera or arm, and the lamp or species), ``value`` as
        measured, and ``withM`` (the visit was read with the m arm). Numbers
        not measured are left out.
    """
    frames = [_empty()]
    if iq is not None and len(iq):
        frames += _iqRows(iq)
    if lines is not None and len(lines):
        frames += _lineRows(lines, iq)
    if twilight is not None and len(twilight):
        frames += _twilightRows(twilight)
    result = pd.concat([frame for frame in frames if len(frame)] or [_empty()], ignore_index=True)
    return result[~np.isnan(result["value"])].reset_index(drop=True)


def _inForce(rows: pd.DataFrame, normalValues: pd.DataFrame | None, value: str = "normal") -> pd.DataFrame:
    """Return, per row, the normal value in force: the latest valid at or before the row's visit.

    Parameters
    ----------
    rows : `pandas.DataFrame`
        With `NORMAL_KEYS` and ``visit``.
    normalValues : `pandas.DataFrame`
        With `NORMAL_KEYS`, ``value``, ``count`` and ``validFrom``.
    value : `str`
        Name of the column returned.

    Returns
    -------
    `pandas.DataFrame`
        Indexed like ``rows``: ``value`` (NaN when there is none) and
        ``validFrom``.
    """
    result = pd.DataFrame({value: np.nan, "validFrom": np.nan}, index=rows.index)
    if normalValues is None or not len(normalValues) or not len(rows):
        return result
    keys = list(NORMAL_KEYS)
    table = normalValues[normalValues["count"] >= MIN_NORMAL_FRAMES]
    table = table.assign(
        validFrom=table["validFrom"].astype("int64"),
        **{key: table[key].fillna("").astype(str) for key in keys},
    ).sort_values("validFrom", kind="stable")
    left = rows.assign(
        _row=np.arange(len(rows)),
        visit=rows["visit"].astype("int64"),
        **{key: rows[key].fillna("").astype(str) for key in keys},
    ).sort_values("visit", kind="stable")
    merged = pd.merge_asof(
        left[["_row", "visit", *keys]],
        table[[*keys, "validFrom", "value"]],
        left_on="visit",
        right_on="validFrom",
        by=keys,
        direction="backward",
    ).sort_values("_row")
    result[value] = merged["value"].to_numpy()
    result["validFrom"] = merged["validFrom"].to_numpy()
    return result


def judge(measured: pd.DataFrame, normalValues: pd.DataFrame | None = None) -> pd.DataFrame:
    """Give each measurement a verdict.

    Parameters
    ----------
    measured : `pandas.DataFrame`
        `measurements`.
    normalValues : `pandas.DataFrame`, optional
        The normal values that may judge these images
        (`pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`): those of the
        images' collection. Without them a relative check gives no verdict.

    Returns
    -------
    `pandas.DataFrame`
        ``measured`` with:

        ``value``
            As judged: less the m-arm offset where one applies.
        ``mArmOffset``
            The offset subtracted, NaN where none was.
        ``normal``
            The normal value in force, for a relative check.
        ``excess``
            What is compared with the limits: the absolute value for an
            offset, the per cent from ``normal`` in the bad direction for a
            relative check, else the value.
        ``warn``, ``fail``
            The check's limits.
        ``status``
            ``PASS``, ``WARN`` or ``FAIL``; empty when not judged.
        ``reason``
            Why, for ``WARN`` and ``FAIL``; why not, when not judged.
    """
    judged = measured.reset_index(drop=True).copy()
    numRows = len(judged)
    judged["mArmOffset"] = np.nan
    for name, arm in M_ARM_OFFSETS.items():
        metric = "medianDeltaX" if name.endswith("X") else "medianDeltaY"
        use = (
            (judged["check"] == "skyOffset")
            & (judged["metric"] == metric)
            & (judged["arm"] == arm)
            & judged["withM"]
        )
        if not use.any():
            continue
        keyed = judged.loc[use].assign(
            metric=name,
            camera=judged.loc[use, "arm"] + judged.loc[use, "spectrograph"].astype(str),
            frameType="",
            selector="",
        )
        offset = _inForce(keyed, normalValues)["normal"]
        judged.loc[use, "mArmOffset"] = offset.to_numpy()
        judged.loc[use, "value"] = judged.loc[use, "value"] - offset.fillna(0.0).to_numpy()

    # In y an offset check judges one species per image: the one furthest off.
    isY = judged["check"].isin(["skyOffset", "calibOffset"]) & (judged["metric"] == "medianDeltaY")
    if isY.any():
        furthest = judged.loc[isY, "value"].abs().groupby([judged[key] for key in IMAGE_KEYS]).idxmax()
        judged = judged[~isY | judged.index.isin(furthest)].reset_index(drop=True)
        numRows = len(judged)

    checks = judged["check"].map(CHECKS)
    relative = np.array([check.kind != FIXED for check in checks], dtype=bool)
    higherIsWorse = np.array([specFor(name).higherIsWorse for name in judged["metric"]], dtype=bool)
    absolute = np.array([specFor(name).absolute for name in judged["metric"]], dtype=bool)
    judged["normal"] = _inForce(judged, normalValues)["normal"].where(relative).to_numpy() if numRows else []
    value = judged["value"].to_numpy(dtype=float)
    normal = judged["normal"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        percent = np.where(higherIsWorse, 1.0, -1.0) * (value / normal - 1.0) * 100.0
    excess = np.where(relative, percent, np.where(absolute, np.abs(value), value))
    judged["excess"] = excess
    judged["warn"] = [check.warn for check in checks]
    judged["fail"] = [check.fail for check in checks]

    # Not judged: no normal value, or a line count whose normal value is itself too small.
    noNormal = relative & ~np.isfinite(normal)
    sparse = relative & (judged["metric"] == "linesPerFiber").to_numpy() & (normal < MIN_LINES_PER_FIBER)
    unjudged = noNormal | sparse | np.isnan(excess)
    status = np.full(numRows, "PASS", dtype=object)
    for level in ("warn", "fail"):
        limit = judged[level].to_numpy(dtype=float)
        # A value at a limit has crossed it, whatever the last bit of a percentage says.
        with np.errstate(invalid="ignore"):
            status[(excess >= limit) | np.isclose(excess, limit, rtol=1e-9, atol=0.0)] = level.upper()
    status[unjudged] = ""
    judged["status"] = status

    reason = np.full(numRows, "", dtype=object)
    reason[noNormal] = "no normal value"
    reason[sparse] = f"normal value under {MIN_LINES_PER_FIBER:g} lines per lit fiber"
    for row in np.flatnonzero(np.isin(status, ["WARN", "FAIL"])):
        reason[row] = _reason(judged.iloc[row])
    judged["reason"] = reason
    return judged


def _reason(row: pd.Series) -> str:
    """Return why a measurement warns or fails."""
    check, spec = CHECKS[row["check"]], specFor(row["metric"])
    limit = row["fail"] if row["status"] == "FAIL" else row["warn"]
    part = str(row["part"])
    # "x offset", "y offset (OH)", "OH residual", "NeI x scatter", "width".
    if part.startswith("y "):
        name = f"{spec.label} ({part[2:]})"
    elif part in ("", "x"):
        name = spec.label or spec.name
    elif row["metric"] == "medianDeltaYKms":
        name = f"{'[O I]' if part == 'OI' else part} {spec.label}"
    else:
        name = f"{part} {spec.label}"
    if check.kind == FIXED:
        text = f"{name} {spec.show(row['value'])}, limit {limit:g}{spec.unitSuffix}"
        if np.isfinite(row["mArmOffset"]):
            text += f" (less the m-arm offset {row['mArmOffset']:+.2f}{spec.unitSuffix})"
        return text
    sign = "+" if spec.higherIsWorse else "-"
    change = row["excess"] if spec.higherIsWorse else -row["excess"]
    return (
        f"{name} {spec.show(row['value'])} is {change:+.1f} % from normal {spec.show(row['normal'])}, "
        f"limit {sign}{limit:g} %"
    )


def gate(
    iq: pd.DataFrame | None = None,
    lines: pd.DataFrame | None = None,
    twilight: pd.DataFrame | None = None,
    normalValues: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Give each image a verdict, the check that decided it, and why.

    Parameters
    ----------
    iq, lines, twilight : `pandas.DataFrame`, optional
        The numbers; see `measurements`.
    normalValues : `pandas.DataFrame`, optional
        The normal values of the images' collection; see `judge`.

    Returns
    -------
    `pandas.DataFrame`
        One row per image of any table, sorted, with `VERDICT_COLUMNS`:

        ``frameType``
            The image's frame type, `None` when it has none.
        ``qaStatus``
            The worst verdict of the image's checks; ``UNKNOWN`` when none
            was judged.
        ``qaDecidedBy``
            The check with that verdict when it is ``WARN`` or ``FAIL``.
        ``qaReason``
            The reasons of every ``WARN`` and ``FAIL``, joined by ``"; "``;
            for ``UNKNOWN``, why nothing was judged.
    """
    keys = list(IMAGE_KEYS)
    images = []
    for table, kind in ((iq, None), (twilight, TWILIGHT), (lines, None)):
        if table is None or not len(table):
            continue
        image = table[keys].copy()
        if kind is not None:
            image["frameType"] = kind
        else:
            illumination = (
                table["illumination"] if "illumination" in table else pd.Series(None, index=table.index)
            )
            image["frameType"] = [
                frameType(o, i) for o, i in zip(table["obsType"], illumination, strict=True)
            ]
        images.append(image)
    if not images:
        return pd.DataFrame(columns=list(VERDICT_COLUMNS))
    # The first table naming an image gives its type: iqQaMetrics knows the illumination, the line residuals do not.
    images = (
        pd.concat(images, ignore_index=True)
        .drop_duplicates(keys)
        .astype({"visit": "int64", "spectrograph": "int64"})
    )

    judged = judge(measurements(iq, lines, twilight), normalValues)
    verdicts = judged[judged["status"] != ""]
    rank = verdicts["status"].map(STATUS_ORDER.index)
    verdicts = verdicts.assign(_rank=rank)
    status = verdicts.groupby(keys, sort=False)["_rank"].max().map(lambda index: STATUS_ORDER[int(index)])
    crossed = verdicts[verdicts["_rank"] >= STATUS_ORDER.index("WARN")]
    decider = (
        crossed.sort_values("_rank", ascending=False, kind="stable")
        .groupby(keys, sort=False)["check"]
        .first()
    )
    reasons = crossed.groupby(keys, sort=False)["reason"].agg("; ".join)
    whyNot = (
        judged[judged["status"] == ""]
        .groupby(keys, sort=False)["reason"]
        .agg(lambda values: "; ".join(dict.fromkeys(value for value in values if value)))
    )

    result = images.join(status.rename("qaStatus"), on=keys).join(decider.rename("qaDecidedBy"), on=keys)
    result = result.join(reasons.rename("qaReason"), on=keys).join(whyNot.rename("_whyNot"), on=keys)
    unknown = result["qaStatus"].isna()
    result["qaStatus"] = result["qaStatus"].fillna("UNKNOWN")
    result["qaDecidedBy"] = result["qaDecidedBy"].fillna("")
    noType = result["frameType"].isna()
    why = (
        result["_whyNot"]
        .fillna("")
        .where(result["_whyNot"].fillna("") != "", "nothing to judge was measured")
    )
    why = why.where(~noType, "no frame type")
    result["qaReason"] = result["qaReason"].fillna("").where(~unknown, why)
    return result[list(VERDICT_COLUMNS)].sort_values(keys, kind="stable", ignore_index=True)
