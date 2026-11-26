"""End-of-run analysis of the AG cameras and the guide loop.

Takes the tables of `pfs.drp.qa.guiders.queries.readAgRun`, never a
database. `prepareRun` turns them into one row per AG exposure (`AgRun`);
the other functions take that and return DataFrames or `Flag` lists, which
`pfs.drp.qa.guiders.runPlotting` draws and `pfs.drp.qa.guiders.runReport`
writes up.

Cameras are ``agc_camera_id``, 0-5 (AG1 is 0). They come in two groups,
AG1/3/5 and AG2/4/6, wired and placed separately, which can fail together
(Run 30 lost AG1/3/5), so the analysis reports them by group.

Times are HST. A night runs from 14:00 to 14:00 and is named by its evening's
date, as in `pfs.drp.qa.comparison.runs` (`NIGHT_START`).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pfs.drp.qa.comparison.runs import NIGHT_START
from pfs.utils import agMetrics
from pfs.utils.coordinates.coordinates import det2dp
from pfs.utils.datamodel.ag import GuideOffsetFlag

__all__ = [
    "CAMERA_GROUPS",
    "DETECTION_FLAG_COLUMNS",
    "JUDGED_STATES",
    "SCIENCE_TYPES",
    "AgRun",
    "CameraState",
    "Flag",
    "RunConfig",
    "cameraFocus",
    "cameraSummary",
    "describePeriod",
    "detectionFlagShares",
    "detectorCells",
    "findDropouts",
    "findFlags",
    "findRejectionClusters",
    "fitModelErrors",
    "groupSummary",
    "isRejectedOffset",
    "magnitudeSummary",
    "nightOf",
    "nightPosition",
    "onSkyRa",
    "prepareRun",
    "runSummary",
    "stateShares",
    "summariseModelErrors",
    "visitGuideErrors",
    "visitMatchFraction",
]

#: The two groups of AG cameras, by ``agc_camera_id``.
CAMERA_GROUPS = {"AG1/3/5": (0, 2, 4), "AG2/4/6": (1, 3, 5)}

#: Sequence types whose guiding is judged.
SCIENCE_TYPES = ("scienceObject", "scienceObject_windowed")


class CameraState:
    """What a camera gave in an AG exposure in which some camera matched a star."""

    #: It matched at least one guide star.
    OK = 0
    #: It matched none, but its design had too few bright guide stars to expect one.
    FEW_STARS = 1
    #: It detected something but matched none of the design's guide stars.
    UNMATCHED = 2
    #: It detected nothing at all. The opdb holds a camera's detections, not whether it returned an image, so
    #: a camera that is dead and one whose image held nothing look the same here.
    DARK = 3
    #: The guider left it out of its fit (``agc_match.camera_enabled`` false).
    DISABLED = 4
    #: No camera matched anything (the sky wasn't usable); the camera isn't judged.
    UNUSABLE = -1


#: The states in which a camera is judged, in the order drawn: what it gave when it was expected to match.
JUDGED_STATES = {
    CameraState.OK: "matched",
    CameraState.UNMATCHED: "detections, no match",
    CameraState.DARK: "no detections",
    CameraState.DISABLED: "disabled",
}


@dataclass(frozen=True)
class RunConfig:
    """Parameters of the end-of-run analysis.

    Parameters
    ----------
    minDesignStars : `int`
        A camera is expected to match a star only when its design put at
        least this many guide stars brighter than the reader's ``magLimit``
        on it.
    minDropout : `int`
        Consecutive judged AG exposures in which a camera matched nothing,
        for a dropout.
    longDropout : `int`
        A dropout at least this long (or one with no detections at all) is
        flagged on its own; shorter ones are summarized.
    minOpenFrames : `int`
        Accepted guide offsets, from AG exposures taken while the
        spectrograph shutters were open, a visit needs for its guide error.
    minFitStars : `int`
        A guide offset from fewer good matches than this is flagged when the
        guider accepted it.
    jitterFactor : `float`
        A visit's guide jitter is flagged above this many times the
        run's median.
    fwhmTolerance : `float`
        A camera's median FWHM is flagged this fraction from the cameras'
        median.
    minFramesNight : `int`
        Nights with fewer AG exposures are left out of the plots.
    maxCorrection : `float`
        The guider's ``max_correction`` (arcsec), drawn on the acquisition plot.
    minClusterPaired, minClusterRejected : `int`
        A camera's AG exposure is judged for a cluster of rejected matches
        when it has at least ``minClusterPaired`` detections paired with a
        guide star and ``minClusterRejected`` of them rejected.
    clusterFraction : `float`
        A cluster needs at least this fraction of the pairs rejected...
    clusterOffsetPix : `float`
        ...and their mean position this far (pixels) from the good ones'.
    minClusters, clusterFactor : `float`
        A camera is flagged for clusters on it alone (no other camera had
        one in that AG exposure: clusters on several cameras at once are the
        whole solution stressed) in at least ``minClusters`` AG exposures,
        and in over ``clusterFactor`` times the other cameras' median (at
        least 1).
    modelErrorRadiusPix : `float`
        A camera's position-model error is quoted at this distance (pixels)
        from its detector's center.
    minModelMatches, minModelSpreadPix : `float`
        A night's position-model fit for a camera needs this many good
        matches, spread at least this far (pixels, the smaller axis of their
        spread about each exposure's mean) across the detector.
    isotropicRatio : `float`
        A model error whose two singular values are closer than this ratio
        (a rotation or uniform scale) has no direction, and none is quoted.
    maxModelError, modelErrorFactor : `float`
        A camera is flagged when its position-model error (µm, at
        ``modelErrorRadiusPix``) is over ``maxModelError`` and over
        ``modelErrorFactor`` times the cameras' median, in the run's nightly
        median. The error measured is never zero: the matches are noisy.
    """

    minDesignStars: int = 3
    minDropout: int = 30
    longDropout: int = 300
    minOpenFrames: int = 5
    minFitStars: int = 3
    jitterFactor: float = 3.0
    fwhmTolerance: float = 0.15
    minFramesNight: int = 20
    maxCorrection: float = 10.0
    minClusterPaired: int = 16
    minClusterRejected: int = 5
    clusterFraction: float = 0.25
    clusterOffsetPix: float = 250.0
    minClusters: int = 10
    clusterFactor: float = 2.5
    modelErrorRadiusPix: float = 500.0
    minModelMatches: int = 100
    minModelSpreadPix: float = 50.0
    isotropicRatio: float = 0.7
    maxModelError: float = 4.0
    modelErrorFactor: float = 1.5


@dataclass
class AgRun:
    """The AG exposures of a run, one row each, ready to analyse and plot.

    Attributes
    ----------
    frames : `pandas.DataFrame`
        One row per AG exposure, time-ordered: the ``exposures`` table plus
        ``night``, ``population`` (science, convergence, focus sweep or
        other), ``pfs_design_id``, ``dec_center_designed``, ``shutter``
        (the whole AG exposure was taken while the visit's spectrograph
        shutters were open; False where the read has no shutter times),
        ``n_good`` (good
        matches over the cameras), ``n_cams`` (cameras with one),
        ``residual_um`` (the good matches' distance from where the guider
        expected them: the cameras' medians, weighted by their matches)
        and, per camera ``c``, ``n_spots{c}``, ``n_good{c}``,
        ``n_disabled{c}``, ``n_design{c}`` (its design's bright guide
        stars) and ``state{c}`` (a `CameraState`).
    cameras : `pandas.DataFrame`
        The ``cameras`` table with ``fwhm`` (arcsec), ``night`` and
        ``population``; a column an older read lacks is NaN (0 for
        ``n_disabled``).
    focus : `pandas.DataFrame`
        The ``focus`` table with ``night``.
    nights : `pandas.Series`
        AG exposures per night.
    grid : `pandas.DataFrame`
        The ``grid`` table with ``night`` (empty if it wasn't read).
    magnitudes : `pandas.DataFrame`
        The ``magnitudes`` table with ``night`` and ``population`` (empty
        if it wasn't read).
    inUse : `tuple` [`int`, ...]
        The cameras (``agc_camera_id``) that detected anything during the
        run. One that never did was not in use: it is named once and left
        out of every comparison between cameras.
    config : `RunConfig`
        The parameters it was prepared with.
    """

    frames: pd.DataFrame
    cameras: pd.DataFrame
    focus: pd.DataFrame
    nights: pd.Series
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)
    magnitudes: pd.DataFrame = field(default_factory=pd.DataFrame)
    inUse: tuple[int, ...] = tuple(range(6))
    config: RunConfig = field(default_factory=RunConfig)

    @property
    def plottedNights(self) -> pd.DatetimeIndex:
        """The nights with at least ``config.minFramesNight`` AG exposures."""
        return self.nights[self.nights >= self.config.minFramesNight].index


@dataclass(frozen=True)
class Flag:
    """A finding worth a reader's attention.

    Attributes
    ----------
    kind : `str`
        What it is about: ``camera``, ``guiding`` or ``image``.
    severe : `bool`
        Whether it is a fault rather than something to look at.
    text : `str`
        One sentence, in Markdown.
    agc_camera_id : `int` or `None`
        The camera it is about, if one.
    """

    kind: str
    severe: bool
    text: str
    agc_camera_id: int | None = None


GRID_COLUMNS = [
    "agc_exposure_id",
    "agc_camera_id",
    "row_cell",
    "column_cell",
    "n_paired",
    "n_good",
    "x_good_pix",
    "y_good_pix",
    "x_rejected_pix",
    "y_rejected_pix",
    "dx_um",
    "dy_um",
]

MAGNITUDE_COLUMNS = ["pfs_visit_id", "agc_camera_id", "magnitude", "n_expected", "n_good"]

#: The columns of ``visits`` giving when a visit's spectrograph shutters opened and closed.
SHUTTER_COLUMNS = ("shutter_open", "shutter_close")

#: The columns of ``cameras`` counting detections with one of ics_agccActor's bad-detection flags, with the
#: flag's name (`pfs.drp.qa.guiders.queries.AG_DETECTION_FLAGS`).
DETECTION_FLAG_COLUMNS = {
    "n_edge": "EDGE",
    "n_saturated": "SATURATED",
    "n_bad_shape": "BAD_SHAPE",
    "n_bad_ellip": "BAD_ELLIP",
    "n_bad_size": "BAD_SIZE",
}


def nightOf(times: pd.Series) -> pd.Series:
    """Return the night (midnight of its evening's date) of each time."""
    return (pd.to_datetime(times) - pd.Timedelta(hours=NIGHT_START)).dt.normalize()


def _population(sequenceType: pd.Series) -> np.ndarray:
    sequenceType = sequenceType.fillna("")
    return np.select(
        [
            sequenceType.isin(SCIENCE_TYPES),
            sequenceType.eq("moveToPfsDesign"),
            sequenceType.eq("agFocusSweep"),
        ],
        ["science", "convergence", "focus sweep"],
        "other",
    )


def prepareRun(tables: dict[str, pd.DataFrame], config: RunConfig | None = None) -> AgRun:
    """Make one row per AG exposure from the tables of `readAgRun`.

    Parameters
    ----------
    tables : `dict` [`str`, `pandas.DataFrame`]
        From `pfs.drp.qa.guiders.queries.readAgRun`.
    config : `RunConfig`, optional
        The parameters.

    Returns
    -------
    run : `AgRun`
        The exposures, cameras and focus offsets of the run.
    """
    config = config or RunConfig()
    frames = tables["exposures"].sort_values("taken_at", ignore_index=True)
    visits = tables["visits"].drop_duplicates("pfs_visit_id")
    # A read older than the shutter times has none: no AG exposure is then known to be with the shutters open.
    visits = visits.assign(**{name: pd.NaT for name in SHUTTER_COLUMNS if name not in visits})
    frames = frames.merge(
        visits[["pfs_visit_id", "pfs_design_id", "dec_center_designed", "sequence_type", *SHUTTER_COLUMNS]],
        how="left",
        on="pfs_visit_id",
    )
    frames["night"] = nightOf(frames.taken_at)
    frames["population"] = _population(frames.sequence_type)
    taken = pd.to_datetime(frames.taken_at)
    frames["shutter"] = (taken >= pd.to_datetime(frames.shutter_open)) & (
        taken + pd.to_timedelta(frames.agc_exptime, unit="s") <= pd.to_datetime(frames.shutter_close)
    )

    cameras = tables["cameras"]
    # A read older than these columns lacks them: no camera was recorded as disabled, and the rest are unknown.
    if "n_disabled" not in cameras:
        cameras = cameras.assign(n_disabled=0)
    cameras = cameras.assign(
        **{name: np.nan for name in ("residual_um", *DETECTION_FLAG_COLUMNS) if name not in cameras}
    )
    perCamera = cameras.groupby(["agc_exposure_id", "agc_camera_id"])[
        ["n_spots", "n_good", "n_disabled"]
    ].sum()
    designStars = (
        tables["designStars"]
        .pivot_table(index="pfs_design_id", columns="agc_camera_id", values="n_bright", aggfunc="sum")
        .reindex(columns=range(6))
        .fillna(0)
    )
    columns = {}
    for name in ("n_spots", "n_good", "n_disabled"):
        wide = perCamera[name].unstack().reindex(index=frames.agc_exposure_id, columns=range(6)).fillna(0)
        for cid in range(6):
            columns[f"{name}{cid}"] = wide[cid].to_numpy().astype(int)
    for cid in range(6):
        columns[f"n_design{cid}"] = (
            frames.pfs_design_id.map(designStars[cid]).fillna(0).to_numpy().astype(int)
        )
    frames = pd.concat([frames, pd.DataFrame(columns, index=frames.index)], axis=1)
    good = frames[[f"n_good{cid}" for cid in range(6)]].to_numpy()
    frames["n_good"] = good.sum(axis=1)
    frames["n_cams"] = (good > 0).sum(axis=1)
    usable = frames.n_cams > 0
    for cid in range(6):
        state = np.select(
            [
                frames[f"n_disabled{cid}"] > 0,
                frames[f"n_good{cid}"] > 0,
                frames[f"n_design{cid}"] < config.minDesignStars,
                frames[f"n_spots{cid}"] > 0,
            ],
            [CameraState.DISABLED, CameraState.OK, CameraState.FEW_STARS, CameraState.UNMATCHED],
            CameraState.DARK,
        )
        frames[f"state{cid}"] = np.where(usable, state, CameraState.UNUSABLE)
    inUse = tuple(cid for cid in range(6) if frames[f"n_spots{cid}"].any()) or tuple(range(6))

    matched = cameras[cameras.residual_um.notna() & (cameras.n_good > 0)]
    residual = (matched.residual_um * matched.n_good).groupby(
        matched.agc_exposure_id
    ).sum() / matched.groupby("agc_exposure_id").n_good.sum()
    frames["residual_um"] = frames.agc_exposure_id.map(residual)

    cameras = cameras.assign(fwhm=agMetrics.storedRadiusToFwhm(cameras.sigma_pix, variancePerMoment=1)).merge(
        frames[["agc_exposure_id", "night", "population"]], on="agc_exposure_id"
    )
    focus = tables["focus"].assign(night=lambda df: nightOf(df.created_at))
    nights = frames.groupby("night").size()
    grid = tables.get("grid")
    if grid is None:
        grid = pd.DataFrame(columns=GRID_COLUMNS)
    grid = grid.merge(frames[["agc_exposure_id", "night"]], on="agc_exposure_id")
    magnitudes = tables.get("magnitudes")
    if magnitudes is None:
        magnitudes = pd.DataFrame(columns=MAGNITUDE_COLUMNS)
    magnitudes = magnitudes.merge(
        frames.drop_duplicates("pfs_visit_id")[["pfs_visit_id", "night", "population"]], on="pfs_visit_id"
    )
    return AgRun(
        frames=frames,
        cameras=cameras,
        focus=focus,
        nights=nights,
        grid=grid,
        magnitudes=magnitudes,
        inUse=inUse,
        config=config,
    )


def onSkyRa(frames: pd.DataFrame) -> pd.Series:
    """Return the guide offsets' ΔRA on the sky, ΔRA cos(Dec) (arcsec).

    ``guide_delta_ra`` is a coordinate offset. The guide Dec is used where
    there is one, else the design's.
    """
    dec = frames.guide_dec.fillna(frames.dec_center_designed)
    return frames.guide_delta_ra * np.cos(np.radians(dec))


def isRejectedOffset(mask: pd.Series, flag: GuideOffsetFlag = GuideOffsetFlag.INVALID_OFFSET) -> pd.Series:
    """Which guide offsets ics_agActor did not use, from ``agc_guide_offset.mask``.

    Parameters
    ----------
    mask : `pandas.Series`
        ``agc_guide_offset.mask`` (`GuideOffsetFlag`); NaN where a frame has no offset.
    flag : `GuideOffsetFlag`, optional
        The bit to test: ``INVALID_OFFSET`` (the default) for every offset not used,
        ``TOO_FEW_STARS`` for those rejected because too few guide stars matched.

    Returns
    -------
    rejected : `pandas.Series` [`bool`]
        True where ``flag`` is set; False where it isn't or there is no offset.
    """
    return (mask.fillna(int(GuideOffsetFlag.OK)).astype(int) & int(flag)) != 0


def visitGuideErrors(run: AgRun) -> pd.DataFrame:
    """Return the size and the frame-to-frame jitter of each science visit's guide offsets.

    A visit is measured from the guide offsets the guider accepted in AG
    exposures taken while the spectrograph shutters were open: what the
    spectrograph saw. AG exposures go on after the shutters close, and the
    pointing may then be moved (in Run 31's windowed visits the offsets
    jump to about 1″ in the first AG exposure after the close), so those
    are no part of the visit's guiding. Visits with fewer than
    ``minOpenFrames`` such offsets are left out, and so is every visit of a
    read with no shutter times. A guide offset is the pointing error the
    guider measured in that AG exposure, so their RMS about zero is how
    large the error was, and the jitter, the standard deviation of
    consecutive differences over √2, which a slow drift barely changes, is
    how steady the measurement was.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    errors : `pandas.DataFrame`
        One row per visit: ``pfs_visit_id``, ``night``, ``taken_at`` (its
        first such frame), ``n_frames``, ``n_good`` and ``n_cams`` (median
        over its frames), ``full`` (stars were matched on every camera in
        use), the jitter ``ra`` (on the sky), ``dec`` and ``inr`` (arcsec),
        the size ``ra_size``, ``dec_size`` and ``inr_size`` (arcsec), and
        ``residual_um`` (the median distance of its good matches from where
        the guider expected them).
    """
    config = run.config
    frames = run.frames[
        run.frames.population.eq("science")
        & run.frames.shutter
        & run.frames["mask"].eq(int(GuideOffsetFlag.OK))
    ].copy()
    frames["ra"] = onSkyRa(frames)

    def jitter(values: pd.Series) -> float:
        differences = np.diff(values.to_numpy(float))
        differences = differences[np.isfinite(differences)]
        # The sample standard deviation (n - 1): a visit may have six frames, and with n the median such visit
        # reads 0.88 of what a long one does (0.87 in 6-frame pieces of Run 31's long visits; 0.97 with n - 1).
        return float(np.std(differences, ddof=1) / np.sqrt(2)) if len(differences) > 1 else np.nan

    def size(values: pd.Series) -> float:
        values = values.to_numpy(float)
        values = values[np.isfinite(values)]
        return float(np.sqrt(np.mean(values**2))) if len(values) else np.nan

    columns = [
        "pfs_visit_id",
        "night",
        "taken_at",
        "n_frames",
        "n_good",
        "n_cams",
        "full",
        "ra",
        "dec",
        "inr",
        "ra_size",
        "dec_size",
        "inr_size",
        "residual_um",
    ]
    rows = []
    for visit, group in frames.groupby("pfs_visit_id"):
        if len(group) < config.minOpenFrames:
            continue
        nCams = int(group.n_cams.median())
        rows.append(
            [
                visit,
                group.night.iloc[0],
                group.taken_at.iloc[0],
                len(group),
                group.n_good.median(),
                nCams,
                nCams >= len(run.inUse),
                jitter(group.ra),
                jitter(group.guide_delta_dec),
                jitter(group.guide_delta_insrot),
                size(group.ra),
                size(group.guide_delta_dec),
                size(group.guide_delta_insrot),
                group.residual_um.median(),
            ]
        )
    return pd.DataFrame(rows, columns=columns).astype({"full": bool})


def findDropouts(run: AgRun) -> pd.DataFrame:
    """Find the stretches in which a camera matched no star although it should have.

    Only the AG exposures in which some camera matched a star, and in which
    the camera's design had at least ``minDesignStars`` bright guide stars,
    are judged (`JUDGED_STATES`). A dropout is at least ``minDropout``
    consecutive judged exposures that aren't OK. A camera not in use
    (`AgRun.inUse`) has none: it is named once, by `findFlags`.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    dropouts : `pandas.DataFrame`
        One row per dropout: ``agc_camera_id``, ``n_frames``, ``start`` and
        ``end`` (times of its first and last exposure), ``start_visit``,
        ``end_visit``, ``to_end`` (it lasts to the end of the run), ``dark``
        (over 90% of its exposures detected nothing), ``disabled`` (over
        90% had the camera left out of the fit) and ``last_good`` (the last
        match before it, or NaT).
    """
    config = run.config
    columns = [
        "agc_camera_id",
        "n_frames",
        "start",
        "end",
        "start_visit",
        "end_visit",
        "to_end",
        "dark",
        "disabled",
        "last_good",
    ]
    rows = []
    for cid in run.inUse:
        frames = run.frames[run.frames[f"state{cid}"].isin(list(JUDGED_STATES))].reset_index(drop=True)
        missing = (frames[f"state{cid}"] != CameraState.OK).to_numpy()
        if not missing.any():
            continue
        stretch = np.cumsum(np.r_[True, missing[1:] != missing[:-1]])
        for label in np.unique(stretch[missing]):
            index = np.flatnonzero(stretch == label)
            if len(index) < config.minDropout:
                continue
            rows.append(
                [
                    cid,
                    len(index),
                    frames.taken_at.iloc[index[0]],
                    frames.taken_at.iloc[index[-1]],
                    int(frames.pfs_visit_id.iloc[index[0]]),
                    int(frames.pfs_visit_id.iloc[index[-1]]),
                    bool(index[-1] == len(frames) - 1),
                    bool((frames[f"state{cid}"].iloc[index] == CameraState.DARK).mean() > 0.9),
                    bool((frames[f"state{cid}"].iloc[index] == CameraState.DISABLED).mean() > 0.9),
                    frames.taken_at.iloc[index[0] - 1] if index[0] > 0 else pd.NaT,
                ]
            )
    return pd.DataFrame(rows, columns=columns)


def groupSummary(run: AgRun) -> pd.DataFrame:
    """Compare the two camera groups.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    summary : `pandas.DataFrame`
        One column per group of `CAMERA_GROUPS`, one row per quantity,
        over its cameras in use: the percentage of judged exposures in
        which they matched a star; the median good matches per camera and
        exposure; the median FWHM (arcsec) on each detector half; the
        median of its cameras' focus minus the median of the cameras in
        use, and of its own median minus theirs (µm), over exposures with
        a focus value from each: what the guider's focus would read with
        only this group. A group with no camera in use is NaN.
    """
    frames = run.frames[run.frames.night.isin(run.plottedNights)]
    cameras = run.cameras[run.cameras.night.isin(run.plottedNights)]
    z = cameraFocus(run, frames)
    zMedian = z.median(axis=1)
    summary = {}
    for name, group in CAMERA_GROUPS.items():
        cids = [cid for cid in group if cid in run.inUse]
        states = frames[[f"state{cid}" for cid in cids]].to_numpy()
        judged = states[np.isin(states, list(JUDGED_STATES))]
        inGroup = cameras.agc_camera_id.isin(cids)
        own = z[cids]
        summary[name] = {
            "matched (% of judged exposures)": 100 * np.mean(judged == CameraState.OK)
            if len(judged)
            else np.nan,
            "good matches per camera": float(np.median(frames[[f"n_good{cid}" for cid in cids]].to_numpy()))
            if cids and len(frames)
            else np.nan,
            "FWHM, without RIGHT (″)": cameras[inGroup & cameras.right_half.eq(0)].fwhm.median(),
            "FWHM, RIGHT half (″)": cameras[inGroup & cameras.right_half.eq(1)].fwhm.median(),
            "focus minus the cameras' median (µm)": (
                float(np.median(own.to_numpy() - zMedian.to_numpy()[:, None])) if cids and len(z) else np.nan
            ),
            "focus from this group alone minus dZ (µm)": (
                float((own.median(axis=1) - zMedian).median()) if cids and len(z) else np.nan
            ),
        }
    return pd.DataFrame(summary)


def cameraFocus(run: AgRun, frames: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return each camera's focus error in the AG exposures that have one from every camera in use.

    Parameters
    ----------
    run : `AgRun`
        The run.
    frames : `pandas.DataFrame`, optional
        The AG exposures (rows of ``run.frames``; default, all).

    Returns
    -------
    focus : `pandas.DataFrame`
        ``guide_delta_z1``-``z6`` of the cameras in use (µm), one column
        per ``agc_camera_id``, indexed as ``frames``.
    """
    frames = run.frames if frames is None else frames
    z = frames[[f"guide_delta_z{cid + 1}" for cid in run.inUse]].astype(float) * 1e3
    z.columns = list(run.inUse)
    return z[np.isfinite(z).all(axis=1)]


def _hst(when) -> str:
    return pd.Timestamp(when).strftime("%Y-%m-%d %H:%M")


def _visitList(visits, limit: int = 12) -> str:
    visits = [str(int(v)) for v in visits]
    return ", ".join(visits[:limit]) + (" …" if len(visits) > limit else "")


def findFlags(
    run: AgRun,
    jitter: pd.DataFrame,
    dropouts: pd.DataFrame,
    clusters: pd.DataFrame | None = None,
    modelErrors: pd.DataFrame | None = None,
) -> list[Flag]:
    """List what the run's AG data show is wrong or worth a look.

    Parameters
    ----------
    run : `AgRun`
        The run.
    jitter : `pandas.DataFrame`
        From `visitGuideErrors`.
    dropouts : `pandas.DataFrame`
        From `findDropouts`.
    clusters : `pandas.DataFrame`, optional
        From `findRejectionClusters`.
    modelErrors : `pandas.DataFrame`, optional
        From `fitModelErrors`.

    Returns
    -------
    flags : `list` [`Flag`]
        Cameras not in use; camera dropouts; cameras whose matches are often rejected in a
        cluster on one side of the detector, or whose position model is off; guide frames rejected,
        missing, or accepted from too few stars; visits with unusual guide
        jitter; cameras with unusual image sizes. The FWHM of a camera's two
        halves is not flagged: they focus differently by design.
    """
    config = run.config
    flags = []
    for cid in sorted(set(range(6)) - set(run.inUse)):
        flags.append(
            Flag(
                "camera",
                False,
                f"AG{cid + 1}: not in use (no detections in any of the run's {len(run.frames)} AG exposures).",
                cid,
            )
        )
    long = dropouts.dark | (dropouts.n_frames >= config.longDropout)
    for _, d in dropouts[long].iterrows():
        what = (
            "no detections at all"
            if d.dark
            else "left out of the guider's fit (disabled)"
            if d.get("disabled", False)
            else "detections but no matched star"
        )
        until = "to the end of the run" if d.to_end else f"to {_hst(d.end)} (visit {d.end_visit})"
        last = f"; last matched star {_hst(d.last_good)}" if pd.notna(d.last_good) else ""
        flags.append(
            Flag(
                "camera",
                True,
                f"**AG{d.agc_camera_id + 1}: {what} in {d.n_frames} consecutive AG exposures** from "
                f"{_hst(d.start)} (visit {d.start_visit}) {until}{last}.",
                int(d.agc_camera_id),
            )
        )
    disabled = (
        dropouts.disabled.astype(bool) if "disabled" in dropouts else pd.Series(False, index=dropouts.index)
    )
    for short, what in (
        (
            dropouts[~long & ~disabled],
            f"a camera with {config.minDesignStars}+ bright designed guide stars matched none",
        ),
        (dropouts[~long & disabled], "a camera was left out of the guider's fit (disabled)"),
    ):
        if not len(short):
            continue
        counts = short.groupby("agc_camera_id").n_frames.agg(["size", "sum"])
        flags.append(
            Flag(
                "camera",
                False,
                f"Shorter stretches ({config.minDropout}+ exposures) in which {what}: "
                + ", ".join(f"AG{cid + 1} {n}× ({total} exposures)" for cid, (n, total) in counts.iterrows())
                + ".",
            )
        )

    if clusters is not None and len(clusters):
        alone = clusters[clusters.alone.astype(bool)]
        counts = alone.groupby("agc_camera_id").size().reindex(list(run.inUse), fill_value=0)
        for cid, group in alone.groupby("agc_camera_id"):
            others = counts.drop(cid).median()
            if len(group) < config.minClusters or len(group) <= config.clusterFactor * max(others, 1):
                continue
            flags.append(
                Flag(
                    "camera",
                    True,
                    f"**AG{cid + 1}: matches rejected in a cluster on one side of its detector, on it alone, in "
                    f"{len(group)} AG exposures** (the other cameras' median: {others:g}), toward "
                    f"{_meanAngle(group.angle_deg):.0f}° (0° = +x, 90° = +y on the detector; direction consistency "
                    f"{_consistency(group.angle_deg):.2f}, 1 if always the same), in "
                    f"{group.pfs_visit_id.nunique()} visits: "
                    f"{_visitList(group.pfs_visit_id.value_counts().index)}.",
                    int(cid),
                )
            )
    if modelErrors is not None and len(modelErrors):
        summary = summariseModelErrors(run, modelErrors)
        typical = summary.error_um.median()
        for cid, row in summary.iterrows():
            if row.error_um > max(config.maxModelError, config.modelErrorFactor * typical):
                direction = (
                    f"growing toward {row.angle_deg:.0f}°"
                    if np.isfinite(row.angle_deg)
                    else "in no one direction (a rotation or uniform scale)"
                )
                flags.append(
                    Flag(
                        "camera",
                        False,
                        f"AG{cid + 1}: its position model is off by {row.error_um:.1f} µm at "
                        f"{config.modelErrorRadiusPix:g} px from its detector's center, {direction} "
                        f"(nightly median; the cameras' median {typical:.1f} µm).",
                        int(cid),
                    )
                )

    frames = run.frames
    mask = frames["mask"]
    rejected = frames[isRejectedOffset(mask)]
    if len(rejected):
        nTooFew = int(isRejectedOffset(rejected["mask"], GuideOffsetFlag.TOO_FEW_STARS).sum())
        tooFew = f", {nTooFew} of them TOO_FEW_STARS" if nTooFew else ""
        flags.append(
            Flag(
                "guiding",
                True,
                f"**{len(rejected)} guide frames rejected** (INVALID_OFFSET: not sent to the telescope{tooFew}), in "
                f"{rejected.pfs_visit_id.nunique()} visits: {_visitList(sorted(rejected.pfs_visit_id.unique()))}.",
            )
        )
    fewStars = frames[mask.eq(int(GuideOffsetFlag.OK)) & (frames.n_good < config.minFitStars)]
    # A focus sweep's offsets are not guiding: counted apart, so the guide loop's own are plain.
    sweep = fewStars.population.eq("focus sweep")
    for chosen, what, sent in (
        (fewStars[~sweep], "guide offsets", " (sent to the telescope when under max_correction)"),
        (fewStars[sweep], "offsets of AG focus sweeps", ""),
    ):
        if chosen.empty:
            continue
        counts = chosen.pfs_visit_id.value_counts()
        nights = chosen.night.nunique()
        last = chosen.sort_values("taken_at").iloc[-1]
        flags.append(
            Flag(
                "guiding",
                True,
                f"**{len(chosen)} {what} accepted from fewer than {config.minFitStars} matched stars**{sent}, "
                f"on {nights} night{'' if nights == 1 else 's'}, the last {_hst(last.taken_at)} "
                f"(visit {int(last.pfs_visit_id)}): "
                + ", ".join(f"visit {v} ({n})" for v, n in counts.head(8).items())
                + (" …" if len(counts) > 8 else "")
                + ".",
            )
        )
    noOffset = int(mask.isna().sum())
    if noOffset:
        flags.append(
            Flag("guiding", False, f"{noOffset} of {len(frames)} AG exposures have no guide offset.")
        )

    if len(jitter):
        full = jitter[jitter.n_cams >= len(run.inUse)]
        reference = full if len(full) else jitter
        ratio = pd.concat([jitter[axis] / reference[axis].median() for axis in ("ra", "dec", "inr")], axis=1)
        jitter = jitter.assign(ratio=ratio.max(axis=1))
        high = jitter[jitter.ratio > config.jitterFactor].sort_values("ratio", ascending=False)
        if len(high):
            flags.append(
                Flag(
                    "guiding",
                    False,
                    f"{len(high)} of {len(jitter)} science visits have guide jitter over "
                    f"{config.jitterFactor:g}× the median on some axis; worst: "
                    + ", ".join(f"{int(r.pfs_visit_id)} ({r.ratio:.0f}×)" for _, r in high.head(8).iterrows())
                    + ".",
                )
            )

    fwhm = run.cameras[run.cameras.night.isin(run.plottedNights)].groupby("agc_camera_id").fwhm.median()
    for cid, value in fwhm.items():
        if abs(value / fwhm.median() - 1) > config.fwhmTolerance:
            flags.append(
                Flag(
                    "image",
                    False,
                    f"AG{cid + 1}: FWHM {100 * (value / fwhm.median() - 1):+.0f}% from the cameras' median "
                    f"({value:.2f}″ against {fwhm.median():.2f}″).",
                    int(cid),
                )
            )
    return flags


def describePeriod(run: AgRun) -> str:
    """Return one line naming the run's nights (by their evening's date), visits and AG exposures."""
    frames = run.frames
    first, last = run.nights.index.min(), run.nights.index.max()
    return (
        f"nights {first:%Y-%m-%d} to {last:%Y-%m-%d}, "
        f"visits {frames.pfs_visit_id.min()}–{frames.pfs_visit_id.max()}, "
        f"{len(frames)} AG exposures on {len(run.plottedNights)} nights"
    )


def nightPosition(times: pd.Series, nights: pd.DatetimeIndex) -> pd.Series:
    """Place times on a strip of nights: the night's index plus the fraction of 18:00-06:00 elapsed.

    Each night spans [i + 0.05, i + 0.95]; times before 18:00 or after 06:00
    are put at its ends, and times on other nights are NaN.
    """
    night = nightOf(times)
    index = pd.Series(np.arange(len(nights)), index=nights)
    hours = (pd.to_datetime(times) - night - pd.Timedelta(hours=18)).dt.total_seconds() / 3600
    return night.map(index) + 0.05 + 0.9 * hours.clip(0, 12) / 12


def _meanAngle(degrees: pd.Series) -> float:
    """Return the circular mean of some angles (degrees, in [0, 360))."""
    radians = np.radians(np.asarray(degrees, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360)


def _consistency(degrees: pd.Series) -> float:
    """Return how consistent some directions are: 1 if all the same, ~0 if random (mean resultant length)."""
    radians = np.radians(np.asarray(degrees, dtype=float))
    return float(np.hypot(np.sin(radians).mean(), np.cos(radians).mean())) if len(radians) else np.nan


def axialMean(degrees: pd.Series) -> float:
    """Return the mean of some axes (directions modulo 180°), in [0, 180).

    An axis at 1° and one at 179° average to 0°, not 90°: the angles are
    doubled, averaged on the circle, and halved.
    """
    return _meanAngle(2 * np.asarray(degrees, dtype=float)) / 2 % 180 if len(degrees) else np.nan


def summariseModelErrors(run: AgRun, modelErrors: pd.DataFrame) -> pd.DataFrame:
    """Summarize each camera's nightly position-model errors over the run.

    Parameters
    ----------
    run : `AgRun`
        The run.
    modelErrors : `pandas.DataFrame`
        From `fitModelErrors`.

    Returns
    -------
    summary : `pandas.DataFrame`
        Indexed by ``agc_camera_id``: the median ``error_um`` and
        ``ratio``, and ``angle_deg``, the axial mean of the nights'
        directions, NaN when the median ratio is over
        ``RunConfig.isotropicRatio`` (no direction).
    """
    rows = {}
    for cid, errors in modelErrors.groupby("agc_camera_id"):
        ratio = errors.ratio.median()
        rows[cid] = {
            "error_um": errors.error_um.median(),
            "angle_deg": axialMean(errors.angle_deg) if ratio < run.config.isotropicRatio else np.nan,
            "ratio": ratio,
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["error_um", "angle_deg", "ratio"])


def findRejectionClusters(run: AgRun) -> pd.DataFrame:
    """Find the AG exposures in which a camera's rejected matches cluster on one side of its detector.

    A rejected match is a detection paired with a guide star but not
    counted as a good match. A few are rejected anywhere on a detector;
    when a camera's position model is off, those far from where it fits best
    are rejected first, wherever that is: AG3's top half in Run 29, a corner
    of AG4. So the measure is direction-free: the distance and direction from
    the good matches' mean position to the rejected ones'.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    clusters : `pandas.DataFrame`
        One row per (AG exposure, camera) with a cluster (see
        `RunConfig`): ``agc_exposure_id``, ``pfs_visit_id``, ``taken_at``,
        ``night``, ``agc_camera_id``, ``n_paired``, ``n_rejected``,
        ``offset_pix``, ``angle_deg`` (from the good matches to the
        rejected ones: 0° = +x, 90° = +y on the detector) and ``alone`` (no
        other camera had a cluster in that AG exposure); on the plotted
        nights only.
    """
    config = run.config
    columns = [
        "agc_exposure_id",
        "pfs_visit_id",
        "taken_at",
        "night",
        "agc_camera_id",
        "n_paired",
        "n_rejected",
        "offset_pix",
        "angle_deg",
        "alone",
    ]
    grid = run.grid[run.grid.night.isin(run.plottedNights)]
    if grid.empty:
        return pd.DataFrame(columns=columns)
    rejected = grid.n_paired - grid.n_good
    weighted = pd.DataFrame(
        {
            "agc_exposure_id": grid.agc_exposure_id,
            "agc_camera_id": grid.agc_camera_id,
            "n_paired": grid.n_paired,
            "n_good": grid.n_good,
            "n_rejected": rejected,
            "xg": grid.x_good_pix.fillna(0) * grid.n_good,
            "yg": grid.y_good_pix.fillna(0) * grid.n_good,
            "xr": grid.x_rejected_pix.fillna(0) * rejected,
            "yr": grid.y_rejected_pix.fillna(0) * rejected,
        }
    )
    sums = weighted.groupby(["agc_exposure_id", "agc_camera_id"]).sum()
    judged = (sums.n_paired >= config.minClusterPaired) & (sums.n_rejected >= config.minClusterRejected)
    sums = sums[judged & (sums.n_good > 0)]
    dx = sums.xr / sums.n_rejected - sums.xg / sums.n_good
    dy = sums.yr / sums.n_rejected - sums.yg / sums.n_good
    sums = sums.assign(offset_pix=np.hypot(dx, dy), angle_deg=np.degrees(np.arctan2(dy, dx)) % 360)
    found = sums[
        (sums.n_rejected / sums.n_paired >= config.clusterFraction)
        & (sums.offset_pix >= config.clusterOffsetPix)
    ].reset_index()
    found = found.merge(
        run.frames[["agc_exposure_id", "pfs_visit_id", "taken_at", "night"]], on="agc_exposure_id"
    )
    found["alone"] = found.groupby("agc_exposure_id").agc_camera_id.transform("size") == 1
    return found[columns].sort_values(["taken_at", "agc_camera_id"], ignore_index=True)


def _withoutPointing(grid: pd.DataFrame) -> pd.DataFrame:
    """Return the ``grid`` cells with good matches, less each camera's mean in each AG exposure.

    The pointing moves a camera's stars together, so the mean residual of a
    camera's good matches in an AG exposure is the pointing, not the camera:
    it is subtracted from ``dx_um`` and ``dy_um``, and the matches' mean
    position from ``x_good_pix`` and ``y_good_pix``.
    """
    grid = grid[grid.n_good > 0].dropna(subset=["x_good_pix", "y_good_pix", "dx_um", "dy_um"]).copy()
    if grid.empty:
        return grid
    keys = [grid.agc_exposure_id, grid.agc_camera_id]
    total = grid.n_good.groupby(keys).transform("sum")
    for column in ("x_good_pix", "y_good_pix", "dx_um", "dy_um"):
        grid[column] = grid[column] - (grid[column] * grid.n_good).groupby(keys).transform("sum") / total
    return grid


def _toDetectorAxes(cid: int) -> np.ndarray:
    """Return the 2x2 matrix taking a focal-plane vector (the opdb's frame) to camera ``cid``'s detector axes.

    The cameras sit around the field at 60° steps and their detectors are
    mirrored, so a residual in focal-plane x and y points somewhere else on
    each detector. The matrix is the inverse of how a detection reaches the
    opdb's focal plane: pfs_utils' ``det2dp``, a rotation and a reflection,
    with y then negated (the opdb stores ``agc_center_y_mm`` the other way
    up from ``det2dp``, as `coordinates.opdbToHardware` undoes).
    """
    origin = np.array(det2dp(cid, 0.0, 0.0), dtype=float)
    perPixel = np.column_stack(
        [np.array(det2dp(cid, *step), dtype=float) - origin for step in ((1.0, 0.0), (0.0, 1.0))]
    )
    perPixel[1] *= -1
    return np.linalg.inv(perPixel) * agMetrics.AG_PIXEL_SIZE_UM / 1000


def detectorCells(run: AgRun) -> pd.DataFrame:
    """Return where on each camera's detector matches were rejected, and how its residual varies across it.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    cells : `pandas.DataFrame`
        One row per (``agc_camera_id``, ``row_cell``, ``column_cell``) of
        the ``grid`` over the plotted nights: ``n_paired`` (detections
        paired with a guide star), ``n_good`` (good matches), ``rejected``
        (the percentage of the paired that were not good matches), and
        ``dx_um`` and ``dy_um``, the good matches' mean residual (measured
        minus expected, µm) along the detector's x and y, less each AG
        exposure's mean over the camera (the pointing), as `fitModelErrors`
        takes it. A camera whose position model is right leaves these at
        the noise everywhere; one that is off has residuals growing away
        from the detector's center. Empty if the ``grid`` wasn't read.
    """
    columns = ["agc_camera_id", "row_cell", "column_cell", "n_paired", "n_good", "rejected", "dx_um", "dy_um"]
    keys = ["agc_camera_id", "row_cell", "column_cell"]
    grid = run.grid[run.grid.night.isin(run.plottedNights)]
    if grid.empty:
        return pd.DataFrame(columns=columns)
    cells = grid.groupby(keys)[["n_paired", "n_good"]].sum()
    cells["rejected"] = 100 * (1 - cells.n_good / cells.n_paired)
    good = _withoutPointing(grid)
    sums = (
        good.assign(dx=good.dx_um * good.n_good, dy=good.dy_um * good.n_good)
        .groupby(keys)[["dx", "dy", "n_good"]]
        .sum()
    )
    residual = sums[["dx", "dy"]].div(sums.n_good, axis=0).reindex(cells.index)
    cells["dx_um"], cells["dy_um"] = np.nan, np.nan
    for cid in cells.index.get_level_values("agc_camera_id").unique():
        mine = cells.index.get_level_values("agc_camera_id") == cid
        onDetector = residual[mine].to_numpy(float) @ _toDetectorAxes(int(cid)).T
        cells.loc[mine, ["dx_um", "dy_um"]] = onDetector
    return cells.reset_index()[columns]


def fitModelErrors(run: AgRun) -> pd.DataFrame:
    """Fit how far off each camera's position model is, night by night.

    From the ``grid``, each cell's good matches give a mean position
    (pixels) and mean residual (measured minus expected, µm). In each AG
    exposure each camera's mean residual and position are subtracted,
    removing the pointing, which moves a camera's stars together; per night
    and camera a 2x2 matrix A is then fitted, residual = A (position - mean),
    weighted by the matches. A camera model whose position, rotation or scale
    is right leaves A at the noise; its largest singular value, times
    ``modelErrorRadiusPix``, is the error at that distance from the center,
    in the direction of its right singular vector. Field rotation adds the
    same small rotation to every camera.

    The residual is the one the guider's own fit sees (agc_match): the
    detection taken to the focal plane by pfs_utils' ``coordinates.det2dp``
    (the camera model ``_DXDP``, ``_DYDP``, ``_DTDP``), against the guide
    star placed by ``Subaru_POPT2_PFS`` (``makeBasis``). It doesn't test
    pfs_utils' other AG chain, ``CoordTransp``'s ``sky_pfi_ag`` and
    ``ag_pfimm_to_pixel``, which places guide stars in pfsDesign and
    pfsConfig; nor can it tell a camera-model error from a distortion-model
    error at that camera.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    errors : `pandas.DataFrame`
        One row per (``night``, ``agc_camera_id``): ``error_um``, the
        model error (µm) at ``modelErrorRadiusPix``; ``angle_deg``, the
        direction on the detector (0-180°; 0° = +x, 90° = +y) it grows
        fastest in; ``ratio``, its smaller singular value over its larger
        (near 1, a rotation or uniform scale, has no direction); ``n_good``;
        and ``A11``-``A22`` (µm per pixel). On the plotted nights, with
        enough matches spread across the detector (`RunConfig`).
    """
    config = run.config
    columns = [
        "night",
        "agc_camera_id",
        "error_um",
        "angle_deg",
        "ratio",
        "n_good",
        "A11",
        "A12",
        "A21",
        "A22",
    ]
    grid = _withoutPointing(run.grid[run.grid.night.isin(run.plottedNights)])
    if grid.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (night, cid), cells in grid.groupby(["night", "agc_camera_id"]):
        weight = cells.n_good.to_numpy(float)
        positions = cells[["x_good_pix", "y_good_pix"]].to_numpy()
        # The fit needs matches spread across the detector: with one cell per exposure, the de-meaned positions
        # are rounding noise, which a relative rank test would take for a spread.
        spread = np.sqrt(np.linalg.eigvalsh(positions.T @ (positions * weight[:, None]) / weight.sum()).min())
        if weight.sum() < config.minModelMatches or spread < config.minModelSpreadPix:
            continue
        X = positions * np.sqrt(weight)[:, None]
        A = np.vstack(
            [
                np.linalg.lstsq(X, cells[c].to_numpy() * np.sqrt(weight), rcond=None)[0]
                for c in ("dx_um", "dy_um")
            ]
        )
        _, singular, vt = np.linalg.svd(A)
        rows.append(
            [
                night,
                cid,
                singular[0] * config.modelErrorRadiusPix,
                np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180,
                singular[1] / singular[0] if singular[0] > 0 else np.nan,
                int(weight.sum()),
                *A.ravel(),
            ]
        )
    return pd.DataFrame(rows, columns=columns)


def cameraSummary(run: AgRun, clusters: pd.DataFrame, modelErrors: pd.DataFrame) -> pd.DataFrame:
    """Summarize each camera over the run.

    Parameters
    ----------
    run : `AgRun`
        The run.
    clusters : `pandas.DataFrame`
        From `findRejectionClusters`.
    modelErrors : `pandas.DataFrame`
        From `fitModelErrors`.

    Returns
    -------
    summary : `pandas.DataFrame`
        One column per camera (``AG1``-``AG6``), one row per quantity:
        the percentage of judged AG exposures in which it matched a star,
        its median good matches per exposure, its median FWHM (arcsec) on
        each half, the median distance (µm) of its good matches from where
        the guider expected them, the percentage of its detections carrying
        each of ics_agccActor's bad-detection flags, its focus minus the
        median of the cameras in use (µm), its AG exposures with a cluster
        of rejected matches on it alone, their mean direction and its
        consistency, and its position-model error (µm, nightly median) and
        the direction it grows in. A camera not in use is NaN throughout.
    """
    frames = run.frames[run.frames.night.isin(run.plottedNights)]
    cameras = run.cameras[run.cameras.night.isin(run.plottedNights)]
    z = cameraFocus(run, frames)
    zMedian = z.median(axis=1)
    radius = run.config.modelErrorRadiusPix
    models = summariseModelErrors(run, modelErrors)
    summary = {}
    for cid in range(6):
        used = cid in run.inUse
        states = frames[f"state{cid}"]
        states = states[states.isin(list(JUDGED_STATES))]
        own = clusters[(clusters.agc_camera_id == cid) & clusters.alone.astype(bool)]
        mine = cameras[cameras.agc_camera_id == cid]
        nSpots = mine.n_spots.sum()
        summary[f"AG{cid + 1}"] = {
            "matched (% of judged exposures)": 100 * float((states == CameraState.OK).mean())
            if used and len(states)
            else np.nan,
            "good matches per exposure": float(frames[f"n_good{cid}"].median())
            if used and len(frames)
            else np.nan,
            "FWHM, without RIGHT (″)": mine[mine.right_half.eq(0)].fwhm.median(),
            "FWHM, RIGHT half (″)": mine[mine.right_half.eq(1)].fwhm.median(),
            "good matches' distance from expected (µm)": mine.residual_um.median(),
            **{
                f"detections flagged {flag} (%)": 100 * mine[column].sum() / nSpots
                if nSpots and mine[column].notna().any()
                else np.nan
                for column, flag in DETECTION_FLAG_COLUMNS.items()
            },
            "focus minus the cameras' median (µm)": float((z[cid] - zMedian).median())
            if used and len(z)
            else np.nan,
            "exposures with rejected matches clustered on it alone": len(own) if used else np.nan,
            "their direction (°)": _meanAngle(own.angle_deg) if len(own) else np.nan,
            "their direction's consistency (0-1)": _consistency(own.angle_deg),
            f"position-model error at {radius:g} px (µm)": models.error_um.get(cid, np.nan),
            "its direction (°, 0-180)": models.angle_deg.get(cid, np.nan),
            "its anisotropy (s2/s1; near 1: no direction)": models.ratio.get(cid, np.nan),
        }
    return pd.DataFrame(summary)


def stateShares(run: AgRun, slotsPerNight: int = 12) -> pd.DataFrame:
    """Count each camera's states through the run, in slots of each night.

    Only the AG exposures in which some camera matched a star, on the
    plotted nights, are counted; a night spans 18:00 to 06:00 HST
    (`nightPosition`).

    Parameters
    ----------
    run : `AgRun`
        The run.
    slotsPerNight : `int`
        Slots in a night (12: an hour each).

    Returns
    -------
    shares : `pandas.DataFrame`
        One row per (``night``, ``slot``, ``agc_camera_id``) with an AG
        exposure: ``n`` exposures, how many of them in each state
        (``matched``, ``unmatched``, ``dark``, ``disabled``, and
        ``few_stars``, which is not judged) and ``start``, the time of the
        slot's first exposure. Cameras not in use are left out.
    """
    columns = [
        "night",
        "slot",
        "agc_camera_id",
        "n",
        "matched",
        "unmatched",
        "dark",
        "disabled",
        "few_stars",
        "start",
    ]
    nights = run.plottedNights
    frames = run.frames[run.frames.night.isin(nights) & (run.frames.n_cams > 0)]
    if frames.empty:
        return pd.DataFrame(columns=columns)
    position = nightPosition(frames.taken_at, nights)
    hours = (position - np.floor(position) - 0.05) / 0.9
    slot = np.clip(np.floor(hours * slotsPerNight), 0, slotsPerNight - 1).astype(int)
    names = {
        CameraState.OK: "matched",
        CameraState.UNMATCHED: "unmatched",
        CameraState.DARK: "dark",
        CameraState.DISABLED: "disabled",
        CameraState.FEW_STARS: "few_stars",
    }
    parts = []
    for cid in run.inUse:
        states = frames[f"state{cid}"]
        counts = pd.DataFrame({name: (states == state).astype(int) for state, name in names.items()})
        counts = counts.assign(night=frames.night, slot=slot, n=1, start=frames.taken_at)
        grouped = counts.groupby(["night", "slot"]).agg(
            n=("n", "sum"), start=("start", "min"), **{name: (name, "sum") for name in names.values()}
        )
        parts.append(grouped.reset_index().assign(agc_camera_id=cid))
    return pd.concat(parts, ignore_index=True)[columns]


def magnitudeSummary(run: AgRun) -> pd.DataFrame:
    """Return how often each camera matched its designed guide stars, by their magnitude.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    summary : `pandas.DataFrame`
        One row per (``agc_camera_id``, ``magnitude``), the lower edge of
        the bin, over the science visits of the plotted nights and the
        cameras in use: ``n_expected`` (guide stars in the designs, times
        AG exposures), ``n_good`` (good matches with them) and
        ``fraction``. Empty if the ``magnitudes`` table wasn't read.
    """
    columns = ["agc_camera_id", "magnitude", "n_expected", "n_good", "fraction"]
    magnitudes = run.magnitudes
    if magnitudes.empty:
        return pd.DataFrame(columns=columns)
    chosen = magnitudes[
        magnitudes.population.eq("science")
        & magnitudes.night.isin(run.plottedNights)
        & magnitudes.agc_camera_id.isin(run.inUse)
    ]
    summary = chosen.groupby(["agc_camera_id", "magnitude"])[["n_expected", "n_good"]].sum().reset_index()
    return summary.assign(fraction=summary.n_good / summary.n_expected)[columns]


def visitMatchFraction(run: AgRun) -> pd.DataFrame:
    """Return the share of each science visit's designed guide stars that were matched.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    fractions : `pandas.DataFrame`
        One row per science visit of the plotted nights: ``pfs_visit_id``,
        ``night``, ``n_expected`` (the design's guide stars on the cameras
        in use, times the visit's AG exposures), ``n_good`` (good matches
        with them) and ``fraction``. A guide star on a camera not in use
        was not available, and is not counted. Empty if the
        ``magnitudes`` table wasn't read.
    """
    columns = ["pfs_visit_id", "night", "n_expected", "n_good", "fraction"]
    magnitudes = run.magnitudes
    if magnitudes.empty:
        return pd.DataFrame(columns=columns)
    chosen = magnitudes[
        magnitudes.population.eq("science")
        & magnitudes.night.isin(run.plottedNights)
        & magnitudes.agc_camera_id.isin(run.inUse)
    ]
    visits = chosen.groupby(["pfs_visit_id", "night"])[["n_expected", "n_good"]].sum().reset_index()
    visits = visits[visits.n_expected > 0]
    return visits.assign(fraction=visits.n_good / visits.n_expected)[columns]


def detectionFlagShares(run: AgRun) -> pd.DataFrame:
    """Return the share of each camera's detections carrying each bad-detection flag, night by night.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    shares : `pandas.DataFrame`
        One row per (``night``, ``agc_camera_id``) on the plotted nights:
        ``n_spots`` and, per flag of `DETECTION_FLAG_COLUMNS`, the
        percentage of them carrying it (a detection can carry several).
        Empty if the read holds no flag counts.
    """
    names = list(DETECTION_FLAG_COLUMNS.values())
    cameras = run.cameras[run.cameras.night.isin(run.plottedNights)]
    if cameras.empty or not cameras[list(DETECTION_FLAG_COLUMNS)].notna().any().any():
        return pd.DataFrame(columns=["night", "agc_camera_id", "n_spots", *names])
    sums = cameras.groupby(["night", "agc_camera_id"])[["n_spots", *DETECTION_FLAG_COLUMNS]].sum()
    shares = sums[list(DETECTION_FLAG_COLUMNS)].div(sums.n_spots, axis=0) * 100
    return shares.rename(columns=DETECTION_FLAG_COLUMNS).assign(n_spots=sums.n_spots).reset_index()


def runSummary(run: AgRun, jitter: pd.DataFrame) -> dict:
    """Return the run's key numbers: the guide error, the matched stars and the guide loop.

    Parameters
    ----------
    run : `AgRun`
        The run.
    jitter : `pandas.DataFrame`
        From `visitGuideErrors`.

    Returns
    -------
    summary : `dict`
        ``nights``, ``exposures``, ``scienceVisits`` and ``measuredVisits``
        (those `visitGuideErrors` could measure);
        ``camerasInUse`` (their names); ``guideError``, per axis (``ra`` on
        the sky, ``dec``, ``inr``, and ``sky``, a visit's ΔRA and ΔDec in
        quadrature; arcsec) the median and 95th percentile over the measured
        science visits of the offsets' RMS (``size``, ``size95``) and of
        their jitter (``jitter``, ``jitter95``); ``matchedFraction``, the
        median and the 5th and 95th percentiles over the science visits of
        the share of the designed guide stars on the cameras in use that
        were matched (`visitMatchFraction`), and how many ``visits``;
        ``residualUm``, the median and 95th percentile over the science AG
        exposures of the good matches' distance from where the guider
        expected them; ``matchedStars``, the median and 5th percentile of
        the good matches per science AG exposure, the median per camera in
        use, and the percentage of exposures with fewer than
        ``minFitStars``; ``fwhm`` (arcsec, median of the good matches); and
        ``guideLoop``, the AG exposures whose offset was rejected, or not
        written. The per-exposure numbers (``residualUm``, ``matchedStars``)
        are of the science AG exposures taken with the shutters open, where
        the read has the shutter times. NaN where there is nothing to
        measure.
    """
    plotted = run.frames[run.frames.night.isin(run.plottedNights)]
    science = plotted[plotted.population.eq("science")]
    scienceVisits = int(science.pfs_visit_id.nunique())
    # What the spectrograph saw: the science AG exposures with the shutters open, where the read says which.
    if science.shutter.any():
        science = science[science.shutter]

    def percentile(values: pd.Series, q: float) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.percentile(values, q)) if len(values) else float("nan")

    guideError = {}
    for axis in ("ra", "dec", "inr"):
        guideError[axis] = {
            "size": percentile(jitter[f"{axis}_size"], 50),
            "size95": percentile(jitter[f"{axis}_size"], 95),
            "jitter": percentile(jitter[axis], 50),
            "jitter95": percentile(jitter[axis], 95),
        }
    # On the sky: a visit's ΔRA and ΔDec together, in quadrature.
    skySize, skyJitter = np.hypot(jitter.ra_size, jitter.dec_size), np.hypot(jitter.ra, jitter.dec)
    guideError["sky"] = {
        "size": percentile(skySize, 50),
        "size95": percentile(skySize, 95),
        "jitter": percentile(skyJitter, 50),
        "jitter95": percentile(skyJitter, 95),
    }
    matched = visitMatchFraction(run).fraction
    cameras = run.cameras[run.cameras.night.isin(run.plottedNights)]
    return {
        "nights": len(run.plottedNights),
        "exposures": len(plotted),
        "scienceVisits": scienceVisits,
        "measuredVisits": len(jitter),
        "camerasInUse": [f"AG{cid + 1}" for cid in run.inUse],
        "guideError": guideError,
        "residualUm": {
            "median": percentile(science.residual_um, 50),
            "p95": percentile(science.residual_um, 95),
        },
        "matchedStars": {
            "perExposure": percentile(science.n_good, 50),
            "p5": percentile(science.n_good, 5),
            "perCamera": {f"AG{cid + 1}": percentile(science[f"n_good{cid}"], 50) for cid in run.inUse},
            "underMinimumPct": 100 * float((science.n_good < run.config.minFitStars).mean())
            if len(science)
            else float("nan"),
        },
        "matchedFraction": {
            "visits": len(matched),
            "median": percentile(matched, 50),
            "p5": percentile(matched, 5),
            "p95": percentile(matched, 95),
        },
        "fwhm": percentile(cameras.fwhm, 50),
        "guideLoop": {
            "rejected": int(isRejectedOffset(plotted["mask"]).sum()),
            "noOffset": int(plotted["mask"].isna().sum()),
        },
    }
