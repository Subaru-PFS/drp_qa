"""End-of-run analysis of the AG cameras and the guide loop.

Takes the tables of `pfs.drp.qa.guiders.queries.readAgRun`, never a
database. `prepareRun` turns them into one row per AG exposure (`AgRun`);
the other functions take that and return DataFrames or `Flag` lists, which
`pfs.drp.qa.guiders.runPlotting` draws and `pfs.drp.qa.guiders.runReport`
writes up.

Cameras are ``agc_camera_id``, 0-5 (AG1 is 0). They come in two groups,
AG1/3/5 and AG2/4/6, wired and placed separately, which can fail together
(Run 30 lost AG1/3/5), so the analysis reports them by group.

Times are HST. A night runs from noon to noon and is named by its evening's
date, as in `pfs.drp.qa.comparison.runs`.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "CAMERA_GROUPS",
    "SCIENCE_TYPES",
    "AgRun",
    "CameraState",
    "Flag",
    "RunConfig",
    "cameraSummary",
    "describePeriod",
    "findDropouts",
    "findFlags",
    "findRejectionClusters",
    "fitModelErrors",
    "groupSummary",
    "nightOf",
    "nightPosition",
    "onSkyRa",
    "prepareRun",
    "settledJitter",
]

#: The two groups of AG cameras, by ``agc_camera_id``.
CAMERA_GROUPS = {"AG1/3/5": (0, 2, 4), "AG2/4/6": (1, 3, 5)}

#: Sequence types whose guiding is judged.
SCIENCE_TYPES = ("scienceObject", "scienceObject_windowed")

#: 1 AG pixel is 0.14 arcsec.
ARCSEC_PER_PIX = 0.14

#: A Gaussian's FWHM over its sigma.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


class CameraState:
    """What a camera gave in an AG exposure in which some camera matched a star."""

    #: It matched at least one guide star.
    OK = 0
    #: It matched none, but its design had too few bright guide stars to expect one.
    FEW_STARS = 1
    #: It detected something but matched none of the design's guide stars.
    UNMATCHED = 2
    #: It detected nothing at all.
    DARK = 3
    #: No camera matched anything (the sky wasn't usable); the camera isn't judged.
    UNUSABLE = -1


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
        flagged on its own; shorter ones are summarised.
    nSettle : `int`
        Guide frames at the start of a visit left out of its settled jitter.
    minSettled : `int`
        Settled frames a visit needs for its jitter.
    minFitStars : `int`
        A guide offset from fewer good matches than this is flagged when the
        guider accepted it.
    jitterFactor : `float`
        A visit's settled jitter is flagged above this many times the
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
        from its detector's centre.
    maxModelError, modelErrorFactor : `float`
        A camera is flagged when its position-model error (µm, at
        ``modelErrorRadiusPix``) is over ``maxModelError`` and over
        ``modelErrorFactor`` times the cameras' median, in the run's nightly
        median. The error measured is never zero: the matches are noisy.
    """

    minDesignStars: int = 3
    minDropout: int = 30
    longDropout: int = 300
    nSettle: int = 5
    minSettled: int = 10
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
        other), ``pfs_design_id``, ``dec_center_designed``, ``n_good`` (good
        matches over the cameras), ``n_cams`` (cameras with one) and, per
        camera ``c``, ``n_spots{c}``, ``n_good{c}``, ``n_design{c}`` (its
        design's bright guide stars) and ``state{c}`` (a `CameraState`).
    cameras : `pandas.DataFrame`
        The ``cameras`` table with ``fwhm`` (arcsec), ``night`` and
        ``population``.
    focus : `pandas.DataFrame`
        The ``focus`` table with ``night``.
    nights : `pandas.Series`
        AG exposures per night.
    grid : `pandas.DataFrame`
        The ``grid`` table with ``night`` (empty if it wasn't read).
    config : `RunConfig`
        The parameters it was prepared with.
    """

    frames: pd.DataFrame
    cameras: pd.DataFrame
    focus: pd.DataFrame
    nights: pd.Series
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)
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

#: An AG detector's centre (pixels), about which pfs_utils' camera model rotates it.
DETECTOR_CENTRE_PIX = (511.5 + 24, 511.5 + 9)


def nightOf(times: pd.Series) -> pd.Series:
    """Return the night (midnight of its evening's date) of each time."""
    return (pd.to_datetime(times) - pd.Timedelta(hours=12)).dt.normalize()


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
    frames = frames.merge(
        visits[["pfs_visit_id", "pfs_design_id", "dec_center_designed", "sequence_type"]],
        how="left",
        on="pfs_visit_id",
    )
    frames["night"] = nightOf(frames.taken_at)
    frames["population"] = _population(frames.sequence_type)

    cameras = tables["cameras"]
    perCamera = cameras.groupby(["agc_exposure_id", "agc_camera_id"])[["n_spots", "n_good"]].sum()
    designStars = (
        tables["designStars"]
        .pivot_table(index="pfs_design_id", columns="agc_camera_id", values="n_bright", aggfunc="sum")
        .reindex(columns=range(6))
        .fillna(0)
    )
    columns = {}
    for name in ("n_spots", "n_good"):
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
                frames[f"n_good{cid}"] > 0,
                frames[f"n_design{cid}"] < config.minDesignStars,
                frames[f"n_spots{cid}"] > 0,
            ],
            [CameraState.OK, CameraState.FEW_STARS, CameraState.UNMATCHED],
            CameraState.DARK,
        )
        frames[f"state{cid}"] = np.where(usable, state, CameraState.UNUSABLE)

    cameras = cameras.assign(fwhm=cameras.sigma_pix * FWHM_PER_SIGMA * ARCSEC_PER_PIX).merge(
        frames[["agc_exposure_id", "night", "population"]], on="agc_exposure_id"
    )
    focus = tables["focus"].assign(night=lambda df: nightOf(df.created_at))
    nights = frames.groupby("night").size()
    grid = tables.get("grid")
    if grid is None:
        grid = pd.DataFrame(columns=GRID_COLUMNS)
    grid = grid.merge(frames[["agc_exposure_id", "night"]], on="agc_exposure_id")
    return AgRun(frames=frames, cameras=cameras, focus=focus, nights=nights, grid=grid, config=config)


def onSkyRa(frames: pd.DataFrame) -> pd.Series:
    """Return the guide offsets' ΔRA on the sky, ΔRA cos(Dec) (arcsec).

    ``guide_delta_ra`` is a coordinate offset. The guide Dec is used where
    there is one, else the design's.
    """
    dec = frames.guide_dec.fillna(frames.dec_center_designed)
    return frames.guide_delta_ra * np.cos(np.radians(dec))


def settledJitter(run: AgRun) -> pd.DataFrame:
    """Return the settled frame-to-frame jitter of each science visit.

    A visit's settled frames are its accepted guide offsets after its first
    ``nSettle`` guide frames, rejected ones counted among those (the loop is
    still converging, whatever it did with them); visits with fewer than
    ``minSettled`` settled frames are left out. The
    jitter is the standard deviation of consecutive differences over √2,
    which a slow drift barely changes.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    jitter : `pandas.DataFrame`
        One row per visit: ``pfs_visit_id``, ``night``, ``taken_at`` (its
        first settled frame), ``n_frames``, ``n_good`` and ``n_cams`` (median
        over its frames), and ``ra`` (on the sky), ``dec`` and ``inr``
        (arcsec).
    """
    config = run.config
    frames = run.frames[run.frames.population.eq("science") & run.frames["mask"].notna()].copy()
    frames["n"] = frames.groupby("pfs_visit_id").cumcount() + 1
    frames = frames[(frames.n > config.nSettle) & frames["mask"].eq(0)]
    frames["ra"] = onSkyRa(frames)

    def jitter(values: pd.Series) -> float:
        differences = np.diff(values.to_numpy(float))
        differences = differences[np.isfinite(differences)]
        return float(np.std(differences) / np.sqrt(2)) if len(differences) else np.nan

    columns = ["pfs_visit_id", "night", "taken_at", "n_frames", "n_good", "n_cams", "ra", "dec", "inr"]
    rows = []
    for visit, group in frames.groupby("pfs_visit_id"):
        if len(group) < config.minSettled:
            continue
        rows.append(
            [
                visit,
                group.night.iloc[0],
                group.taken_at.iloc[0],
                len(group),
                group.n_good.median(),
                int(group.n_cams.median()),
                jitter(group.ra),
                jitter(group.guide_delta_dec),
                jitter(group.guide_delta_insrot),
            ]
        )
    return pd.DataFrame(rows, columns=columns)


def findDropouts(run: AgRun) -> pd.DataFrame:
    """Find the stretches in which a camera matched no star although it should have.

    Only the AG exposures in which some camera matched a star, and in which
    the camera's design had at least ``minDesignStars`` bright guide stars,
    are judged (`CameraState` OK, UNMATCHED or DARK). A dropout is at least
    ``minDropout`` consecutive judged exposures that aren't OK.

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
        (over 90% of its exposures detected nothing) and ``last_good`` (the
        last match before it, or NaT).
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
        "last_good",
    ]
    judged = [CameraState.OK, CameraState.UNMATCHED, CameraState.DARK]
    rows = []
    for cid in range(6):
        frames = run.frames[run.frames[f"state{cid}"].isin(judged)].reset_index(drop=True)
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
        One column per group of `CAMERA_GROUPS`, one row per quantity: the
        percentage of judged exposures in which its cameras matched a star;
        the median good matches per camera and exposure; the median FWHM
        (arcsec) on each detector half; the median of its cameras' focus
        minus the six cameras' median, and of its own median minus the
        six cameras' (µm), over exposures with all six focus values: what
        the guider's focus would read with only this group.
    """
    frames = run.frames[run.frames.night.isin(run.plottedNights)]
    cameras = run.cameras[run.cameras.night.isin(run.plottedNights)]
    zColumns = [f"guide_delta_z{cid + 1}" for cid in range(6)]
    z = frames[zColumns].dropna().to_numpy() * 1e3
    zMedian = np.median(z, axis=1) if len(z) else np.array([])
    judged = [CameraState.OK, CameraState.UNMATCHED, CameraState.DARK]
    summary = {}
    for name, cids in CAMERA_GROUPS.items():
        states = frames[[f"state{cid}" for cid in cids]].to_numpy()
        inGroup = cameras.agc_camera_id.isin(cids)
        summary[name] = {
            "matched (% of judged exposures)": 100
            * np.mean(states[np.isin(states, judged)] == CameraState.OK),
            "good matches per camera": float(np.median(frames[[f"n_good{cid}" for cid in cids]].to_numpy())),
            "FWHM, glass half (″)": cameras[inGroup & cameras.right_half.eq(0)].fwhm.median(),
            "FWHM, no-glass half (″)": cameras[inGroup & cameras.right_half.eq(1)].fwhm.median(),
            "focus minus 6-camera median (µm)": (
                float(np.median(z[:, list(cids)] - zMedian[:, None])) if len(z) else np.nan
            ),
            "focus from this group alone minus dZ (µm)": (
                float(np.median(np.median(z[:, list(cids)], axis=1) - zMedian)) if len(z) else np.nan
            ),
        }
    return pd.DataFrame(summary)


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
        From `settledJitter`.
    dropouts : `pandas.DataFrame`
        From `findDropouts`.
    clusters : `pandas.DataFrame`, optional
        From `findRejectionClusters`.
    modelErrors : `pandas.DataFrame`, optional
        From `fitModelErrors`.

    Returns
    -------
    flags : `list` [`Flag`]
        Camera dropouts; cameras whose matches are often rejected in a
        cluster on one side of the detector, or whose position model is off; guide frames rejected,
        missing, or accepted from too few stars; visits with unusual settled
        jitter; cameras with unusual image sizes. The FWHM of a camera's two
        halves is not flagged: they focus differently by design.
    """
    config = run.config
    flags = []
    long = dropouts.dark | (dropouts.n_frames >= config.longDropout)
    for _, d in dropouts[long].iterrows():
        what = "no detections at all" if d.dark else "detections but no matched star"
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
    short = dropouts[~long]
    if len(short):
        counts = short.groupby("agc_camera_id").n_frames.agg(["size", "sum"])
        flags.append(
            Flag(
                "camera",
                False,
                f"Shorter stretches ({config.minDropout}+ exposures) in which a camera with "
                f"{config.minDesignStars}+ bright designed guide stars matched none: "
                + ", ".join(f"AG{cid + 1} {n}× ({total} exposures)" for cid, (n, total) in counts.iterrows())
                + ".",
            )
        )

    if clusters is not None and len(clusters):
        alone = clusters[clusters.alone.astype(bool)]
        counts = alone.groupby("agc_camera_id").size().reindex(range(6), fill_value=0)
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
        median = modelErrors.groupby("agc_camera_id").error_um.median()
        typical = median.median()
        for cid, error in median.items():
            if error > max(config.maxModelError, config.modelErrorFactor * typical):
                angle = _meanAngle(2 * modelErrors[modelErrors.agc_camera_id == cid].angle_deg) / 2 % 180
                flags.append(
                    Flag(
                        "camera",
                        False,
                        f"AG{cid + 1}: its position model is off by {error:.1f} µm at "
                        f"{config.modelErrorRadiusPix:g} px from its detector's centre, growing toward {angle:.0f}° "
                        f"(nightly median; the cameras' median {typical:.1f} µm).",
                        int(cid),
                    )
                )

    frames = run.frames
    mask = frames["mask"]
    rejected = frames[(mask.fillna(0).astype(int) & 1) == 1]
    if len(rejected):
        flags.append(
            Flag(
                "guiding",
                True,
                f"**{len(rejected)} guide frames rejected** (INVALID_OFFSET: not sent to the telescope), in "
                f"{rejected.pfs_visit_id.nunique()} visits: {_visitList(sorted(rejected.pfs_visit_id.unique()))}.",
            )
        )
    fewStars = frames[mask.eq(0) & (frames.n_good < config.minFitStars)]
    if len(fewStars):
        counts = fewStars.pfs_visit_id.value_counts()
        flags.append(
            Flag(
                "guiding",
                True,
                f"**{len(fewStars)} guide offsets accepted from fewer than {config.minFitStars} matched stars** "
                "(sent to the telescope when under max_correction): "
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
        full = jitter[jitter.n_cams >= 6]
        reference = full if len(full) else jitter
        ratio = pd.concat([jitter[axis] / reference[axis].median() for axis in ("ra", "dec", "inr")], axis=1)
        jitter = jitter.assign(ratio=ratio.max(axis=1))
        high = jitter[jitter.ratio > config.jitterFactor].sort_values("ratio", ascending=False)
        if len(high):
            flags.append(
                Flag(
                    "guiding",
                    False,
                    f"{len(high)} of {len(jitter)} science visits have settled jitter over "
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
        other camera had a cluster in that AG exposure).
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
    grid = run.grid
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


def fitModelErrors(run: AgRun) -> pd.DataFrame:
    """Fit how far off each camera's position model is, night by night.

    From the ``grid``, each cell's good matches give a mean position
    (pixels) and mean residual (measured minus expected, µm). In each AG
    exposure each camera's mean residual and position are subtracted,
    removing the pointing, which moves a camera's stars together; per night
    and camera a 2x2 matrix A is then fitted, residual = A (position - mean),
    weighted by the matches. A camera model whose position, rotation or scale
    is right leaves A at the noise; its largest singular value, times
    ``modelErrorRadiusPix``, is the error at that distance from the centre,
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
        fastest in; ``n_good``; and ``A11``-``A22`` (µm per pixel).
    """
    columns = ["night", "agc_camera_id", "error_um", "angle_deg", "n_good", "A11", "A12", "A21", "A22"]
    grid = run.grid[run.grid.n_good > 0].dropna(subset=["x_good_pix", "y_good_pix", "dx_um", "dy_um"])
    if grid.empty:
        return pd.DataFrame(columns=columns)
    grid = grid.copy()
    keys = [grid.agc_exposure_id, grid.agc_camera_id]
    total = grid.n_good.groupby(keys).transform("sum")
    for column in ("x_good_pix", "y_good_pix", "dx_um", "dy_um"):
        grid[column] = grid[column] - (grid[column] * grid.n_good).groupby(keys).transform("sum") / total
    rows = []
    for (night, cid), cells in grid.groupby(["night", "agc_camera_id"]):
        weight = np.sqrt(cells.n_good.to_numpy(float))
        X = cells[["x_good_pix", "y_good_pix"]].to_numpy() * weight[:, None]
        if np.linalg.matrix_rank(X) < 2:
            continue
        A = np.vstack(
            [np.linalg.lstsq(X, cells[c].to_numpy() * weight, rcond=None)[0] for c in ("dx_um", "dy_um")]
        )
        _, singular, vt = np.linalg.svd(A)
        rows.append(
            [
                night,
                cid,
                singular[0] * run.config.modelErrorRadiusPix,
                np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180,
                int(cells.n_good.sum()),
                *A.ravel(),
            ]
        )
    return pd.DataFrame(rows, columns=columns)


def cameraSummary(run: AgRun, clusters: pd.DataFrame, modelErrors: pd.DataFrame) -> pd.DataFrame:
    """Summarise each camera over the run.

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
        each half, its focus minus the six cameras' median (µm), its AG
        exposures with a cluster of rejected matches on it alone, their mean
        direction and its consistency, and its position-model error (µm, nightly median) and the
        direction it grows in.
    """
    frames = run.frames[run.frames.night.isin(run.plottedNights)]
    cameras = run.cameras[run.cameras.night.isin(run.plottedNights)]
    zColumns = [f"guide_delta_z{cid + 1}" for cid in range(6)]
    z = frames[zColumns].dropna().to_numpy() * 1e3
    zMedian = np.median(z, axis=1) if len(z) else np.array([])
    judged = [CameraState.OK, CameraState.UNMATCHED, CameraState.DARK]
    radius = run.config.modelErrorRadiusPix
    summary = {}
    for cid in range(6):
        states = frames[f"state{cid}"]
        states = states[states.isin(judged)]
        own = clusters[(clusters.agc_camera_id == cid) & clusters.alone.astype(bool)]
        errors = modelErrors[modelErrors.agc_camera_id == cid]
        summary[f"AG{cid + 1}"] = {
            "matched (% of judged exposures)": 100 * float((states == CameraState.OK).mean())
            if len(states)
            else np.nan,
            "good matches per exposure": float(frames[f"n_good{cid}"].median()) if len(frames) else np.nan,
            "FWHM, glass half (″)": cameras[
                (cameras.agc_camera_id == cid) & cameras.right_half.eq(0)
            ].fwhm.median(),
            "FWHM, no-glass half (″)": cameras[
                (cameras.agc_camera_id == cid) & cameras.right_half.eq(1)
            ].fwhm.median(),
            "focus minus 6-camera median (µm)": float(np.median(z[:, cid] - zMedian)) if len(z) else np.nan,
            "exposures with rejected matches clustered on it alone": len(own),
            "their direction (°)": _meanAngle(own.angle_deg) if len(own) else np.nan,
            "their direction's consistency (0-1)": _consistency(own.angle_deg),
            f"position-model error at {radius:g} px (µm)": errors.error_um.median()
            if len(errors)
            else np.nan,
            "its direction (°, 0-180)": _meanAngle(2 * errors.angle_deg) / 2 % 180 if len(errors) else np.nan,
        }
    return pd.DataFrame(summary)
