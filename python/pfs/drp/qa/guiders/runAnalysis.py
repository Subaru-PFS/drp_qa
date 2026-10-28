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
    "describePeriod",
    "findDropouts",
    "findFlags",
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
    config : `RunConfig`
        The parameters it was prepared with.
    """

    frames: pd.DataFrame
    cameras: pd.DataFrame
    focus: pd.DataFrame
    nights: pd.Series
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
    """

    kind: str
    severe: bool
    text: str


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
    return AgRun(frames=frames, cameras=cameras, focus=focus, nights=nights, config=config)


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


def findFlags(run: AgRun, jitter: pd.DataFrame, dropouts: pd.DataFrame) -> list[Flag]:
    """List what the run's AG data show is wrong or worth a look.

    Parameters
    ----------
    run : `AgRun`
        The run.
    jitter : `pandas.DataFrame`
        From `settledJitter`.
    dropouts : `pandas.DataFrame`
        From `findDropouts`.

    Returns
    -------
    flags : `list` [`Flag`]
        Camera dropouts; guide frames rejected, missing, or accepted from
        too few stars; visits with unusual settled jitter; cameras with
        unusual image sizes. The FWHM of a camera's two halves is not
        flagged: they focus differently by design.
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
