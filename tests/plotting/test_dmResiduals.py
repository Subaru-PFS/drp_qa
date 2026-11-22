"""Tests for `pfs.drp.qa.plotting.dmResiduals.plotDetectorResiduals`.

Stack-free: a synthetic ``dmLineResidualData`` frame (current schema, with the km/s
columns, realistic per-fiber trace sampling and several lamp species) drives the figure.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pfs.drp.qa.plotting.dmResiduals import plotDetectorResiduals

_W, _H = 4096, 4176


def _lineData(seed: int = 0, *, kms: bool = True) -> pd.DataFrame:
    """Return a synthetic one-detector ``dmLineResidualData`` frame.

    Traces are sampled at varied rows per fiber (no artificial banding); lamp lines sit at
    discrete wavelengths of three species. The calib residual carries a coherent drift (a net
    offset and a gradient); the visit residual is centred noise with a few outliers.
    """
    rng = np.random.default_rng(seed)
    fiberIds = np.linspace(2, 650, 120).astype(int)
    fiberX = np.interp(fiberIds, (fiberIds.min(), fiberIds.max()), (120, _W - 120))

    rows = []
    for fid, x0 in zip(fiberIds, fiberX, strict=True):
        for y in np.sort(rng.uniform(150, _H - 150, 20)):
            rows.append(("Trace", fid, np.nan, x0 + rng.normal(0, 3), y))
    species = {"NeI": (955, 1010, 1120), "HgI": (980, 1040, 1130), "CdI": (970, 1050)}
    for name, wavelengths in species.items():
        for fid, x0 in zip(fiberIds[::2], fiberX[::2], strict=True):
            for wl in wavelengths:
                y = np.interp(wl, (950, 1250), (150, _H - 150)) + rng.normal(0, 10)
                rows.append((name, fid, wl, x0 + rng.normal(0, 5), y))

    df = pd.DataFrame(rows, columns=["description", "fiberId", "wavelength", "x", "y"])
    n = len(df)
    df["visit"], df["arm"], df["spectrograph"] = 152000, "n", 3
    df["xErr"], df["yErr"] = 0.02, 0.02
    df["statusType"] = rng.choice(["USED", "RESERVED"], n, p=[0.8, 0.2])
    xn, yn = (df["x"] - _W / 2) / (_W / 2), (df["y"] - _H / 2) / (_H / 2)
    df["dxCalib"] = 0.06 + 0.05 * xn + rng.normal(0, 0.02, n)  # coherent drift
    df["dyCalib"] = -0.04 + 0.03 * yn + rng.normal(0, 0.02, n)
    df["dxVisit"] = rng.normal(0, 0.03, n)
    df["dyVisit"] = rng.normal(0, 0.03, n)
    isTrace = df["description"] == "Trace"
    df.loc[isTrace, ["dyCalib", "dyVisit"]] = np.nan  # traces measure x only
    if kms:
        df["dyKmsCalib"] = df["dyCalib"] * 25.0
        df["dyKmsVisit"] = df["dyVisit"] * 25.0
    return df


def _ylabels(fig: Figure) -> list[str]:
    return [ax.get_ylabel() for ax in fig.axes]


def _legendTexts(fig: Figure) -> list[str]:
    return [t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()]


def _dashed(fig: Figure) -> list:
    return [ln for ax in fig.axes for ln in ax.get_lines() if ln.get_linestyle() == "--"]


def testBuildsWithExpectedPanels():
    fig = plotDetectorResiduals(_lineData(), title="n3 152000")
    # Six panels (2 rows × [calib map, visit map, trend]) plus one colorbar per row.
    assert len(fig.axes) == 8
    assert "DetectorMap residuals" in fig.get_suptitle()


def testKmsUsedWhenStored_pixelFallbackWhenNot():
    withKms = plotDetectorResiduals(_lineData(kms=True))
    assert any("km/s" in label for label in _ylabels(withKms))
    # Negative control: a collection without the km/s columns shows the Δy trend in pixels,
    # never km/s.
    withoutKms = plotDetectorResiduals(_lineData(kms=False))
    assert not any("km/s" in label for label in _ylabels(withoutKms))
    assert any(label == "residual (pix)" for label in _ylabels(withoutKms))


def testGateBandsDrawnOnlyWithLimits():
    limits = {"dx": (0.1, 0.2), "dy": (5.0, 10.0)}
    withBands = plotDetectorResiduals(_lineData(), limits=limits)
    # warn and fail, either side of zero, in both trend panels: 2 × 2 × 2.
    assert len(_dashed(withBands)) == 8
    # A one-sided limit (fail only) on one row draws just that band (±) and only there.
    assert len(_dashed(plotDetectorResiduals(_lineData(), limits={"dx": (None, 0.2)}))) == 2
    # Negative control: no limits, no gate bands (the only reference line is the solid zero).
    assert _dashed(plotDetectorResiduals(_lineData())) == []
    # Negative control: the Greek key the internals use is NOT the public key, so it draws nothing.
    assert _dashed(plotDetectorResiduals(_lineData(), limits={"Δx": (0.1, 0.2)})) == []


def testSpeciesAndReferencesInLegend():
    texts = _legendTexts(plotDetectorResiduals(_lineData()))
    assert {"NeI", "HgI", "CdI"} <= set(texts)
    assert any("vs calib (drift)" in t for t in texts)
    assert any("vs visit (scatter)" in t for t in texts)


def testStatsBoxReportsBothReferences():
    fig = plotDetectorResiduals(_lineData())
    boxes = [t.get_text() for ax in fig.axes for t in ax.texts if "reserved" in t.get_text()]
    assert boxes, "each trend panel should carry a reserved-stats box"
    assert all("Calib" in b and "Visit" in b for b in boxes)
    # The synthetic calib residual is a +0.06 px drift, the visit residual is centred: the
    # Δx box must show a clearly non-zero calib median and a near-zero visit median.
    deltaX = next(b for b in boxes if "pix" in b)
    calibMedian = float(deltaX.split("Calib")[1].split("med")[1].split()[0])
    visitMedian = float(deltaX.split("Visit")[1].split("med")[1].split()[0])
    assert abs(calibMedian) > 0.03  # drift is seen
    assert abs(visitMedian) < abs(calibMedian)  # Negative control: visit is not the drift


def testUnfittedFrameStillClipsScale():
    # An UNFITTED frame has no RESERVED lines; a lone wild line must not autoscale the trend
    # (the y-scale falls back to all lines' robust spread, not the reserved ones).
    data = _lineData()
    data = data[data["description"] != "Trace"].copy()
    data["statusType"] = "UNFITTED"
    data.iloc[0, data.columns.get_loc("dxVisit")] = 50.0  # one wild outlier
    fig = plotDetectorResiduals(data)
    lo, hi = fig.axes[3].get_ylim()  # the Δx trend (map, map, colorbar, trend)
    assert hi - lo < 10  # held, not stretched to the 50 px outlier


def testTraceOnlyFrameDrawsSpatialOnly():
    # A quartz/trace frame has no lines: Δy is empty but the figure still builds.
    traces = _lineData()
    traces = traces[traces["description"] == "Trace"].copy()
    fig = plotDetectorResiduals(traces, title="n3 152000 (trace only)")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 8
