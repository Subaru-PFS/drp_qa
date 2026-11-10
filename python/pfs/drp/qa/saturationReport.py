"""A report on the saturation of a detector image.

For a visit, each saturated detector (``nSaturatedLines`` or
``pctSaturatedPixels`` above zero in its ``iqQaMetrics``), or the cameras
named, gets a figure (`pfs.drp.qa.plotting.saturation.plotSaturation`) and a
Markdown file listing the saturated lines, written to a directory:
``saturation-<visit>-<camera>.png`` and ``.md``.

`readDetector` is the only code here touching a Butler, which it is given, and
it needs the stack for the calexp; `writeReport` is stack-free. The
measurement is ``imageQualityQa``'s own (`pfs.drp.qa.metrics.saturation`), made
again from the calexp, so a report can be made for an image the task has not
judged.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.lineFlux import TRACE
from pfs.drp.qa.metrics.saturation import measureSaturation, saturatedLineSummary

__all__ = [
    "hasSaturationColumns",
    "main",
    "readDetector",
    "reportMarkdown",
    "saturatedCameras",
    "writeReport",
]


def hasSaturationColumns(metrics: pd.DataFrame) -> bool:
    """Return whether ``iqQaMetrics`` rows carry the saturation columns at all."""
    return "nSaturatedLines" in metrics.columns or "pctSaturatedPixels" in metrics.columns


def saturatedCameras(metrics: pd.DataFrame) -> list[tuple[str, int]]:
    """Return the ``(arm, spectrograph)`` of the saturated detector images in ``iqQaMetrics`` rows."""
    lines = metrics.get("nSaturatedLines", pd.Series(0.0, index=metrics.index)).fillna(0)
    pixels = metrics.get("pctSaturatedPixels", pd.Series(0.0, index=metrics.index)).fillna(0)
    hit = metrics[(lines > 0) | (pixels > 0)].sort_values(["arm", "spectrograph"])
    return [
        (str(arm), int(spectrograph))
        for arm, spectrograph in zip(hit["arm"], hit["spectrograph"], strict=True)
    ]


def readDetector(butler: Any, dataId: dict) -> dict:
    """Read and measure one detector image.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler whose collections hold the visit's ``calexp``, ``detectorMap``
        and ``lines``, and its ``pfsConfig`` if any.
    dataId : `dict`
        ``instrument``, ``visit``, ``arm`` and ``spectrograph``.

    Returns
    -------
    `dict`
        The arguments of `writeReport` other than the directory: ``image``,
        ``saturated``, ``lines`` (lamp or sky lines on lit fibers, with their
        expected ``x``, ``y`` and ``saturated``), ``pctSaturatedPixels`` and
        ``seqName``.
    """
    from pfs.drp.qa.imageQualityQa import engineeringLit, litFibers

    calexp = butler.get("calexp", dataId)
    detectorMap = butler.get("detectorMap", dataId)
    data = butler.get("lines", dataId).data
    pfsConfig = None
    if butler.exists("pfsConfig", instrument=dataId["instrument"], visit=dataId["visit"]):
        pfsConfig = butler.get("pfsConfig", instrument=dataId["instrument"], visit=dataId["visit"])
    engineering = pfsConfig is not None and engineeringLit(pfsConfig, dataId["spectrograph"])
    lit, _ = litFibers(pfsConfig, detectorMap.fiberId, engineering=engineering)

    saturated = (calexp.mask.array & calexp.mask.getPlaneBitMask(["SAT"])) != 0
    lineSaturated, x, y, pctSaturatedPixels = measureSaturation(saturated, detectorMap, data, lit)
    lines = pd.DataFrame(
        {
            "fiberId": data["fiberId"].to_numpy(),
            "description": data["description"].to_numpy(dtype=object),
            "wavelength": data["wavelength"].to_numpy(dtype=float),
            "x": x,
            "y": y,
            "saturated": lineSaturated,
        }
    )
    keep = lines["description"] != TRACE
    if lit is not None:
        keep &= lines["fiberId"].isin(lit)
    return {
        "image": np.asarray(calexp.image.array),
        "saturated": saturated,
        "lines": lines[keep].reset_index(drop=True),
        "pctSaturatedPixels": pctSaturatedPixels,
        "seqName": str(calexp.getMetadata().get("W_SEQNAM") or "").strip(),
    }


def reportMarkdown(
    dataId: dict,
    saturated: np.ndarray,
    lines: pd.DataFrame,
    pctSaturatedPixels: float,
    seqName: str,
    figure: str,
) -> str:
    """Return the Markdown of one detector image's report.

    Parameters
    ----------
    dataId : `dict`
        ``visit``, ``arm`` and ``spectrograph``.
    saturated : `numpy.ndarray` [`bool`]
        The calexp's ``SAT`` pixels.
    lines : `pandas.DataFrame`
        As `readDetector` returns.
    pctSaturatedPixels : `float`
        Percentage of the lit fibers' aperture pixels that are saturated.
    seqName : `str`
        The sequence name, for the heading.
    figure : `str`
        File name of the figure, beside the report.

    Returns
    -------
    `str`
        A heading, the counts, the figure, and each saturated line with the
        fibers it is saturated on.
    """
    camera = f"{dataId['arm']}{dataId['spectrograph']}"
    hit = lines[lines["saturated"].to_numpy(dtype=bool)]
    text = [
        f"# Saturation: visit {dataId['visit']} {camera}" + (f" ({seqName})" if seqName else ""),
        "",
        f"- SAT pixels on the detector: {int(np.asarray(saturated).sum()):,}",
        f"- `pctSaturatedPixels` (of the lit fibers' apertures): {pctSaturatedPixels:.3f} %",
        f"- `nSaturatedLines`: {len(hit):,} of {len(lines):,} lines on lit fibers,"
        f" on {hit['fiberId'].nunique()} of {lines['fiberId'].nunique()} fibers",
        "",
        f"![saturation]({figure})",
        "",
    ]
    summary = saturatedLineSummary(lines)
    if len(summary):
        text += [
            "| species | wavelength (nm) | fibers saturated | fibers with the line |",
            "|---|---|---|---|",
        ]
        text += [
            f"| {row.description} | {row.wavelength:.3f} | {row.nSaturated} | {row.nFibers} |"
            for row in summary.itertuples()
        ]
    else:
        text += ["No line is saturated."]
    return "\n".join(text) + "\n"


def writeReport(
    outDir: str | Path,
    dataId: dict,
    image: np.ndarray,
    saturated: np.ndarray,
    lines: pd.DataFrame,
    pctSaturatedPixels: float,
    seqName: str = "",
) -> Path:
    """Write one detector image's figure and Markdown report; return the report's path."""
    from matplotlib import pyplot as plt

    from pfs.drp.qa.plotting.saturation import plotSaturation

    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    camera = f"{dataId['arm']}{dataId['spectrograph']}"
    stem = f"saturation-{dataId['visit']}-{camera}"
    title = f"{dataId['visit']} {camera}" + (f"  {seqName}" if seqName else "")
    fig = plotSaturation(image, saturated, lines, title=title)
    fig.savefig(outDir / f"{stem}.png", dpi=150)
    plt.close(fig)
    report = outDir / f"{stem}.md"
    report.write_text(reportMarkdown(dataId, saturated, lines, pctSaturatedPixels, seqName, f"{stem}.png"))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the saturation of a visit's detector images.")
    parser.add_argument("-b", "--butler", default="/work/datastore", help="Butler repository")
    parser.add_argument(
        "-c",
        "--collections",
        required=True,
        help="comma-separated collections: the QA outputs, then the reductions",
    )
    parser.add_argument("--visit", type=int, required=True)
    parser.add_argument(
        "--cameras",
        nargs="+",
        metavar="CAMERA",
        help="e.g. r1 r2; default: those iqQaMetrics says are saturated",
    )
    parser.add_argument("--instrument", default="PFS")
    parser.add_argument("-o", "--out", default=".", help="directory to write to")
    args = parser.parse_args(argv)

    from lsst.daf.butler import Butler

    butler = Butler(args.butler, collections=args.collections.split(","), writeable=False)
    if args.cameras:
        cameras = [(camera[0], int(camera[1:])) for camera in args.cameras]
    else:
        # findFirst: each camera once, from the first collection holding it, not from a superseded RUN too.
        refs = butler.registry.queryDatasets(
            "iqQaMetrics", instrument=args.instrument, visit=args.visit, findFirst=True
        )
        frames = [butler.get(ref) for ref in refs]
        if frames and not hasSaturationColumns(pd.concat(frames, ignore_index=True)):
            print(
                f"visit {args.visit}: these iqQaMetrics were written before saturation was measured;"
                " name the cameras with --cameras",
                flush=True,
            )
            return 1
        cameras = saturatedCameras(pd.concat(frames, ignore_index=True)) if frames else []
        print(f"visit {args.visit}: {len(cameras)} of {len(frames)} detector images saturated", flush=True)
    for number, (arm, spectrograph) in enumerate(cameras, start=1):
        dataId = {
            "instrument": args.instrument,
            "visit": args.visit,
            "arm": arm,
            "spectrograph": spectrograph,
        }
        report = writeReport(args.out, dataId, **readDetector(butler, dataId))
        print(f"[{number}/{len(cameras)}] {report}", flush=True)
    return 0
