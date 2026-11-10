"""`pfs.drp.qa.plotting.saturation` and the stack-free half of `pfs.drp.qa.saturationReport`."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pfs.drp.qa.plotting.saturation import plotSaturation
from pfs.drp.qa.saturationReport import hasSaturationColumns, reportMarkdown, saturatedCameras, writeReport

DATA_ID = {"instrument": "PFS", "visit": 153432, "arm": "r", "spectrograph": 2}


def makeImage(numSaturated: int = 3):
    """Return an image, its SAT plane and its lines: 8 fibers x 2 lines, the first ``numSaturated`` saturated."""
    rng = np.random.default_rng(1)
    image = rng.normal(0.0, 1.0, (200, 120))
    saturated = np.zeros(image.shape, dtype=bool)
    fiberId = np.repeat(np.arange(1, 9), 2)
    lines = pd.DataFrame(
        {
            "fiberId": fiberId,
            "description": "KrI",
            "wavelength": np.tile([760.15, 811.29], 8),
            "x": 12.0 * fiberId,
            "y": np.tile([50.0, 150.0], 8),
            "saturated": np.arange(16) < numSaturated,
        }
    )
    for line in lines[lines["saturated"]].itertuples():
        row, column = round(line.y), round(line.x)
        image[row - 3 : row + 4, column - 1 : column + 2] += 60000.0
        saturated[row - 2 : row + 3, column] = True
    return image, saturated, lines


def testPlotSaturationDrawsTheMapTheLinesAndTheCutouts():
    image, saturated, lines = makeImage()
    fig = plotSaturation(image, saturated, lines, title="153432 r2")
    mapAxes, barAxes, *cutouts = fig.axes
    assert "15 SAT pixels, 3 saturated lines" in mapAxes.get_title(loc="left")
    # Two lines are saturated (760.15 on two fibers, 811.29 on one): the shortest wavelength on top,
    # so the last tick (bars are drawn upwards).
    assert [tick.get_text() for tick in barAxes.get_yticklabels()] == ["KrI 811.29", "KrI 760.15"]
    # Negative control: with the longer wavelength the more saturated, it still comes second;
    # ordered by how saturated, it would be on top.
    longer = lines.assign(saturated=(lines["wavelength"] > 800) & (lines["fiberId"] <= 3))
    longer.loc[0, "saturated"] = True
    byWavelength = plotSaturation(image, saturated, longer)
    assert [tick.get_text() for tick in byWavelength.axes[1].get_yticklabels()] == [
        "KrI 811.29",
        "KrI 760.15",
    ]
    plt.close(byWavelength)
    # More lines than shown: the most saturated are kept, still in order of wavelength.
    fewer = plotSaturation(image, saturated, lines, maxLines=1)
    assert [tick.get_text() for tick in fewer.axes[1].get_yticklabels()] == ["KrI 760.15"]
    plt.close(fewer)
    assert sum(axes.axison for axes in cutouts) == 3
    plt.close(fig)


def testPlotSaturationOfAnUnsaturatedImage():
    """Nothing saturated still draws: the report may be asked for any camera."""
    image, saturated, lines = makeImage(numSaturated=0)
    fig = plotSaturation(image, saturated, lines)
    assert "0 SAT pixels, 0 saturated lines" in fig.axes[0].get_title(loc="left")
    plt.close(fig)


def testReportListsTheSaturatedLines(tmp_path):
    image, saturated, lines = makeImage()
    report = writeReport(tmp_path, DATA_ID, image, saturated, lines, 0.25, seqName="ImageQuality (Kr)")
    assert report.name == "saturation-153432-r2.md"
    assert (tmp_path / "saturation-153432-r2.png").stat().st_size > 0
    text = report.read_text()
    assert "# Saturation: visit 153432 r2 (ImageQuality (Kr))" in text
    assert "`nSaturatedLines`: 3 of 16 lines on lit fibers, on 2 of 8 fibers" in text
    assert text.index("| KrI | 760.150 | 2 | 8 |") < text.index("| KrI | 811.290 | 1 | 8 |")
    assert "No line is saturated." in reportMarkdown(
        DATA_ID, saturated, lines.assign(saturated=False), 0.0, "", "f.png"
    )


def testSaturatedCamerasAreThoseWithLinesOrPixels():
    metrics = pd.DataFrame(
        {
            "arm": ["r", "r", "b", "n"],
            "spectrograph": [2, 1, 1, 3],
            "nSaturatedLines": [5, 0, np.nan, 0],
            "pctSaturatedPixels": [0.4, 0.0, np.nan, 0.001],
        }
    )
    assert saturatedCameras(metrics) == [("n", 3), ("r", 2)]
    # Metrics written before the columns existed report nothing, and say so.
    assert saturatedCameras(metrics[["arm", "spectrograph"]]) == []
    assert hasSaturationColumns(metrics) and not hasSaturationColumns(metrics[["arm", "spectrograph"]])
