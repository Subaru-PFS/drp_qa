#!/usr/bin/env python3
import argparse
import logging
import os

from lsst.daf.butler import Butler
from pfs.drp.qa.skySubtraction.prepare import prepareDataset
from pfs.drp.qa.skySubtraction.summaryPlots import (
    plot_1d_spectrograph, plot_2d_spectrograph, plot_outlier_summary, plot_vs_sky_brightness
)
from pfs.drp.stella.fitFocalPlane import FitBlockedOversampledSplineConfig

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Change to DEBUG for more details
)


def saveFigure(fig, output_dir, function_name):
    """Save a matplotlib figure as a PNG file in the output directory."""
    filename = os.path.join(output_dir, f"{function_name}.png")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    logging.info(f"Saved plot: {filename}")


def availableArms(butler, dataId, arms):
    """Return available and unavailable arms for that visit and spectrograph."""
    unavailable = []
    available = []

    for arm in arms:
        try:
            butler.get('pfsArm', dataId, arm=arm)
            available.append(arm)
        except LookupError:
            unavailable.append(arm)

    return available, unavailable


def evaluateSkySubtractionQA(collection, visit, spectrograph, output_dir='.', arms=['b', 'r'],
                             plot_1d=True, plot_2d=True, plot_outlier=True, plot_sky_brightness=True,
                             **fitSkyModelConfigKwargs):
    """
    Run sky subtraction QA plots and save results as PNG images.

    Parameters
    ----------
    collection : `str`
        Butler collection name.
    visit : `int`
        Visit number.
    spectrograph : `int`
        Spectrograph number.
    arms : `list` of `str`, optional
        List of spectral arms to process (default: `['b', 'r']`).
    plot_1d, plot_2d, plot_outlier, plot_sky_brightness : `bool`, optional
        Flags to enable or disable specific plots.
    blockSize, rejIterations, rejThreshold, mask, oversample : optional
        Sky model fitting parameters. If None, defaults are taken from the butler configuration.

    Saves
    -----
    - PNG plots in a directory `plots/visit_{visit}_spec_{spectrograph}/`
    """
    butler = Butler('/work/datastore', collections=[collection])

    # Get default sky model configuration
    fitSkyModelConfig = FitBlockedOversampledSplineConfig()
    defaultConfig = butler.get("mergeArms_config").fitSkyModel.toDict()

    # Update only if user provides values
    for k, v in fitSkyModelConfigKwargs.items():
        if v is not None:  # Allow 0 or False as valid values
            defaultConfig[k] = v

    fitSkyModelConfig.update(**defaultConfig)

    # Define data identifier
    dataId = dict(spectrograph=spectrograph, visit=visit)

    arms, unavailable = availableArms(butler, dataId, arms)
    if unavailable:
        logging.warning(f"Skipping unavailable arms: {unavailable}")
    if not arms:
        logging.error(f"No pfsArm available for dataId: {dataId}, collection: {collection}")
        raise RuntimeError(f'No pfsArm available for dataId: {dataId}, collection: {collection}')

    logging.info(f"Preparing Sky Subtraction QA for dataId {dataId} with arms {arms}.")
    logging.debug(f"Using fitSkyModelConfig: {fitSkyModelConfig}")

    # Prepare dataset
    hold, holdAsDict, plotId = prepareDataset(collection, dataId, arms, fitSkyModelConfig)

    # Create output directory
    output_dir = os.path.abspath(os.path.join(output_dir, f"plots/visit_{visit}_spec_{spectrograph}"))
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save plots
    if plot_1d:
        fig, ax_dict = plot_1d_spectrograph(holdAsDict, plotId, arms)
        saveFigure(fig, output_dir, "plot_1d_spectrograph")

    if plot_2d:
        fig, ax_dict = plot_2d_spectrograph(hold, plotId, arms)
        saveFigure(fig, output_dir, "plot_2d_spectrograph")

    if plot_outlier:
        figs, ax_dicts = plot_outlier_summary(hold, holdAsDict, plotId, arms)
        for fig, arm in zip(figs, arms):
            saveFigure(fig, output_dir, f"plot_outlier_summary_arm_{arm}")

    if plot_sky_brightness:
        fig, ax_dict = plot_vs_sky_brightness(hold, plotId, arms)
        saveFigure(fig, output_dir, "plot_vs_sky_brightness")

    logging.info(f"All plots saved in {output_dir}")


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Run sky subtraction QA plots and save as PNG images.")
    parser.add_argument("--collection",
                        type=str, required=True, help="Data collection to process.")
    parser.add_argument("--visit", type=int, required=True, help="Observation visit number.")
    parser.add_argument("--spectrograph", type=int, required=True, help="Spectrograph number.")
    parser.add_argument("--arms",
                        nargs="+", default=['b', 'r', 'n'], help="List of spectral arms to process")
    parser.add_argument("--blockSize", type=int, help="Block size for sky model fitting")
    parser.add_argument("--rejIterations", type=int, help="Number of rejection iterations")
    parser.add_argument("--rejThreshold", type=float, help="Rejection threshold")
    parser.add_argument("--mask", nargs="+", help="Mask types")
    parser.add_argument("--oversample", type=int, help="Oversampling factor")
    parser.add_argument("--no_plot_1d",
                        action="store_false", dest="plot_1d", help="Disable 1D spectrograph plot")
    parser.add_argument("--no_plot_2d",
                        action="store_false", dest="plot_2d", help="Disable 2D spectrograph plot")
    parser.add_argument("--no_plot_outlier",
                        action="store_false", dest="plot_outlier", help="Disable outlier summary plot")
    parser.add_argument("--no_plot_sky_brightness",
                        action="store_false", dest="plot_sky_brightness", help="Disable sky brightness plot")

    args = parser.parse_args()

    # Run the function with parsed arguments
    evaluateSkySubtractionQA(
        collection=args.collection,
        visit=args.visit,
        spectrograph=args.spectrograph,
        arms=args.arms,
        plot_1d=args.plot_1d,
        plot_2d=args.plot_2d,
        plot_outlier=args.plot_outlier,
        plot_sky_brightness=args.plot_sky_brightness,
        blockSize=args.blockSize,
        rejIterations=args.rejIterations,
        rejThreshold=args.rejThreshold,
        mask=args.mask,
        oversample=args.oversample,
    )
