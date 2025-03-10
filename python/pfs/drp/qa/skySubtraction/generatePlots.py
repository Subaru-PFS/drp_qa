#!/usr/bin/env python3
import argparse
import os

from pfs.drp.qa.skySubtraction.prepare import prepareDataset
from pfs.drp.qa.skySubtraction.summaryPlots import plot_1d_spectrograph, plot_2d_spectrograph
from pfs.drp.qa.skySubtraction.summaryPlots import plot_outlier_summary, plot_vs_sky_brightness
from pfs.drp.stella.fitFocalPlane import FitBlockedOversampledSplineConfig


def evaluateSkySubtractionQA(collection, visit, spectrograph, output_dir='.', arms=['b', 'r'],
                             plot_1d=True, plot_2d=True, plot_outlier=True, plot_sky_brightness=True,
                             blockSize=20, rejIterations=1, rejThreshold=4.0,
                             mask=["NO_DATA", "BAD_FLAT", "BAD_FIBERNORMS", "SUSPECT"]):
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
    blockSize : `int`, optional
        Block size for sky model fitting.
    rejIterations : `int`, optional
        Number of rejection iterations.
    rejThreshold : `float`, optional
        Rejection threshold.
    mask : `list` of `str`, optional
        Mask types for sky subtraction.

    Saves
    -----
    - PNG plots in a directory `plots/visit_{visit}_spec_{spectrograph}/`
    """

    # Configure sky model fitting
    fitSkyModelConfig = FitBlockedOversampledSplineConfig()
    fitSkyModelConfig.blockSize = blockSize
    fitSkyModelConfig.rejIterations = rejIterations
    fitSkyModelConfig.rejThreshold = rejThreshold
    fitSkyModelConfig.mask = mask

    # Define data identifier
    dataId = dict(spectrograph=spectrograph, visit=visit)

    # Prepare dataset
    hold, holdAsDict, plotId = prepareDataset(collection, dataId, arms, fitSkyModelConfig)

    # Create output directory
    output_dir = os.path.join(output_dir, f"plots/visit_{visit}_spec_{spectrograph}")
    os.makedirs(output_dir, exist_ok=True)

    # Function to save figures
    def save_figure(fig, function_name):
        """Save a matplotlib figure as a PNG file in the output directory."""
        filename = os.path.join(output_dir, f"{function_name}.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")

    # Generate and save plots
    if plot_1d:
        fig, ax_dict = plot_1d_spectrograph(holdAsDict, plotId, arms)
        save_figure(fig, "plot_1d_spectrograph")

    if plot_2d:
        fig, ax_dict = plot_2d_spectrograph(hold, plotId, arms)
        save_figure(fig, "plot_2d_spectrograph")

    if plot_outlier:
        figs, ax_dicts = plot_outlier_summary(hold, holdAsDict, plotId, arms)
        for fig, arm in zip(figs, arms):
            save_figure(fig, f"plot_outlier_summary_arm_{arm}")

    if plot_sky_brightness:
        fig, ax_dict = plot_vs_sky_brightness(hold, plotId, arms)
        save_figure(fig, "plot_vs_sky_brightness")

    print(f"All plots saved in {output_dir}")


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Run sky subtraction QA plots and save as PNG images.")
    parser.add_argument("--collection", type=str, required=True, help="Data collection to process.")
    parser.add_argument("--visit", type=int, required=True, help="Observation visit number.")
    parser.add_argument("--spectrograph", type=int, required=True, help="Spectrograph number.")
    parser.add_argument("--arms", nargs="+", default=['b', 'r'], help="List of spectral arms to process")
    parser.add_argument("--blockSize", type=int, default=20, help="Block size for sky model fitting")
    parser.add_argument("--rejIterations", type=int, default=1, help="Number of rejection iterations")
    parser.add_argument("--rejThreshold", type=float, default=4.0, help="Rejection threshold")
    parser.add_argument("--mask", nargs="+", default=["NO_DATA", "BAD_FLAT", "BAD_FIBERNORMS", "SUSPECT"],
                        help="Mask types")
    parser.add_argument("--no_1d", action="store_false", dest="plot_1d",
                        help="Disable 1D spectrograph plot")
    parser.add_argument("--no_2d", action="store_false", dest="plot_2d",
                        help="Disable 2D spectrograph plot")
    parser.add_argument("--no_outlier", action="store_false", dest="plot_outlier",
                        help="Disable outlier summary plot")
    parser.add_argument("--no_sky", action="store_false", dest="plot_sky_brightness",
                        help="Disable sky brightness plot")

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
    )
