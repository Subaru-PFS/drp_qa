DRP QA
======

## Introduction

This repository contains the pipeline and corresponding tasks for the quality
assurance (QA) of the Prime Focus Spectrograph (PFS) Data Release Production (DRP)
pipeline. The QA tasks are implementations of the `PipelineTask` class in the LSST
Science Pipelines. The tasks are run on the output of the DRP pipeline to assess
the quality of the data products.

## QA Pipeline

The QA pipelines is located at `pipelines/drpQA.yaml`, which runs all of the QA
tasks.

> Note: individual tasks can be specified by using the `pipelines/drpQA.yaml#extractionQA`
> syntax.
>
See [documentation](https://pipelines.lsst.io/modules/lsst.pipe.base/creating-a-pipeline.html#command-line-options-for-running-pipelines)
> for details.

Also see the example notebook [`examples/QA Pipelines.ipynb`](examples/QA%20Pipelines.ipynb).

The pipeline contains the following tasks:

### Tasks

#### `dmResiduals`

Measures the residuals in the spatial and wavelength directions between the
detector map and the arcline centroids. The residuals are measured for each
visit.

##### Options

- `dmResiduals:useSigmaRange`: Use the sigma range for the color scale in the residual plots. Default is `False`.
- `dmResiduals:spatialRange`: The range of the x-center (i.e. spatial) in the residual plots. Default is `0.1`.
- `dmResiduals:wavelengthRange`: The range of the y-center (i.e. wavelength) in the residual plots. Default is `0.1`.
- `dmResiduals:binWavelength`: The bin size in wavelength for the residual plots in nm. Default is `0.1`.

##### Outputs

| DataSet Type        | Dimensions                             | Description                                                                                 |
|---------------------|----------------------------------------|---------------------------------------------------------------------------------------------|
| `dmQaResidualData`  | `instrument, visit, arm, spectrograph` | Residual data for the given detector and visit.                                             | 
| `dmQaResidualStats` | `instrument, visit, arm, spectrograph` | Summary statistics for the given detector and visit.                                        | 
| `dmQaResidualPlot`  | `instrument, visit, arm, spectrograph` | 1D and 2D plots of the residual between the detectormap and the arclines for a given visit. |

#### `dmCombinedResiduals`

Determines the aggregate statistics for all detectors across all given visits.

##### Options

N/A

##### Outputs

| DataSet Type               | Dimensions   | Description                                                                                       |
|----------------------------|--------------|---------------------------------------------------------------------------------------------------|
| `dmQaCombinedResidualPlot` | `instrument` | 1D and 2D plots of the residual between the detectormap and the arclines for the entire detector. |
| `dmQaDetectorStats`        | `instrument` | Statistics of the residual analysis per detector.                                                 |

#### `imageQualityQa`

Measures image quality from arc-line second-moment measurements for a given detector and visit.
Computes Gaussian-equivalent FWHM metrics and produces configurable QA plots as a multi-page PDF.

##### Options

- `imageQualityQa:showWhisker`: Show a whisker/quiver plot of FWHM and position angle. Default `True`.
- `imageQualityQa:showFWHM`: Show a 2D spatial hexbin map of FWHM. Default `False`.
- `imageQualityQa:showFWHMAgainstLambda`: Plot FWHM vs log(flux) (or S/N if `useSN` is `True`). Default `False`.
- `imageQualityQa:showFWHMHistogram`: Show a histogram of FWHM values. Default `False`.
- `imageQualityQa:showFluxHistogram`: Show a histogram of line fluxes. Default `False`.
- `imageQualityQa:useSN`: Use signal-to-noise rather than log10(flux) in the `showFWHMAgainstLambda` plot. Default `False`.
- `imageQualityQa:minFluxPercentile`: Minimum flux percentile for line selection in spatial plots. Default `10.0`.
- `imageQualityQa:vmin`: Minimum FWHM for color scale (pixels). Default `2.5`.
- `imageQualityQa:vmax`: Maximum FWHM for color scale (pixels). Default `3.5`.
- `imageQualityQa:maxFwhm`: Upper FWHM cutoff for line selection and histogram binning (pixels). Default `8.0`.
- `imageQualityQa:logScale`: Log y-axis for histogram plots. Default `True`.
- `imageQualityQa:gridsize`: Grid size for hexbin FWHM map; use `<=0` for a scatter plot. Default `100`.
- `imageQualityQa:stride`: Fiber stride for downsampling arc lines in spatial plots. Default `1`.

##### Outputs

| DataSet Type | Dimensions                             | Description                                                          |
|--------------|----------------------------------------|----------------------------------------------------------------------|
| `iqQaData`   | `instrument, visit, arm, spectrograph` | Per-line image quality measurements, including FWHM and position angle. |
| `iqQaPlot`   | `instrument, visit, arm, spectrograph` | Image quality QA plots (one page per enabled plot type).             |

#### `imageQualityCombinedQa`

Aggregates per-detector image quality data from `imageQualityQa` across all visit/arm/spectrograph
quanta and produces a single multi-panel figure laid out as an arm × spectrograph grid.

##### Options

- `imageQualityCombinedQa:showWhisker`: Show a whisker/quiver plot of FWHM and position angle. Default `True`.
- `imageQualityCombinedQa:showFWHM`: Show a 2D spatial hexbin map of FWHM. Default `False`.
- `imageQualityCombinedQa:showFWHMAgainstLambda`: Plot FWHM vs log(flux) (or S/N if `useSN` is `True`). Default `False`.
- `imageQualityCombinedQa:showFWHMHistogram`: Show a histogram of FWHM values. Default `False`.
- `imageQualityCombinedQa:showFluxHistogram`: Show a histogram of line fluxes. Default `False`.
- `imageQualityCombinedQa:useSN`: Use signal-to-noise rather than log10(flux) in the `showFWHMAgainstLambda` plot. Default `False`.
- `imageQualityCombinedQa:minFluxPercentile`: Minimum flux percentile for line selection in spatial plots. Default `10.0`.
- `imageQualityCombinedQa:vmin`: Minimum FWHM for color scale (pixels). Default `2.5`.
- `imageQualityCombinedQa:vmax`: Maximum FWHM for color scale (pixels). Default `3.5`.
- `imageQualityCombinedQa:maxFwhm`: Upper FWHM cutoff for line selection and histogram binning (pixels). Default `8.0`.
- `imageQualityCombinedQa:logScale`: Log y-axis for histogram plots. Default `True`.
- `imageQualityCombinedQa:gridsize`: Grid size for hexbin FWHM map; use `<=0` for a scatter plot. Default `100`.
- `imageQualityCombinedQa:stride`: Fiber stride for downsampling arc lines in spatial plots. Default `1`.

##### Outputs

| DataSet Type         | Dimensions   | Description                                                                                                        |
|----------------------|--------------|--------------------------------------------------------------------------------------------------------------------|
| `iqQaCombinedPlot`   | `instrument` | Combined image quality QA figure arranged as an arm × spectrograph grid (one page per enabled plot type).         |

#### `extractionQa`

Determines the quality of the fiber extraction.

##### Options

- `extractionQa:fixWidth`: Fix the widths during Gaussian fitting, default `False`.
- `extractionQa:rowNum`: Number of rows picked up for profile analysis, default `200`.
- `extractionQa:thresError`: Threshold of the fitting error, default `0.1`.
- `extractionQa:thresChi`: Threshold for chi standard deviation, default `1.5`.
- `extractionQa:fiberWidth`: Half width of a fiber region (pix), default `3`.
- `extractionQa:fitWidth`: Half width of a fitting region (pix), default `3`.
- `extractionQa:plotWidth`: Half width of plot (pix), default `15`.
- `extractionQa:plotFiberNum`: Maximum fiber number of detailed plots, default `20`.
- `extractionQa:figureDpi`: resolution of plot for residual, default `72`.

##### Outputs

| DataSet Type        | Dimensions                             | Description                                                                                                                        |
|---------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `extQaStats`        | `instrument, visit, arm, spectrograph` | Summary plots. Results of the residual analysis of extraction are plotted.                                                         |
| `extQaImage`        | `instrument, visit, arm, spectrograph` | Residual, and chi comparisons of the postISRCCD profile and fiberProfiles are plotted for some fibers with bad extraction quality. |
| `extQaImage_pickle` | `instrument, visit, arm, spectrograph` | Statistics of the residual analysis.                                                                                               |

#### `fiberNormsQa`

Plot the fiber normalization for the given detector and visit.

##### Options

- `fiberNormsQa:plotLower`: Lower bound for plot (standard deviations from median), default 2.5.
- `fiberNormsQa:plotUpper`: Upper bound for plot (standard deviations from median), default 2.5.

##### Outputs

| DataSet Type     | Dimensions               | Description                                   |
|------------------|--------------------------|-----------------------------------------------|
| `fiberNormsPlot` | `instrument, visit, arm` | Plot of the fiber normalizations for a visit. |

#### `fluxCalQa`

##### Options

- `fluxCalQa:filterSet`: Filter set to use, default `ps1`.
- `fluxCalQa:includeFakeJ`: Include the fake narrow J filter, default `True`.
- `fluxCalQa:fakeJoffset`: Offset from the ps1 bands for the fake narrow J, default `0.054`.
- `fluxCalQa:diffFilter`: Filter to use for the color magnitude difference, default `g_ps1`.
- `fluxCalQa:doAnonymize`: Mask sensitive information in the plot, default `True`.

##### Outputs

| DataSet Type         | Dimensions          | Description                                        |
|----------------------|---------------------|----------------------------------------------------|
| `fluxCalStats`       | `instrument, visit` | Statistics of the flux calibration analysis.       |
| `fluxCalMagDiffPlot` | `instrument, visit` | Plot of the flux calibration magnitude difference. |

#### `skySubtractionQa`

There are two tasks in the Sky Subtraction QA, the `skyArmSubtractionQa` and `skySubtractionQa`.
The `skyArmSubtractionQa` task is used to subtract the sky from the spectra of each arm
and the `skySubtractionQa` task is used to plot the results of the sky subtraction for
the entire visit.

##### Options

If config options are not passed, the default values come from `mergeArms_config.fitSkyModel`.

- `skyArmSubtractionQa:blockSize`: Block size for the sky subtraction, default `None`.
- `skyArmSubtractionQa:rejIterations`: Number of rejection iterations, default `None`.
- `skyArmSubtractionQa:rejThreshold`: Rejection threshold, default `None`.
- `skyArmSubtractionQa:oversample`: Oversampling factor, default `None`.
- `skyArmSubtractionQa:mask`: Mask types to use, default `None`.

##### Outputs

| DataSet Type           | Dimensions               | Description                                                                                 |
|------------------------|--------------------------|---------------------------------------------------------------------------------------------| 
| `skySubtractionQaPlot` | `instrument, visit, arm` | PDF of various plots related to sky subtraction <br/>built from all the arms for the visit. |
