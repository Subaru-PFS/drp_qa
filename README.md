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

#### `extractionQa`

Determines the quality of the fiber extraction.

##### Options

- `extractionQa:fiberWidth`: Half width of a fiber region (pix), default `3`.
- `extractionQa:plotMinChiMed`: Minimum median Chi to plot, default `-1.5`.
- `extractionQa:plotMaxChiMed`: Maximum median Chi to plot, default `1.5`.
- `extractionQa:plotMinChiStd`: Minimum standard deviation of Chi to plot, default `0.0`.
- `extractionQa:plotMaxChiStd`: Maximum standard deviation of Chi to plot, default `3.5`.
- `extractionQa:plotMinChiAtPeak`: Minimum Chi at peak to plot, default `-1.5`.
- `extractionQa:plotMaxChiAtPeak`: Maximum Chi at peak to plot, default `1.5`.
- `extractionQa:plotMinResFrac`: Minimum residual fraction, default `-5.0`.
- `extractionQa:plotMaxResFrac`: Maximum residual fraction, default `5.0`.
- `extractionQa:plotHistRangeScale`: The scale factor for the Chi histogram range, default `1.5`.
- `extractionQa:plotHistNbin`: The number of bins for the Chi histogram, default `100`.
- `extractionQa:targetType`: Target type for which to calculate statistics, default `["^ENGINEERING"]`.
- `extractionQa:figureDpi`: Resolution of plot for residual, default `72`.

##### Outputs

| DataSet Type        | Dimensions                             | Description                                                                                                                                                 |
|---------------------|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `extQaStats`        | `instrument, visit, arm, spectrograph` | Results of the residual analysis of extraction against fiberId are plotted (Chi_median, Chi_stddev, Chi_atProfilePeak, residual_fraction).                   |
| `extQaImage`        | `instrument, visit, arm, spectrograph` | Summary plot for 2D residual, chi, etc. (p.1),  2D image of chi (p.2), zoom-in of the chi image (p.3), and the histogram of residual fraction and chi (p.4) |
| `extQaImage_pickle` | `instrument, visit, arm, spectrograph` | Statistics of the residual analysis.                                                                                                                        |

#### `extractionQaCombined`

##### Options

- `extractionQaCombined:plotMinChiMed`: Minimum median Chi to plot, default `-1.5`.
- `extractionQaCombined:plotMaxChiMed`: Maximum median Chi to plot, default `1.5`.
- `extractionQaCombined:plotMinChiStd`: Minimum standard deviation of Chi to plot, default `0.0`.
- `extractionQaCombined:plotMaxChiStd`: Maximum standard deviation of Chi to plot, default `3.5`.
- `extractionQaCombined:plotMinChiAtPeak`: Minimum Chi at peak to plot, default `-1.5`.
- `extractionQaCombined:plotMaxChiAtPeak`: Maximum Chi at peak to plot, default `1.5`.
- `extractionQaCombined:plotMinResFrac`: Minimum residual fraction, default `-5.0`.
- `extractionQaCombined:plotMaxResFrac`: Maximum residual fraction, default `5.0`.
- `extractionQaCombined:targetType`: Target type for which to calculate statistics, default `["^ENGINEERING"]`.
- `extractionQaCombined:figureDpi`: Resolution of plot for residual, default `72`.
- `extractionQaCombined:footnoteSize`: Fontsize of the footnote, default `9`.

##### Outputs

| DataSet Type         | Dimensions                             | Description                                                                                                                        |
|----------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `extQaStatsCombined` | `instrument, visit, arm` | Results of the residual analysis of extraction plotted against fiberId for all spectrographs (left) and PFI positions (right).                   |

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
