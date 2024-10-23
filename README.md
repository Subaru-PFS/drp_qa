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
> See [documentation](https://pipelines.lsst.io/modules/lsst.pipe.base/creating-a-pipeline.html#command-line-options-for-running-pipelines)
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

- `fixWidth`: Fix the widths during Gaussian fitting.
- `rowNum`: Number of rows picked up for profile analysis.
- `thresError`: Threshold of the fitting error.
- `thresChi`: Threshold for chi standard deviation.
- `fiberWidth`: Half width of a fiber region (pix).
- `fitWidth`: Half width  of a fitting region (pix).
- `plotWidth`: Half width  of plot (pix).
- `plotFiberNum`: Maximum fiber number of detailed plots.
- `figureDpi`: resolution of plot for residual.

##### Outputs

| DataSet Type        | Dimensions                                | Description                                                                                                                        |
|---------------------|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `extQaStats`        | `instrument, exposure, arm, spectrograph` | Summary plots. Results of the residual analysis of extraction are plotted.                                                         |
| `extQaImage`        | `instrument, exposure, arm, spectrograph` | Residual, and chi comparisons of the postISRCCD profile and fiberProfiles are plotted for some fibers with bad extraction quality. |
| `extQaImage_pickle` | `instrument, exposure, arm, spectrograph` | Statistics of the residual analysis.                                                                                               |

#### `fluxCalQa`

> NOTE: This is not fully updated for `gen3`. In particular the `dimensions` in the output table will change.

Determines the quality of the flux calibration.

##### Options

- `fluxCalQa:filterSet`: The filter set to use for the flux calibration. Default is `ps1`.
- `fluxCalQa:includeFakeJ`: Include the fake J band in the flux calibration. Default is `False`.
- `fluxCalQa:diffFilter`: The filter to use for the differential flux calibration. Default is `g_ps1`.

##### Outputs

| DataSet Type           | Dimensions   | Description                                                                                  |
|------------------------|--------------|----------------------------------------------------------------------------------------------|
| `fluxCalMagDiffPlot`   | `instrument` | Plot of the difference between the standard star magnitudes and the instrumental magnitudes. |
| `fluxCalColorDiffPlot` | `instrument` | Plot of the difference between the instrumental magnitudes and a given filter.               |
| `fluxCalStats`         | `instrument` | Statistics of the flux calibration analysis.                                                 |
