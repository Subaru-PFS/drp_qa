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
