DRP QA
======

## Introduction

This repository contains the pipeline and corresponding tasks for the quality
assurance (QA) of the Prime Focus Spectrograph (PFS) Data Release Production (DRP)
pipeline. The QA tasks are implementations of the `PipelineTask` class in the LSST
Science Pipelines. The tasks are run on the output of the DRP pipeline to assess
the quality of the data products.

## Available Pipelines

The following pipelines are available and can be used as the pipeline parameter
option (i.e. `-p`) to the `pipetask` command.

### `pipelines/detectorMapQa.yaml`

Measures the residuals in the spatial and wavelength directions between the
detector map and the arcline centroids. The residuals are measured for each
visit and for the combined data from multiple visits.

The pipeline contains the following tasks:

#### Tasks

##### dmResiduals

Determines the residuals between `lines` and `detectorMaps` for a given exposure.

###### Options

- `dmResiduals:useSigmaRange`: Use the sigma range for the color scale in the residual plots. Default is `False`.
- `dmResiduals:spatialRange`: The range of the x-center (i.e. spatial) in the residual plots. Default is `0.1`.
- `dmResiduals:wavelengthRange`: The range of the y-center (i.e. wavelength) in the residual plots. Default is `0.1`.
- `dmResiduals:binWavelength`: The bin size in wavelength for the residual plots in nm. Default is `0.1`.

###### Outputs

| DataSet Type        | Description                               | Dimensions                                                                                  |
|---------------------|-------------------------------------------|---------------------------------------------------------------------------------------------|
| `dmQaResidualStats` | `instrument, exposure, arm, spectrograph` | Summary statistics for the given detector and exposure.                                     | 
| `dmQaResidualPlot`  | `instrument, exposure, arm, spectrograph` | 1D and 2D plots of the residual between the detectormap and the arclines for a given visit. |

##### dmCombinedResiduals

Determines the aggregate statistics for all detectors across all given exposures.

###### Options

N/A

###### Outputs

| DataSet Type               | Description  | Dimensions                                                                                        |
|----------------------------|--------------|---------------------------------------------------------------------------------------------------|
| `dmQaCombinedResidualPlot` | `instrument` | 1D and 2D plots of the residual between the detectormap and the arclines for the entire detector. |
| `dmQaDetectorStats`        | `instrument` | Statistics of the residual analysis per detector.                                                 |

### `pipelines/fluxCalibration.yaml`

Measures the residuals in the flux calibration between the standard star
magnitudes and the instrumental magnitudes.

#### Command and Options

The command to run the `FluxCalibrationQA` task is:

```bash
fluxCalQA.py </PATH/TO/DATA/REPO> --rerun <RERUN_NAME> --id <ID_STR> 
```

Available config options are:

- `--filterSet`: The filter set to use for the flux calibration. Default is `ps1`.
- `--includeFakeJ`: Include the fake J band in the flux calibration. Default is `False`.
- `--diffFilter`: The filter to use for the differential flux calibration. Default is `g_ps1`.

#### Outputs

Outputs of the `FluxCalibrationQA` task are:

- `fluxCalMagDiffPlot` : Plot of the difference between the standard star magnitudes and the instrumental magnitudes.
- `fluxCalColorDiffPlot` : Plot of the difference between the instrumental magnitudes and a given filter.
- `fluxCalStats` : Statistics of the flux calibration analysis.
