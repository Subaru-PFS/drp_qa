DRP QA
======

## Introduction

This repository contains command line tasks for quality assurance (QA) of the
Data Release Production (DRP) pipeline. The QA tasks are implemented as
`CmdLineTask` classes in the LSST Science Pipelines. The tasks are run on the
output of the DRP pipeline to assess the quality of the data products.

## Available Commands

Each of the following commands are available on the command line.

### DetectorMap QA

Measures the residuals in the spatial and wavelength directions between the
detector map and the arcline centroids. The residuals are measured for each
visit and for the combined data from multiple visits.

#### Command and Options

The command to run the `DetectorMapQA` task is:

```bash
detectorMapQA.py </PATH/TO/DATA/REPO> --rerun <RERUN_NAME> --id <ID_STR> 
```

Available config options are:

- `--combineVisits`: Combine the data from multiple visits into a single plot. Default is `True`.
- `--makeResidualPlots`: Make residual plots for each visit or combined visit, if `--combineVisits` is set. Default is
  `True`.
- `--useSigmaRange`: Use the sigma range to determine the range of the color scale in the residual plots. Default is
  `False`.
- `--xrange`: The range of the x-center (i.e. spatial) in the residual plots. Default is `0.1`.
- `--wrange`: The range of the y-center (i.e. wavelength) in the residual plots. Default is `0.1`.
- `--binWavelength`: The bin size in wavelength for the residual plots in nm. Default is `0.1`.

#### Outputs

Outputs of the `DetectorMapQA` task are:

- `dmQaResidualPlot` : 1D and 2D plots of the residual between the detectormap and the arclines for a given visit.
- `dmQaCombinedResidualPlot` : 1D and 2D plots of the residual between the detectormap and the arclines for the entire
  detector.
- `dmQaResidualStats` : Statistics of the residual analysis per visit.
- `dmQaDetectorStats` :  Statistics of the residual analysis per detector.

### Extraction QA

### Flux Calibration QA
