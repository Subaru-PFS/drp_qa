# Guider plots

What each plot in `pfs.drp.qa.guiders.plotting` shows, where its data come from, and what to look for.
Each has a sample drawn from the real AG data of the tests (engineering visits of Run 30, in
`tests/guiders/data`), with numbered callouts explained beside it.

Offsets are a star's measured center minus a reference position, in microns; positions are in hardware
coordinates (`pfs.drp.qa.guiders.coordinates`). 1 AG pixel is 13 µm, or 0.14 arcsec. The plots take
AG data from `queries.readAgcData`, or the results of `analysis`; see the README's
"Guider tools" section for an example.

This file and the images are made by `makeGuiderPlotDocs.py`, from the descriptions in it; edit those
and rerun it, rather than editing this file.

- [Guide error of each AG exposure](#showagcerrorsforvisits)
- [Each camera's guide errors, relative to the others](#showagcerrorsforvisitsbycamera)
- [Each camera's guide errors, y against x](#showagcerrorsforvisitsbycamera-plotxy)
- [The guide stars on the PFI](#showguidererrors)
- [The guide stars on the PFI, coloured by other quantities](#showguidererrorsbyparams)
- [The guide corrections sent to the telescope](#showtelescopeerrors)
- [The drift of the guide stars](#plotdriftrate)
- [Each camera's guide errors on the PFI](#plotguideerrors)
- [The guide stars' positions: the guider's and pfs_utils's](#plotpfsutilscomparison)
- [Focus from the AG cameras](#plotfocus)
- [Focus from the AG cameras, in time](#plotfocus-byexposure)
- [Each camera's focus relative to the others](#plotfocusbyag)

<a id="showagcerrorsforvisits"></a>
## Guide error of each AG exposure

```python
plotting.showAgcErrorsForVisits(agcData)
```

![showAgcErrorsForVisits](showAgcErrorsForVisits.png)

The mean offset of the guide stars from where the guider expected them, center minus nominal, in each AG exposure: its length r, and its x and y in hardware coordinates (µm). This is what the guider tries to keep at 0.

**Data.** readAgcData: agc_match's agc_center_[xy]_mm and agc_nominal_[xy]_mm of the valid matches (agc_match.flags GOOD_MATCH); agc_exposure.taken_at; shutter_open from the sps_exposure times.

**Reading it.**

1. Each visit has its own colour; its legend is on the top panel.
2. Large dots: AG exposures taken while the spectrograph shutters were open. These are the ones that matter for the science.
3. A jump after the shutters close: in a raster scan the telescope moves to the next position, and the guider takes about 40 s to catch up.
4. 148292, the start of the second scan, has no SpS exposure (shutter_open 2). Its first errors are hundreds of microns, as the guider starts on a new pointing.
5. x and y show the direction of the error, in hardware coordinates.

**Look for.**

- r within a few microns (1 AG pixel is 13 µm) while the shutters are open.
- Spikes, or errors that persist through a visit: the guider isn't correcting.
- A slow trend in x or y: a drift the guider follows late.

**Options.** byTime (HST on the x axis), yLimit_um, reference (e.g. boresight), pfsVisitIds and agcExposureIds. ics_pfsPlotActor draws it on its own three axes (axes=).

*Sample:* Raster scan, visits 148284-148292.

<a id="showagcerrorsforvisitsbycamera"></a>
## Each camera's guide errors, relative to the others

```python
plotting.showAgcErrorsForVisitsByCamera(agcData)
```

![showAgcErrorsForVisitsByCamera](showAgcErrorsForVisitsByCamera.png)

Each camera's mean offset in each AG exposure (center minus nominal), less the camera's median, as the cameras' positions aren't known well, and less the exposure's mean over the cameras, which is the telescope's pointing error. What is left is how the cameras move relative to each other.

**Data.** readAgcData: valid matches without bad detection flags, taken while the shutters weren't closed. With fitPfiModel, offsets are from analysis.fitGlobalModel's model instead of the nominal positions.

**Reading it.**

1. y points towards the zenith (rotateToZenith), x towards the Opt side.
2. Each point is one camera in one AG exposure, less the camera's median and the exposure's mean over the cameras: what is left is how the cameras move relative to each other.
3. Outliers: a camera with one badly measured star, or a camera that moved.

**Look for.**

- Scatter of a few microns about 0.
- Trends with altitude or rotator angle (plotBy): flexure.
- Opposite cameras with opposite signs: a rotation or scale left in (try fitPfiModel).
- One camera apart from the others: that camera, or its stars.

**Options.** plotBy (agc_exposure_id, altitude, insrot), colorBy (camera, visit, altitude, insrot), plotPerCamera, showAltInsrot (mean offsets binned by altitude and rotator angle), plotDzDfocus (against the AG actor's focus error), drawVisitBoundaries, rotateToZenith.

*Sample:* Raster scan, visits 148284-148292.

<a id="showagcerrorsforvisitsbycamera-plotxy"></a>
## Each camera's guide errors, y against x

```python
plotting.showAgcErrorsForVisitsByCamera(agcData, plotXY=True, plotPerCamera=True, plotXYStride=None, showCovariance=True)
```

![showAgcErrorsForVisitsByCamera-plotXY](showAgcErrorsForVisitsByCamera-plotXY.png)

The same offsets as above, y against x: each AG exposure, or with plotXYStride=None each visit.

**Data.** As above.

**Reading it.**

1. One panel per camera; each point is the camera's mean offset in one visit.
2. The red ellipse holds 1 sigma of the points (clipped second moments). A long ellipse means the camera moves along one direction, e.g. flexure towards the zenith.

**Look for.**

- Points clustered on 0.
- Long ellipses: motion along one direction, such as flexure towards the zenith.
- Clusters away from 0 for some visits: the cameras moved between visits.

**Options.** connectDxDy joins the points in order; nVisitMin drops visits with few AG exposures.

*Sample:* Raster scan, visits 148284-148292; one point per visit.

<a id="showguidererrors"></a>
## The guide stars on the PFI

```python
fit = analysis.fitGuiderModel(agcData, analysis.GuiderFitConfig(...))
plotting.showGuiderErrors(fit, plotting.GuiderPlotConfig(...))
```

![showGuiderErrors](showGuiderErrors.png)

Each AG camera at its position on the PFI (mm), and each guide star at its offset from where the guider's models put it (µm, so magnified 1000 times) about the camera. With the models off, as in ics_pfsPlotActor, that is the offset from the guider's nominal position; with modelBoresightOffset and modelCCDOffset, what is left after an offset, rotation and scale of each exposure and of each camera.

**Data.** analysis.fitGuiderModel on readAgcData. The stars plotted are those the fit selected: valid matches without bad detection flags, shutters open, in AG exposures whose guide error passes maxGuideError_um.

**Reading it.**

1. +: the mean position of the camera's stars (mm). Each star is drawn at its offset from where the guider expected it (µm) from there.
2. A cloud off its +: the camera's stars are systematically away from where the guider expected them; here AG3 and AG5 by about 80 µm. Without the models these offsets mix the boresight's offset, rotation and scale with each camera's own.
3. No AG1 stars: none of its matches in 148258 was valid.
4. The cartoon: where the cameras are, and the sense of the rotator.
5. Colour: the AG exposure, so the time.

**Look for.**

- Clouds centred on their +, a few tens of microns across.
- Clouds off their + in a pattern round the ring: tangential is a rotation, radial a scale.
- Colour changing across a cloud: the stars drift during the visit.
- A missing camera: none of its matches is valid.

**Options.** GuiderPlotConfig: showGuideStarsAsArrows, showAverageGuideStarPos and Path (each camera's mean per exposure), rotateToAG1Down (minus the rotator angle), showGuideStarPositions, guideStarFrac. colorbars= updates the colorbar when redrawing (ics_pfsPlotActor).

*Sample:* All-sky exposure 148258 with ics_pfsPlotActor's settings: no models, guide error under 100 µm, 30% of the guide stars.

<a id="showguidererrorsbyparams"></a>
## The guide stars on the PFI, coloured by other quantities

```python
plotting.showGuiderErrorsByParams(fit, ["altitude", "azimuth", "insrot", "agc_exposure_id"])
```

![showGuiderErrorsByParams](showGuiderErrorsByParams.png)

Each camera's mean offset in each AG exposure, drawn as showGuiderErrors draws them, once per quantity and coloured by it.

**Data.** GuiderFit.guideErrorByCamera from analysis.fitGuiderModel, and the quantities from its agcData.

**Reading it.**

1. Each point is one camera's mean offset in one AG exposure, coloured by the quantity of the panel.
2. A colour gradient across a cloud: the offsets follow that quantity.

**Look for.**

- Colour gradients along a cloud: offsets that follow altitude, rotator angle, time...

*Sample:* Raster scan, visits 148284-148292, with the boresight and camera models.

<a id="showtelescopeerrors"></a>
## The guide corrections sent to the telescope

```python
plotting.showTelescopeErrors(agcData)
```

![showTelescopeErrors](showTelescopeErrors.png)

The corrections the AG actor sent the telescope after each AG exposure with the shutters open: altitude and azimuth (arcsec) and rotator angle.

**Data.** readAgcData: agc_guide_offset's guide_delta_el, guide_delta_az and guide_delta_insrot (arcsec).

**Reading it.**

1. How often the AG actor sent each altitude and azimuth correction.
2. The same corrections, coloured by AG exposure: their order in time.
3. The rotator correction, as the motion it makes at the AG cameras (24 cm from the axis).
4. The rotator correction by azimuth and altitude.

**Look for.**

- Corrections centred on 0, of a few tenths of an arcsec.
- Corrections that keep one sign, or trend in time: tracking or pointing model errors.
- Large rotator corrections: a rotator or position angle error.

**Options.** showTheta: the rotator correction in arcsec, rather than microns at the AG cameras.

*Sample:* All-sky exposure 148258.

<a id="plotdriftrate"></a>
## The drift of the guide stars

```python
plotting.plotDriftRate(analysis.fitDriftRate(agcData))
```

![plotDriftRate](plotDriftRate.png)

The guide stars' offsets against time, split into radial and tangential components, with the fitted drift rates.

**Data.** analysis.fitDriftRate on readAgcData: valid matches with the shutters open, center minus nominal by default, averaged per camera and AG exposure, less each camera's mean.

**Reading it.**

1. Each point: one camera's mean offset in one AG exposure, less the camera's mean.
2. The fitted line; its slope is the rate.
3. Radial: away from the boresight. Tangential: anticlockwise round it.

**Look for.**

- Rates within a fraction of a micron per minute.
- A radial drift with one sign on every camera: a scale change (focus, temperature).
- A tangential drift: a rotation.

**Options.** byTime=False plots against agc_exposure_id; fitDriftRate(radialTangential=False) gives x and y.

*Sample:* All-sky exposure 148258: -0.004 µm/min radial, 0.22 µm/min tangential.

<a id="plotguideerrors"></a>
## Each camera's guide errors on the PFI

```python
guideErrors = analysis.estimateGuideErrors(agcData)
plotting.plotGuideErrors(guideErrors)
```

![plotGuideErrors](plotGuideErrors.png)

Each camera's mean offset in each AG exposure (or visit) from a reference position, drawn about the camera's position, and their mean over the cameras about the boresight. drp_stella's estimateGuideErrors(plot=True).

**Data.** analysis.estimateGuideErrors on readAgcData: valid matches, center minus a reference: center0 (each star's median position over the data), nominal, boresight...

**Reading it.**

1. Each camera's mean offset (µm) in each AG exposure, about the camera's median position (mm, red +). Here a tight cluster, a few microns across.
2. The mean over the cameras, about the boresight.
3. No AG1 points: none of its matches in 148258 was valid. In a raster scan, each camera's points trace the scan instead.

**Look for.**

- Each camera's points in a tight cluster.
- The same pattern on every camera: the telescope moved (in a raster scan, for instance).
- Patterns that turn round the ring: a rotation.

**Options.** colorBy (agc_exposure_id, pfs_visit_id, time), drawTrack, rotateToAG1Down, expand; showClosedShutter adds the closed-shutter AG exposures, as small dots, given estimateGuideErrors(includeClosedShutter=True).

*Sample:* All-sky exposure 148258, from each star's median position (center0).

<a id="plotpfsutilscomparison"></a>
## The guide stars' positions: the guider's and pfs_utils's

```python
comparison = analysis.comparePfsUtilsPositions(queries.readAGCStars(opdb, designId, visit), agcData)
plotting.plotPfsUtilsComparison(comparison)
```

![plotPfsUtilsComparison](plotPfsUtilsComparison.png)

Where the guider expected each guide star (agc_nominal), and where pfs_utils puts it with the AG actor's model and inputs, about each camera.

**Data.** readAGCStars (the design's guide stars) and readAgcData (the AG actor's field center, position angle, ADC and M2 positions, and detector half); medians over the visit's first 10 AG exposures.

**Reading it.**

1. Each camera's stars, the cameras drawn 10 times closer to the boresight. The guider's positions (crosses) lie on pfs_utils's (circles).
2. d(theta): the median difference of the stars' angles about the boresight.

**Look for.**

- The two on top of each other: they agree to well under a micron.
- d(theta) near 0.
- Any disagreement: the AG actor and pfs_utils no longer use the same model.

**Options.** alignCenterPosition (in comparePfsUtilsPositions) removes the mean offset; plotUsingScatter.

*Sample:* All-sky exposure 148258.

<a id="plotfocus"></a>
## Focus from the AG cameras

```python
plotting.plotFocus(agcData, showMedian=True)
```

![plotFocus](plotFocus.png)

The glass has been removed from one half of each AG detector, so its two halves focus at different M2_OFF3, and the difference of the stars' sizes on the two halves gives the focus error. Three rows: the AG actor's focus error; the same from the stars' sizes, measured here; and each star's FWHM.

**Data.** readAgcData: agc_guide_offset.guide_delta_z1-6 (the AG actor's), agc_data's second moments and detection flags (RIGHT marks the half), tel_status.m2_off3. Isolated GAIA stars, valid matches. analysis.estimateFocusErrors uses the AG actor's calibration.

**Reading it.**

1. The AG actor's focus error of each camera (guide_delta_z1-6).
2. The focus error from the stars' sizes on the two halves of each detector. It crosses 0 at best focus, here M2_OFF3 = 3.28 mm.
3. Right axis: the same, as a change of M2_OFF3.
4. Each star's FWHM: red on the left halves, green on the right, with their medians. Each half is sharpest on its own side of best focus.
5. Far from focus the focus error saturates, at about 450 µm. AG6's line is straight: in this trimmed sample it has stars on both halves at only three M2_OFF3.

**Look for.**

- Focus errors crossing 0 at best focus; the AG actor's and ours agreeing.
- Cameras disagreeing about best focus: the focal plane is tilted.
- Each half's FWHM smallest on its own side of best focus.

**Options.** plotBy (focus, agc_exposure_id, altitude, insrot), colorBy, plotPerCamera, averageByFocusPosition, showCameraId, showPfiFocusPosition. Click a panel to set M2_OFF3: the top panel then shows the focus error expected about it (ShowFocusFit).

*Sample:* Focus sweep, visits 148266 and 148270-148277, M2_OFF3 from 2.725 to 3.55 mm.

<a id="plotfocus-byexposure"></a>
## Focus from the AG cameras, in time

```python
plotting.plotFocus(agcData, plotBy="agc_exposure_id", showOpdbFocus=False, showFWHM=False, showFocusSets=True)
```

![plotFocus-byExposure](plotFocus-byExposure.png)

The AG actor's focus error of each camera against AG exposure.

**Data.** As above.

**Reading it.**

1. Shading: each run of AG exposures at one M2_OFF3 (showFocusSets).
2. Grey lines: the last AG exposure of each visit. The cursor readout names the visit.
3. The focus error steps with each move of M2_OFF3.

**Look for.**

- A step at each change of M2_OFF3: about 600-700 µm of focus error per mm near focus in Run 30.
- Drifts within a run at one M2_OFF3: the focus changing (temperature, altitude).

*Sample:* Focus sweep, as above; ics_pfsPlotActor's FocusPlot.

<a id="plotfocusbyag"></a>
## Each camera's focus relative to the others

```python
plotting.plotFocusByAG(agcData)
```

![plotFocusByAG](plotFocusByAG.png)

Each camera's focus error in each visit, less the mean of the other cameras but AG1 (as drp_stella did) and the overall mean, negated to match Kawanomoto-san's plots.

**Data.** analysis.estimateFocusErrors per camera, from the stars' sizes; the median of each visit.

**Reading it.**

1. One line per visit: each camera's focus relative to the others.
2. Black stars, and the horizontal lines: each camera's mean over the visits.

**Look for.**

- The same pattern in every visit: a fixed tilt or piston of the cameras.
- A pattern that changes between visits: the focal plane moving.

**Options.** byCamera=False plots each camera against the visit; byExposureId uses each AG exposure.

*Sample:* Focus sweep, as above.
