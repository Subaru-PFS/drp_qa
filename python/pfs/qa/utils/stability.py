import logging
from pathlib import Path
from typing import Optional
from contextlib import suppress

import lsst.daf.persistence as dafPersist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from dataclasses import dataclass, field, InitVar
from matplotlib import colors
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from scipy.stats import iqr

from pfs.drp.stella.utils import addPfsCursor

from pfs.datamodel import TargetType
from pfs.drp.stella import ArcLineSet, DetectorMap, PfsArm, ReferenceLineStatus


def iqr_std(x):
    return iqr(x) / 1.349


# Make a dataclass for the stability statistics.
@dataclass
class DetectorMapStatistics:
    visit: int
    spectrograph: int
    arm: str
    label: str

    rerunName: str
    repoDir: Path = Path('/work/drp')
    calibDir: Path = Path('/work/drp/CALIB')

    arcLines: ArcLineSet = field(init=False)
    detectorMap: DetectorMap = field(init=False)
    arcData: pd.DataFrame = field(init=False)

    butler: dafPersist.Butler = None

    loadData: InitVar[bool] = True

    category_palette: dict = None

    def __post_init__(self, loadData):
        self.rerun = self.repoDir / 'rerun' / self.rerunName

        if self.butler is None:
            logging.info('Creating a new butler')
            self.butler = dafPersist.Butler(self.rerun.as_posix(), calibRoot=self.calibDir.as_posix())

        self.arcLines = self.butler.get('arcLines', self.dataId)
        self.detectorMap = self.butler.get('detectorMap_used', self.dataId)

        if loadData is True:
            self.arcData = self.getData()

    @property
    def dataId(self):
        return dict(visit=self.visit, arm=self.arm, spectrograph=self.spectrograph, label=self.label)

    @property
    def ccd(self):
        return self.arm + str(self.spectrograph)

    @property
    def uid(self):
        return f'v{self.visit}-{self.ccd}-{self.label}'

    def getData(self, dropNaColumns: bool = True):
        """Looks up the data in butler and returns a dataframe with the arcline data.

        The arcline data includes basic statistics, such as the median and sigma of the residuals.

        This method is called on init.

        Parameters
        ----------
        dropNaColumns : `bool`, optional
            Drop columns where all values are NaN. Default is True.

        """

        # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
        self.arcData = self.getArclineData(dropNaColumns=dropNaColumns)
        self.arcData = self.addTraceLambdaToArclines()
        self.arcData = self.addResidualsToArclines()
        
        # Add dataId info to dataframe.
        for col in ['arm', 'spectrograph', 'visit', 'rerun', 'label']:
            self.arcData[col] = getattr(self, col)

        # TODO make the palette more consistent.
        status_categories = self.arcData.status_name.dtype.categories
        self.category_palette = {n: c for n, c in
                                 zip(status_categories,
                                     sb.color_palette(palette='Set1', n_colors=len(status_categories)))}

        self.arcData.reset_index(drop=True, inplace=True)
        return self.arcData


    def getArclineData(self, dropNaColumns: bool = False, removeFlagged: bool = True, addTargetType: bool = False) -> pd.DataFrame:
        """Gets a copy of the arcline data, with some columns added.

        Parameters
        ----------
        dropNaColumns : `bool`, optional
            Drop columns where all values are NaN. Default is True.
        removeFlagged : `bool`, optional
            Remove rows with ``flag=True``? Default is False.

        Returns
        -------
        arc_data : `pandas.DataFrame`
        """
        # Get the data from the ArcLineSet.
        arc_data = self.arcLines.data.copy()

        if removeFlagged:
            arc_data = arc_data.query('flag == False')

        if dropNaColumns:
            arc_data = arc_data.dropna(axis=1, how='all')

        # Drop rows without enough info.
        arc_data = arc_data.dropna(subset=['x', 'y'])

        # Change some of the dtypes explicitly.
        arc_data.y = arc_data.y.astype(np.float64)

        # Replace inf with nans.
        arc_data = arc_data.replace([np.inf, -np.inf], np.nan)
        
        # Only use DETECTORMAP_USED (32) and DETECTORMAP_RESERVED (64)
        arc_data = arc_data.query('status in [32, 64]')
        arc_data = arc_data.copy()
        
        # Get status names.
        arc_data['status_name'] = arc_data.status.map(lambda x: ReferenceLineStatus(x).name)
        arc_data['status_name'] = arc_data['status_name'].astype('category')

        # Make a one-hot for the Trace.
        arc_data['isTrace'] = False
        with suppress():
            arc_data.loc[arc_data.query('description == "Trace"').index, 'isTrace'] = True            

        # Add TargetType for each fiber.
        if addTargetType:
            pfsConfig = self.butler.get('pfsConfig', self.dataId)
            arc_data = arc_data.merge(pd.DataFrame({'fiberId': pfsConfig.fiberId, 'targetType': [TargetType(x).name for x in pfsConfig.targetType]}), left_on='fiberId', right_on='fiberId')
            arc_data['targetType'] = arc_data.targetType.astype('category')

        return arc_data

    def addTraceLambdaToArclines(self) -> pd.DataFrame:
        """Adds detector map trace position and wavelength to arcline data.

        Returns
        -------
        arc_data : `pandas.DataFrame`
        """
        # Get the wavelength according to the detectormap for fiberId.
        arc_data = self.arcData
        fiberList = arc_data.fiberId.to_numpy()
        yList = arc_data.y.to_numpy()
        
        arc_data['lam'] = self.detectorMap.findWavelength(fiberList, yList)
        arc_data['lamErr'] = arc_data.yErr * arc_data.lam / arc_data.y

        # Convert nm to pixels.
        # dispersion = self.detectorMap.getDispersion(arc_data.fiberId.to_numpy(), arc_data.wavelength.to_numpy())
        dispersion = self.detectorMap.getDispersionAtCenter()
        arc_data['dispersion'] = dispersion

        # Get the trace positions according to the detectormap.
        arc_data['tracePos'] = self.detectorMap.getXCenter(fiberList, yList)

        return arc_data

    def addResidualsToArclines(self, fitYTo: str = 'y') -> pd.DataFrame:
        """Adds residuals to arcline data.

        This will calculate residuals for the X-center position and wavelength.

        Adds the following columns to the arcline data:

        - ``dx``: X-center position minus trace position (from DetectorMap `findPoint`).
        - ``dxResidualLinear``: Linear fit to ``dx``.
        - ``dxResidualMean``: Mean fit to ``dx``.
        - ``dxResidualMedian``: Median fit to ``dx``.
        - ``dy``: Y-center position minus trace position (from DetectorMap `findPoint`).
        - ``dyResidualLinear``: Linear fit to ``dy``.
        - ``dyResidualMean``: Mean fit to ``dy``.
        - ``dyResidualMedian``: Median fit to ``dy``.
        - ``dy_nm``: Wavelength minus trace wavelength.
        - ``dy_nmResidualLinear``: Linear fit to ``dy_nm``.
        - ``dy_nmResidualMean``: Mean fit to ``dy_nm``.
        - ``dy_nmResidualMedian``: Median fit to ``dy_nm``.
        - ``centroidErr``: Hypotenuse of ``xErr`` and ``yErr``.
        - ``detectorMapErr``: Hypotenuse of ``dx`` and ``dy``.


        Parameters
        ----------
        fitYTo : `str`, optional
            Column to fit Y to. Default is ``y``, could also be ``wavelength``.

        Returns
        -------
        arc_data : `pandas.DataFrame`
        """
        arc_data = self.arcData

        # Get `observed - expected` for position and wavelength.
        arc_data['dx'] = arc_data.tracePos - arc_data.x
        arc_data['dy_nm'] = arc_data.lam - arc_data.wavelength
        
        # Set the dy columns to NA (instead of 0) for Trace.
        arc_data.dy_nm = arc_data.apply(lambda row: row.dy_nm if row.isTrace == False else np.NaN, axis=1)
        
        # Do the dispersion correction to get pixels.
        arc_data['dy'] = arc_data.dy_nm / arc_data.dispersion

        # self.calculateResiduals('dx', fitYTo=fitYTo)
        # self.calculateResiduals('dy', fitYTo=fitYTo)

        arc_data['centroidErr'] = np.hypot(arc_data.xErr, arc_data.yErr)
        arc_data['detectorMapErr'] = np.hypot(arc_data.dx, arc_data.dy)

        return arc_data

    def calculateResiduals(self, targetCol: str, fitXTo: str = 'fiberId', fitYTo: str = 'y') -> pd.DataFrame:
        """Calculates residuals.

        This will calculate residuals for the X-center position and wavelength.

        Adds the following columns to the arcline data:

        - ``dx``: X-center position minus trace position (from DetectorMap `findPoint`).
        - ``dxResidualLinear``: Linear fit to ``dx``.
        - ``dxResidualMean``: Mean fit to ``dx``.
        - ``dxResidualMedian``: Median fit to ``dx``.

        Parameters
        ----------
        targetCol : `str`
            Column to calculate residuals for.
        fitXTo : `str`, optional
            Column to fit X to. Default is ``fiberId``, could also be ``x``.
        fitYTo : `str`, optional
            Column to fit Y to. Default is ``wavelength``, could also be ``y``.

        Returns
        -------
        arc_data : `pandas.DataFrame`
        """
        arc_data = self.arcData

        # Linear fit
        a = arc_data[fitXTo]
        b = arc_data[fitYTo]

        X = np.vstack([np.ones_like(a), a, b]).T
        Y = arc_data[targetCol]

        #c0, c1, c2 = np.linalg.lstsq(X, Y, rcond=None)[0]

        #fit = c0 + (c1 * a) + (c2 * b)
        #arc_data[f'{targetCol}Fit'] = fit
        #arc_data[f'{targetCol}ResidualLinear'] = Y - fit

        # Mean and median fits.
        fiberGroup = arc_data.groupby('fiberId')
        arc_data[f'{targetCol}ResidualMean'] = Y - fiberGroup[targetCol].transform('mean')
        arc_data[f'{targetCol}ResidualMedian'] = Y - fiberGroup[targetCol].transform('median')

        return arc_data


    def plotResiduals2DQuiver(self, arrowScale: float = 0.01, usePixels: bool = True,
                              plotKws: dict = None) -> Figure:
        """
        Plot residuals as a quiver plot.

        Parameters
        ----------
        arc_data: `pandas.DataFrame`
            Arc line data.
        arrowScale: `float`
            Scale for quiver arrows.
        usePixels: `bool`
            If wavelength should be plotted in pixels, default True.
        plotKws: `dict`
            Arguments passed to plotting function.

        Returns
        -------
        fig: `matplotlib.figure.Figure`
        """
        plotKws = plotKws or dict()
        plotKws.setdefault('cmap', 'magma_r')

        fig, ax0 = plt.subplots(1, 1)

        arc_data = self.arcData
        wavelength_col = arc_data.dyResidualMedian if usePixels is True else arc_data.dy_nm

        C = np.hypot(arc_data.dxResidualMedian, wavelength_col)
        Cnorm = (C - C.min()) / (C.max() - C.min())
        im = ax0.quiver(arc_data.tracePos, arc_data.y, arc_data.dxResidualMedian, wavelength_col, C,
                        norm=colors.Normalize(),
                        angles='xy', scale_units='xy', scale=arrowScale, units='xy',
                        # alpha=Cnorm,
                        **plotKws
                        )

        if arrowScale is not None:
            ax0.quiverkey(im, 0.1, 1., arrowScale, label=f'{arrowScale=}')

        divider = make_axes_locatable(ax0)
        cax = divider.append_axes('right', size='3%', pad=0.03)
        fig.colorbar(im, ax=ax0, cax=cax, orientation='vertical', shrink=0.6)
        ax0.set_aspect('equal')

        ax0.set_xlim(0, self.detectorMap.bbox.width)
        ax0.set_ylim(0, self.detectorMap.bbox.height)

        fig.suptitle(f'Residuals quiver {self.dataId}\n{self.rerunName}')

        return fig

    
    def plotResiduals2D(self,
                        positionCol='dx', wavelengthCol='dy',
                        showWavelength=False,
                        hexBin=True, gridsize=250, 
                        plotKws: dict = None,
                        title: str = None,
                        addCursor: bool = False
                       ) -> Figure:
        """ Plot residuals as a 2D histogram.

        Parameters
        ----------
        positionCol: `str`
            The column to plot for the position residuals. Default is 'dx'.
        wavelengthCol: `str`
            The column to plot for the wavelength residuals. Default is 'dy'.
        showWavelength: `bool`
            Show the y-axis. Default is ``True``.
        hexBin: `bool`
            Use hexbin plot. Default is ``True``.
        gridsize: `int`
            Grid size for hexbin plot. Default is 250.
        plotKws: `dict`
            Arguments passed to plotting function.

        Returns
        -------
        fig: `matplotlib.figure.Figure`
        """
        plotKws = plotKws or dict()

        # 2D residuals are of the reserved data.
        arc_data = self.arcData.query('status_name.str.contains("RESERVED")')
        
        # Don't use trace for wavelength.
        if showWavelength:
            arc_data = arc_data.query('isTrace == False')
                    
        width = self.detectorMap.getBBox().width
        height = self.detectorMap.getBBox().height

        # ncols = 2 if showWavelength else 1
        ncols = 1
        
        if addCursor:
            pfs_format_coord = addPfsCursor(None, self.detectorMap)

        fig, axes = plt.subplots(1, ncols, sharex=True, sharey=True)

        def _make_subplot(ax, data, subtitle='', normalize=colors.Normalize):
            norm = normalize(vmin=plotKws.pop('vmin', None), vmax=plotKws.pop('vmax', None))

            if hexBin:
                im = ax.hexbin(arc_data.tracePos, arc_data.y, data, norm=norm, gridsize=gridsize, **plotKws)
            else:
                im = ax.scatter(arc_data.tracePos, arc_data.y, c=data, s=1, norm=norm, **plotKws)

            stats_string = f'median={data.median():.03e} sigma={iqr_std(data):.03e}'
                
            ax.set_title(f"{stats_string} \n 2D residuals {subtitle}")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)
            fig.colorbar(im, ax=ax, cax=cax, orientation='vertical', shrink=0.6, extend='both', label=subtitle)

            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect('equal')
            
            if addCursor:
                ax.format_coord = pfs_format_coord

        if showWavelength:
            # _make_subplot(axes[0], arc_data[positionCol], subtitle=f'{positionCol} [pixel]')
            ax_title = _make_subplot(axes, arc_data[wavelengthCol], subtitle=f'{wavelengthCol} {"(pixel)" if wavelengthCol == "dy" else ""}')
        else:
            ax_title = _make_subplot(axes, arc_data[positionCol], subtitle=f'{positionCol} (pixel)')

        suptitle = f'{self.dataId}\n{self.rerunName}'
        if title and title > '':
            suptitle += f"\n{title}"
            
        fig.suptitle(suptitle, y=0.975)

        return fig
