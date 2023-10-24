import logging
from pathlib import Path
from typing import Optional

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

from pfs.datamodel import TargetType
from pfs.drp.stella import ArcLineSet, DetectorMap, PfsArm, ReferenceLineStatus


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
    statusTypes: InitVar[list, None] = None

    category_palette: dict = None

    def __post_init__(self, loadData, statusTypes):
        self.rerun = self.repoDir / 'rerun' / self.rerunName

        if self.butler is None:
            logging.info('Creating a new butler')
            self.butler = dafPersist.Butler(self.rerun.as_posix(), calibRoot=self.calibDir.as_posix())

        self.arcLines = self.butler.get('arcLines', self.dataId)
        self.detectorMap = self.butler.get('detectorMap_used', self.dataId)
        self.pfsConfig = self.butler.get('pfsConfig', self.dataId)

        if loadData is True:
            self.arcData = self.getData(statusTypes=statusTypes)

    @property
    def dataId(self):
        return dict(visit=self.visit, arm=self.arm, spectrograph=self.spectrograph, label=self.label)

    @property
    def ccd(self):
        return self.arm + str(self.spectrograph)

    @property
    def uid(self):
        return f'v{self.visit}-{self.ccd}-{self.label}'

    def getData(self, statusTypes: Optional[list] = None, dropNaColumns: bool = True):
        """Looks up the data in butler and returns a dataframe with the arcline data.

        The arcline data includes basic statistics, such as the median and sigma of the residuals.

        This method is called on init.

        Parameters
        ----------
        dropNaColumns : `bool`, optional
            Drop columns where all values are NaN. Default is True.

        """

        # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
        self.arcData = self.getArclineData(statusTypes=statusTypes, dropNaColumns=dropNaColumns)
        self.arcData = self.addTraceLambdaToArclines()
        self.arcData = self.addResidualsToArclines()

        # TODO make the palette more consistent.
        status_categories = self.arcData.status_name.dtype.categories
        self.category_palette = {n: c for n, c in
                                 zip(status_categories,
                                     sb.color_palette(palette='Set1', n_colors=len(status_categories)))}

        return self.arcData

    def getStatistics(self, hd5_fn=None, agg_stats=None):
        """Gets statistics for the residuals.

        Parameters
        ----------
        hd5_fn : `str`, optional
            Filename to write the statistics to. Default is None.
        agg_stats : `list` of `str`, optional
            List of statistics to calculate. Default is ``['count', 'mean', 'median', 'std', iqr_std]``
            where ``iqr_std`` is the interquartile range divided by 1.349.

        """

        def iqr_std(x):
            return iqr(x) / 1.349

        # Get aggregate stats.
        if agg_stats is None:
            agg_stats = ['mean', 'median', 'std', 'sem', iqr_std]

        agg_columns = {
            'status_name': 'count',
            'dx': agg_stats,
            'dy': agg_stats,
            'centroidErr': agg_stats,
            'detectorMapErr': agg_stats
        }

        # For grouped stats for entire detector and per fiber.
        ccd_stats = self.arcData.groupby(['status_name']).agg(agg_columns)
        agg_columns.pop('status_name')
        fiber_stats = self.arcData.groupby(['fiberId', 'status_name']).agg(agg_columns)

        # Add metadata.
        for df in [ccd_stats, fiber_stats]:
            df.reset_index(inplace=True)
            df.insert(0, 'visit', self.visit)
            df.insert(1, 'ccd', self.ccd)
            df.insert(2, 'label', self.label)
            df.insert(3, 'rerun', self.rerunName)

            # Make single level column names.
            df.columns = [f'{c[0]}_{c[1]}' if c[1] > '' else c[0] for c in df.columns]

            # Make some adjustments for the string storage.
            df.status_name = df.status_name.astype(str)

        if hd5_fn is not None:
            itemsize_dict = dict(status_name=75, label=75, rerun=75)

            ccd_stats.to_hdf(hd5_fn, key=f'ccd', format='table', append=True, index=False,
                             min_itemsize=itemsize_dict)

            fiber_stats.to_hdf(hd5_fn, key=f'fibers', format='table', append=True, index=False,
                               min_itemsize=itemsize_dict)

        return ccd_stats, fiber_stats

    def getArclineData(self,
                       dropNaColumns: bool = False,
                       removeFlagged: bool = False,
                       oneHotStatus: bool = False,
                       removeTrace: bool = False,
                       statusTypes=None
                       ) -> pd.DataFrame:
        """Gets a copy of the arcline data, with some columns added.

        Parameters
        ----------
        dropNaColumns : `bool`, optional
            Drop columns where all values are NaN. Default is True.
        removeFlagged : `bool`, optional
            Remove rows with ``flag=True``? Default is True.
        oneHotStatus : `bool`, optional
            Add one-hot columns for the status? Default is False.
        removeTrace : `bool`, optional
            Remove rows with ``Trace==True``? Default is False.
        statusTypes : `list` of `pfs.drp.stella.ReferenceLineStatus`, optional
            Status types to include. Default is ``[DETECTORMAP_RESERVED, DETECTORMAP_USED]``.

        Returns
        -------
        arc_data : `pandas.DataFrame`
        """
        # Get the data from the ArcLineSet.
        arc_data = self.arcLines.data.copy()

        if statusTypes is None:
            statusTypes = [ReferenceLineStatus.DETECTORMAP_RESERVED,
                           ReferenceLineStatus.DETECTORMAP_USED]

        if removeFlagged:
            arc_data = arc_data.query('flag == False')

        if len(statusTypes):
            arc_data = arc_data.query(' or '.join(f'status == {s}' for s in statusTypes))

        if dropNaColumns:
            arc_data = arc_data.dropna(axis=1, how='all')

        # Drop rows without enough info.
        arc_data = arc_data.dropna(subset=['x', 'y'])

        arc_data = arc_data.copy()

        # Change some of the dtypes explicitly.
        arc_data.y = arc_data.y.astype(np.float64)

        # Replace inf with nans.
        arc_data = arc_data.replace([np.inf, -np.inf], np.nan)

        # Get status names.
        arc_data['status_name'] = arc_data.status.map(lambda x: str(ReferenceLineStatus(x)).split('.')[-1])
        arc_data['status_name'] = arc_data['status_name'].astype('category')

        # Clean up categories.
        ignore_lines = [
            # 'MERGED',
            # 'NOT_VISIBLE',
            # 'REJECTED',
        ]

        # Ignore bad line categories.
        for ignore in ignore_lines:
            arc_data = arc_data[~arc_data.status_name.str.contains(ignore)]

        arc_data.status_name = arc_data.status_name.cat.remove_unused_categories()

        # Get one hot for status.
        if oneHotStatus:
            arc_data = arc_data.merge(pd.get_dummies(arc_data.status_name), left_index=True, right_index=True)

        # Make a one-hot for the Trace.
        try:
            arc_data['Trace'] = arc_data.description.str.get_dummies()['Trace']
        except KeyError:
            arc_data['Trace'] = False

        if removeTrace is True:
            arc_data = arc_data.query(f'Trace == False').copy()

        # Add TargetType for each fiber.
        arc_data = arc_data.merge(pd.DataFrame({'fiberId': self.pfsConfig.fiberId, 'targetType': [TargetType(x).name for x in self.pfsConfig.targetType]}), left_on='fiberId', right_on='fiberId')
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

        arc_data['lam'] = self.detectorMap.findWavelength(arc_data.fiberId.to_numpy(), arc_data.y.to_numpy())
        arc_data['lamErr'] = arc_data.yErr * arc_data.lam / arc_data.y

        # Convert nm to pixels.
        dispersion = self.detectorMap.getDispersion(arc_data.fiberId.to_numpy(), arc_data.wavelength.to_numpy())
        arc_data['lam_pix'] = arc_data.lam / dispersion
        arc_data['lamErr_pix'] = arc_data.lamErr / dispersion
        arc_data['wavelengthDispersion'] = arc_data.wavelength / dispersion

        # Get the trace positions according to the detectormap.
        points = self.detectorMap.findPoint(arc_data.fiberId.to_numpy(), arc_data.wavelength.to_numpy())
        arc_data['tracePosX'] = points[:, 0]
        arc_data['tracePosY'] = points[:, 1]

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
        arc_data['dx'] = arc_data.tracePosX - arc_data.x
        arc_data['dy'] = arc_data.lam_pix - arc_data.wavelengthDispersion
        arc_data['dy_nm'] = arc_data.lam - arc_data.wavelength

        self.calculateResiduals('dx', fitYTo=fitYTo)
        self.calculateResiduals('dy', fitYTo=fitYTo)

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

        c0, c1, c2 = np.linalg.lstsq(X, Y, rcond=None)[0]

        fit = c0 + (c1 * a) + (c2 * b)
        arc_data[f'{targetCol}Fit'] = fit
        arc_data[f'{targetCol}ResidualLinear'] = Y - fit

        # Mean and median fits
        arc_data[f'{targetCol}ResidualMean'] = Y - Y.mean()
        arc_data[f'{targetCol}ResidualMedian'] = Y - Y.median()

        return arc_data

    def plotResiduals1D(self, by: str = 'wavelength', usePixels: bool = True, setLimits: bool = True):
        """Plots residuals as a FacetGrid.

        Parameters
        ----------
        by : `str`, optional
            Column to plot on the x-axis. Default is ``wavelength``, could also be ``fiberId``.
        usePixels : `bool`, optional
            If wavelength should be plotted in pixels, default True.
        setLimits : `bool`, optional
            If limits should be set on the x-axis. Default True.

        Returns
        -------
        fg : `seaborn.FacetGrid`
        """
        plot_cols = ['dx', 'dy' if usePixels else 'dy_nm']

        # Put the data in long format.
        id_cols = ['fiberId', 'wavelength', 'status_name', 'targetType']
        plot_cols.extend(id_cols)
        plot_data = self.arcData[plot_cols].melt(id_vars=id_cols, value_name='residual')

        fg = sb.FacetGrid(plot_data, row='variable', sharex=True, sharey=False, margin_titles=True)
        fg.figure.set_size_inches(12, 5)

        # Plot the data.
        fg.map_dataframe(sb.scatterplot, x=by, y='residual',
                         hue='targetType', style='status_name', s=5,
                         alpha=0.5
                         )

        fg.axes[0][0].axhline(0, ls='--', alpha=0.25, color='g')
        fg.axes[1][0].axhline(0, ls='--', alpha=0.25, color='g')
        if setLimits is True:
            fg.axes[0][0].set_ylim(-1.5, 1.5)

        fg.add_legend(shadow=True, fontsize='small')
        fg.figure.suptitle(f'Residuals by {by}\n{self.dataId}\n{self.rerunName}', y=0.97, fontsize='small',
                           bbox=dict(facecolor='grey', edgecolor='k', alpha=0.45, pad=5.0))

        return fg

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
        wavelength_col = arc_data.dy if usePixels is True else arc_data.dy_nm

        C = np.hypot(arc_data.dx, wavelength_col)
        Cnorm = (C - C.min()) / (C.max() - C.min())
        im = ax0.quiver(arc_data.tracePosX, arc_data.tracePosY, arc_data.dx, wavelength_col, C,
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
                        showWavelength=True,
                        hexBin=True, gridsize=250, plotKws: dict = None) -> Figure:
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

        arc_data = self.arcData

        width = self.detectorMap.getBBox().width
        height = self.detectorMap.getBBox().height

        # ncols = 2 if showWavelength else 1
        ncols = 1

        fig, axes = plt.subplots(1, ncols, sharex=True, sharey=True)

        def _make_subplot(ax, data, subtitle='', normalize=colors.SymLogNorm):
            norm = normalize(linthresh=0.01, vmin=plotKws.pop('vmin', None), vmax=plotKws.pop('vmax', None))

            if hexBin:
                im = ax.hexbin(arc_data.tracePosX, arc_data.tracePosY, data, norm=norm, gridsize=gridsize, **plotKws)
            else:
                im = ax.scatter(arc_data.tracePosX, arc_data.tracePosY, c=data, s=3, norm=norm, **plotKws)

            ax.set_aspect('equal')
            ax.set_title(f"{subtitle}")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)
            fig.colorbar(im, ax=ax, cax=cax, orientation='vertical', shrink=0.6, extend='both')

            ax.set_xlim(0, width)
            ax.set_ylim(0, height)

        if showWavelength:
            # _make_subplot(axes[0], arc_data[positionCol], subtitle=f'{positionCol} [pixel]')
            _make_subplot(axes, arc_data[wavelengthCol], subtitle=f'{wavelengthCol} [pixel]')
        else:
            _make_subplot(axes, arc_data[positionCol], subtitle='dx [pixel]')

        fig.suptitle(f'2D residuals {self.dataId}\n{self.rerunName}')

        return fig
