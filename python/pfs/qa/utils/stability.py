from pathlib import Path
from typing import Optional

import lsst.afw.display as afwDisplay
import lsst.daf.persistence as dafPersist
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from dataclasses import dataclass, field
from matplotlib import colors
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pfs.drp.stella import ArcLineSet, DetectorMap, PfsArm, ReferenceLineStatus
from scipy.stats import iqr

afwDisplay.setDefaultBackend("matplotlib")

sb.set_style('whitegrid')
mpl.rcParams.update({
    'grid.alpha': 0.1,
    'image.cmap': 'coolwarm',
})


# Make a dataclass for the stability statistics.
@dataclass
class StabilityStatistics:
    visit: int
    spectrograph: int
    arm: str
    label: str

    rerunName: str
    repoDir: Path = Path('/work/drp')
    calibDir: Path = Path('/work/drp/CALIB')

    pfsArm: PfsArm = field(init=False)
    arcLines: ArcLineSet = field(init=False)
    detectorMap: DetectorMap = field(init=False)
    arcData: pd.DataFrame = field(init=False)

    butler: dafPersist.Butler = field(init=False)

    def __post_init__(self):
        self.rerun = self.repoDir / 'rerun' / self.rerunName
        self.butler = dafPersist.Butler(self.rerun.as_posix(), calibRoot=self.calibDir.as_posix())

        self.pfsArm = self.butler.get('pfsArm', self.dataId)
        self.arcLines = self.butler.get('arcLines', self.dataId)
        self.detectorMap = self.butler.get('detectorMap_used', self.dataId)

        self.arcData = self.getData()

    @property
    def dataId(self):
        return dict(visit=self.visit, arm=self.arm, spectrograph=self.spectrograph)

    @property
    def ccd(self):
        return self.arm + str(self.spectrograph)

    def getData(self):
        """Looks up the data in butler and returns a dataframe with the arcline data.

        The arcline data includes basic statistics, such as the median and sigma of the residuals.

        This method is called on init.
        """

        # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
        self.arcData = self.getArclineData(statusTypes=list(), dropNa=True, dropColumns=['xx', 'yy', 'xy'])
        self.arcData = self.addTraceLambdaToArclines()
        self.arcData = self.addResidualsToArclines()

        return self.arcData

    def getStatistics(self, hd5_fn=None, agg_stats=None):
        """Gets statistics for the residuals.

        Parameters
        ----------
        hd5_fn : `str`, optional
            Filename to write the statistics to. Default is None.
        agg_stats : `list` of `str`, optional
            List of statistics to calculate. Default is ``['count', 'mean', 'median', 'std', sigma]``
            where ``sigma`` is the interquartile range divided by 1.349.

        """

        def iqr_sigma(x):
            return iqr(x) / 1.349

        # Get aggregate stats.
        if agg_stats is None:
            agg_stats = ['count', 'mean', 'median', 'std', iqr_sigma]

        # For entire detector.
        ccd_stats = self.arcData.groupby(['status_name']).agg({'dx': agg_stats, 'dy': agg_stats})
        ccd_stats = ccd_stats.reset_index().melt(id_vars=['status_name'], var_name=['col', 'metric'])
        ccd_stats.insert(0, 'visit', self.visit)
        ccd_stats.insert(1, 'ccd', self.ccd)

        # Per fiber.
        fiber_stats = self.arcData.groupby(['fiberId', 'status_name']).agg({'dx': agg_stats, 'dy': agg_stats})
        fiber_stats = fiber_stats.reset_index().melt(id_vars=['fiberId', 'status_name'], var_name=['col', 'metric'])
        fiber_stats.insert(0, 'visit', self.visit)
        fiber_stats.insert(1, 'ccd', self.ccd)

        # # By wavelength (?).
        # wavelength_stats = self.arcData.groupby(['wavelength', 'status_name']).agg({'dx': agg_stats, 'dy': agg_stats})
        # wavelength_stats = wavelength_stats.query('')
        # wavelength_stats = wavelength_stats.reset_index().melt(id_vars=['wavelength', 'status_name'], var_name=['col', 'metric'])
        # wavelength_stats.insert(0, 'visit', self.visit)
        # wavelength_stats.insert(1, 'ccd', self.ccd)

        if hd5_fn is not None:
            ccd_stats.to_hdf(hd5_fn, key='ccd', format='table', append=True, index=False)
            fiber_stats.to_hdf(hd5_fn, key='fiber', format='table', append=True, index=False)
            # wavelength_stats.to_hdf(hd5_fn, key='wavelength', format='table', append=True, index=False)
        else:
            return ccd_stats, fiber_stats  # , wavelength_stats

    def plotResidualsFacetGrid(self, xColumn: str = 'wavelength', usePixels: bool = True, setLimits=True):
        """Plots residuals as a FacetGrid.

        Parameters
        ----------
        xColumn : `str`, optional
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
        id_cols = ['fiberId', 'wavelength', 'status_name', 'Trace']
        plot_cols.extend(id_cols)
        plot_data = self.arcData[plot_cols].melt(id_vars=id_cols, value_name='residual')

        fg = sb.FacetGrid(plot_data, row='variable', sharex=True, sharey=False, margin_titles=True)
        fg.figure.set_size_inches(12, 5)

        # TODO make the palette more consistent.
        categories = plot_data.status_name.dtype.categories
        palette = {n: c for n, c in zip(categories, sb.color_palette(palette='Set1', n_colors=len(categories)))}

        # Plot the data.
        fg.map_dataframe(sb.scatterplot, x=xColumn, y='residual',
                         hue='status_name', palette=palette,
                         style='Trace', markers={0: 'o', 1: '.'},
                         alpha=0.5
                         )

        fg.axes[0][0].axhline(0, ls='--', alpha=0.25, color='g')
        fg.axes[1][0].axhline(0, ls='--', alpha=0.25, color='g')
        if setLimits is True:
            fg.axes[0][0].set_ylim(-1.5, 1.5)

        fg.add_legend(shadow=True, fontsize='small')
        fg.figure.suptitle(f'Residuals by {xColumn}\n{self.dataId}\n{self.rerunName}', y=0.97, fontsize='small',
                           bbox=dict(facecolor='grey', edgecolor='k', alpha=0.45, pad=5.0))

        return fg

    def getArclineData(self,
                       dropNa: bool = False,
                       dropColumns: Optional[list] = None,
                       removeFlagged: bool = False,
                       oneHotStatus: bool = False,
                       removeTrace: bool = False,
                       statusTypes=None
                       ) -> pd.DataFrame:
        """Gets a copy of the arcline data, with some columns added.

        Parameters
        ----------
        dropNa : `bool`, optional
            Drop rows with NaN values? Default is True.
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

        if dropColumns is not None:
            arc_data = arc_data.drop(columns=dropColumns)

        if removeFlagged:
            arc_data = arc_data.query('flag == False')

        if len(statusTypes):
            arc_data = arc_data.query(' or '.join(f'status == {s}' for s in statusTypes))

        if dropNa:
            arc_data = arc_data.dropna()

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
            'REJECTED',
            'NOT_VISIBLE',
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

    def plotResidualsQuiver(self, title: str = '', arrowScale: float = 0.01, usePixels=True,
                            width: int = None, height: int = None) -> Figure:
        """
        Plot residuals as a quiver plot.

        Parameters
        ----------
        arc_data: `pandas.DataFrame`
            Arc line data.
        title: `str`
            Title for plot.
        arrowScale: `float`
            Scale for quiver arrows.
        usePixels: `bool`
            If wavelength should be plotted in pixels, default True.
        width: `int`
            Width of the detectormap.
        height: `int`
            Height of the detectormap.

        Returns
        -------
        fig: `matplotlib.figure.Figure`
        """
        arc_data = self.arcData

        fig, ax0 = plt.subplots(1, 1)

        wavelength_col = arc_data.dy if usePixels is True else arc_data.dy_nm

        C = np.hypot(arc_data.dx, wavelength_col)
        im = ax0.quiver(arc_data.tracePosX, arc_data.tracePosY, arc_data.dx, wavelength_col, C,
                        norm=colors.Normalize(vmin=0, vmax=1),
                        angles='xy', scale_units='xy', scale=arrowScale, units='width',
                        cmap='coolwarm')
        ax0.quiverkey(im, 0.1, 1., arrowScale, label=f'{arrowScale=}')

        divider = make_axes_locatable(ax0)
        cax = divider.append_axes('right', size='3%', pad=0.02)
        fig.colorbar(im, ax=ax0, cax=cax, orientation='vertical', shrink=0.6)
        ax0.set_aspect('equal')
        fig.suptitle(f"Residual quiver\n{title}")

        ax0.set_xlim(0, width)
        ax0.set_ylim(0, height)

        return fig

    def plotArcResiduals2D(self,
                           positionCol='dx', wavelengthCol='dy',
                           showWavelength=True,
                           hexBin=False, gridsize=100,
                           width: int = None, height: int = None) -> Figure:
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
            Use hexbin plot. Default is ``False``.
        gridsize: `int`
            Grid size for hexbin plot. Default is 100.
        width: `int`
            Width of the detectormap.
        height: `int`
            Height of the detectormap.

        Returns
        -------
        fig: `matplotlib.figure.Figure`
        """
        arc_data = self.arcData

        ncols = 2 if showWavelength else 1

        fig, axes = plt.subplots(1, ncols, sharex=True, sharey=True)

        def _make_subplot(ax, data, subtitle=''):
            vmin, vmax = -1., 1
            norm = colors.Normalize(vmin, vmax)

            if hexBin:
                im = ax.hexbin(arc_data.x, arc_data.y, data, norm=norm, gridsize=gridsize)
            else:
                im = ax.scatter(arc_data.x, arc_data.y, c=data, s=3, norm=norm)

            ax.set_aspect('equal')
            ax.set_title(f"{subtitle}")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.02)
            fig.colorbar(im, ax=ax, cax=cax, orientation='vertical', shrink=0.6)

            ax.set_xlim(0, width)
            ax.set_ylim(0, height)

        if showWavelength:
            _make_subplot(axes[0], arc_data[positionCol], subtitle=f'{positionCol} [pixel]')
            _make_subplot(axes[1], arc_data[wavelengthCol], subtitle=f'{wavelengthCol} [pixel]')
        else:
            _make_subplot(axes, arc_data[positionCol], subtitle='dx [pixel]')

        return fig
