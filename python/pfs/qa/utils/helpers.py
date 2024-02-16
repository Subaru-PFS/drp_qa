import pandas as pd
import numpy as np

from pathlib import Path
from typing import Iterable, Any, Dict, Optional

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


def iqr_sigma(x):
    return iqr(x) / 1.349


def getObjects():
    butler = dafPersist.Butler(self.rerun.as_posix(), calibRoot=self.calibDir.as_posix())
    arcLines = self.butler.get('arcLines', self.dataId)
    detectorMap = self.butler.get('detectorMap_used', self.dataId)


def getArclineData(als, dropNaColumns: bool = False, removeFlagged: bool = True) -> pd.DataFrame:
    """Gets a copy of the arcLineSet data, with some columns added.

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
    arc_data = als.data.copy()

    if removeFlagged:
        arc_data = arc_data.query('flag == False')

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
         'NOT_VISIBLE',
         'REJECTED',
         'PROTECTED',
         'MERGED',
         'LAM_FOCUS',
         'BLEND',
         'BROAD',
    ]

    # Ignore bad line categories.
    for ignore in ignore_lines:
        arc_data = arc_data[~arc_data.status_name.str.contains(ignore)]

    # Make a one-hot for the Trace.
    try:
        arc_data['isTrace'] = arc_data.description.str.get_dummies()['Trace'].astype(bool)
    except KeyError:
        arc_data['isTrace'] = False
        
    # Make one-hot columns for status_names
    status_dummies = arc_data.status_name.str.get_dummies()
    arc_data['isUsed'] = status_dummies.get('DETECTORMAP_USED', np.zeros(len(status_dummies))).astype(bool)
    arc_data['isReserved'] = status_dummies.get('DETECTORMAP_RESERVED', np.zeros(len(status_dummies))).astype(bool)
        
    # Only show reserved/used?
    #arc_data = arc_data.query('isUsed == True or isReserved == True').copy()

    arc_data.status_name = arc_data.status_name.cat.remove_unused_categories()

    return arc_data


def getTargetType(arc_data, pfsConfig):
    # Add TargetType for each fiber.
    arc_data = arc_data.merge(pd.DataFrame({
        'fiberId': pfsConfig.fiberId, 
        'targetType': [TargetType(x).name for x in pfsConfig.targetType]
    }), left_on='fiberId', right_on='fiberId')
    arc_data['targetType'] = arc_data.targetType.astype('category')

    return arc_data


def plotResiduals1D(arcLines: ArcLineSet, 
                    detectorMap: DetectorMap, 
                    arcData: pd.DataFrame,
                    # statistics: Dict[str, Any], 
                    showAllRange: bool = False,
                    xrange: float = 0.2,
                    wrange: float = 0.03,
                    pointSize: float = 0.2,
                    quivLength: float = 0.2
                   ) -> plt.Figure:
    """Plot the residuals as a function of wavelength and fiberId.

    Parameters:
        arcLines: The arc lines.
        detectorMap: The detector map.
        statistics: The statistics.

    Returns:
        The figure.
    """
    fmin, fmax = np.amin(arcLines.fiberId), np.amax(arcLines.fiberId)
    dmapUsed = (arcLines.status & ReferenceLineStatus.DETECTORMAP_USED) != 0
    dmapReserved = (arcLines.status & ReferenceLineStatus.DETECTORMAP_RESERVED) != 0

    measured = (
            np.logical_not(np.isnan(arcLines.flux))
            & np.logical_not(np.isnan(arcLines.x))
            & np.logical_not(np.isnan(arcLines.y))
            & np.logical_not(np.isnan(arcLines.xErr))
            & np.logical_not(np.isnan(arcLines.yErr))
            & np.logical_not(np.isnan(arcLines.fluxErr))
    )

    flist = []
    for f in range(fmin, fmax + 1):
        notNan_f = (arcLines.fiberId == f) & measured
        if np.sum(notNan_f) > 0:
            flist.append(f)

    arcLinesMeasured = arcLines[measured]
    residualX = arcLinesMeasured.x - detectorMap.getXCenter(arcLinesMeasured.fiberId,
                                                            arcLinesMeasured.y.astype(np.float64))
    residualW = arcLinesMeasured.wavelength - detectorMap.findWavelength(
        fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y.astype(np.float64)
    )
    minw = np.amin(
        detectorMap.findWavelength(fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y.astype(np.float64)))
    maxw = np.amax(
        detectorMap.findWavelength(fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y.astype(np.float64)))
    bufw = (maxw - minw) * 0.02

    dmUsedMeasured = dmapUsed[measured]
    dmReservedMeasured = dmapReserved[measured]
    if showAllRange:
        residualXMax = max(np.amax(residualX[dmUsedMeasured]), np.amax(residualX[dmReservedMeasured]))
        residualXMin = min(np.amin(residualX[dmUsedMeasured]), np.amin(residualX[dmReservedMeasured]))
        residualWMax = max(np.amax(residualW[dmUsedMeasured]), np.amax(residualW[dmReservedMeasured]))
        residualWMin = min(np.amin(residualW[dmUsedMeasured]), np.amin(residualW[dmReservedMeasured]))
        yxmax = xrange * 3 if residualXMax < xrange * 3 else residualXMax * 1.05
        yxmin = -xrange * 3 if residualXMin > -xrange * 3 else residualXMin * 1.05
        ywmax = wrange * 3 if residualWMax < wrange * 3 else residualWMax * 1.05
        ywmin = -wrange * 3 if residualWMin > -wrange * 3 else residualWMin * 1.05
    else:
        yxmax = xrange * 3
        yxmin = -xrange * 3
        ywmax = wrange * 3
        ywmin = -wrange * 3
        largeX = residualX > yxmax
        smallX = residualX < yxmin
        largeW = residualW > ywmax
        smallW = residualW < ywmin

    # Set up the figure and the axes
    fig1 = plt.figure()
    ax1 = [
        plt.axes([0.08, 0.08, 0.37, 0.36]),
        plt.axes([0.46, 0.08, 0.07, 0.36]),
        plt.axes([0.08, 0.54, 0.37, 0.36]),
        plt.axes([0.46, 0.54, 0.07, 0.36]),
        plt.axes([0.58, 0.08, 0.37, 0.36]),
        plt.axes([0.58, 0.54, 0.37, 0.36]),
    ]
    bl_ax = ax1[0]
    bl_hist_ax = ax1[1]
    tl_ax = ax1[2]
    tl_hist_ax = ax1[3]
    br_ax = ax1[4]
    tr_ax = ax1[5]

    # X center residual of 'used' data.
    bl_ax.scatter(
        arcLinesMeasured.wavelength[dmUsedMeasured],
        residualX[dmUsedMeasured],
        s=pointSize,
        c="b",
        label="DETECTORMAP_USED\n(median:{:.2e}, sigma:{:.2e})".format(
            np.median(residualX[dmUsedMeasured]), iqr(residualX[dmUsedMeasured]) / 1.349
        ),
    )
    # Show full range on X center plot if requested.
    if not showAllRange:
        if np.sum(largeX) + np.sum(smallX) > 0:
            bl_ax.quiver(arcLinesMeasured.wavelength[dmUsedMeasured & largeX],
                         np.zeros(np.sum(
                             dmUsedMeasured & largeX)) + yxmax - xrange * quivLength, 0,
                         xrange * quivLength,
                         label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                             yxmax, np.sum(dmUsedMeasured & largeX) / np.sum(dmUsedMeasured) * 100
                         ),
                         color="b",
                         angles="xy",
                         scale_units="xy",
                         scale=2,
                         )
            bl_ax.quiver(
                arcLinesMeasured.wavelength[dmUsedMeasured & smallX],
                np.zeros(np.sum(dmUsedMeasured & smallX)) + yxmin + xrange * quivLength, 0,
                -xrange * quivLength,
                color="b",
                angles="xy",
                scale_units="xy",
                scale=2,
            )

    # X center residual of 'reserved' data.
    bl_ax.scatter(
        arcLinesMeasured.wavelength[dmReservedMeasured],
        residualX[dmReservedMeasured],
        s=pointSize,
        c="r",
        label="DETECTORMAP_RESERVED\n(median:{:.2e}, sigma:{:.2e})".format(
            np.median(residualX[dmReservedMeasured]), iqr(residualX[dmReservedMeasured]) / 1.349
        ),
    )
    # Show full range on X center plot if requested.
    if not showAllRange:
        if np.sum(largeX) + np.sum(smallX) > 0:
            bl_ax.quiver(arcLinesMeasured.wavelength[dmReservedMeasured & largeX],
                         np.zeros(np.sum(
                             dmReservedMeasured & largeX)) + yxmax - xrange * quivLength,
                         0,
                         xrange * quivLength,
                         label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                             yxmax,
                             (np.sum(dmReservedMeasured & largeX) + np.sum(dmReservedMeasured & smallX))
                             / np.sum(dmReservedMeasured)
                             * 100,
                         ),
                         color="r",
                         angles="xy",
                         scale_units="xy",
                         scale=2,
                         )
            bl_ax.quiver(
                arcLinesMeasured.wavelength[dmReservedMeasured & smallX],
                np.zeros(np.sum(dmReservedMeasured & smallX)) + yxmin + xrange * quivLength,
                0,
                -xrange * quivLength,
                color="r",
                angles="xy",
                scale_units="xy",
                scale=2,
            )

    # X center residual histogram of 'used'.
    bl_hist_ax.hist(
        residualX[dmUsedMeasured],
        color="b",
        range=(-xrange * 3, xrange * 3),
        bins=35,
        orientation="horizontal",
    )
    # X center residual histogram 'reserved'.
    bl_hist_ax.hist(
        residualX[dmReservedMeasured],
        color="r",
        range=(-xrange * 3, xrange * 3),
        bins=35,
        orientation="horizontal",
    )

    # Wavelength residual of 'used' data.
    tl_ax.scatter(
        arcLinesMeasured.wavelength[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
        residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
        s=pointSize,
        c="b",
        label="DETECTORMAP_USED\n(median:{:.2e}, sigma:{:.2e})".format(
            np.median(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")]),
            iqr(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349,
        ),
    )
    if not showAllRange:
        if np.sum(largeW) + np.sum(smallW) > 0:
            tl_ax.quiver(
                arcLinesMeasured.wavelength[dmUsedMeasured & largeW],
                np.zeros(np.sum(dmUsedMeasured & largeW)) + ywmax - wrange * quivLength, 0,
                wrange * quivLength,
                label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                    ywmax,
                    (np.sum(dmUsedMeasured & largeW) + np.sum(dmUsedMeasured & smallW))
                    / np.sum(dmUsedMeasured)
                    * 100,
                ),
                color="b",
                angles="xy",
                scale_units="xy",
                scale=2,
            )
            tl_ax.quiver(
                arcLinesMeasured.wavelength[dmUsedMeasured & smallW],
                np.zeros(np.sum(dmUsedMeasured & smallW)) + ywmin + wrange * quivLength, 0,
                wrange * quivLength,
                color="b",
                angles="xy",
                scale_units="xy",
                scale=2,
            )

    # Wavelength residual of 'reserved' data.
    tl_ax.scatter(
        arcLinesMeasured.wavelength[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
        residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
        s=pointSize,
        c="r",
        label="DETECTORMAP_RESERVED\n(median:{:.2e}, sigma:{:.2e})".format(
            np.median(residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")]),
            iqr(residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349,
        ),
    )
    if not showAllRange:
        if np.sum(largeW) + np.sum(smallW) > 0:
            tl_ax.quiver(
                arcLinesMeasured.wavelength[dmReservedMeasured & largeW],
                np.zeros(np.sum(dmReservedMeasured & largeW)) + ywmax - wrange * quivLength,
                0,
                wrange * quivLength,
                label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                    ywmax,
                    (np.sum(dmReservedMeasured & largeW) + np.sum(dmReservedMeasured & smallW))
                    / np.sum(dmReservedMeasured)
                    * 100,
                ),
                color="r",
                angles="xy",
                scale_units="xy",
                scale=2,
            )
            tl_ax.quiver(
                arcLinesMeasured.wavelength[dmReservedMeasured & smallW],
                np.zeros(np.sum(dmReservedMeasured & smallW)) + ywmin + wrange * quivLength,
                0,
                -wrange * quivLength,
                color="r",
                angles="xy",
                scale_units="xy",
                scale=2,
            )

    # Wavelength residual histogram of 'used'.
    tl_hist_ax.hist(
        residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
        color="b",
        range=(-wrange * 3, wrange * 3),
        bins=35,
        orientation="horizontal",
    )
    # Wavelength residual histogram of 'reserved'.
    tl_hist_ax.hist(
        residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
        color="r",
        range=(-wrange * 3, wrange * 3),
        bins=35,
        orientation="horizontal",
    )
    
    colors = {
        'DETECTORMAP_USED': (0, 0, 1, 1),
        'DETECTORMAP_USED_ERROR': (0, 0, 1, 0.25),
        'DETECTORMAP_RESERVED': (1, 0, 0, 1),  
        'DETECTORMAP_RESERVED_ERROR': (1, 0, 0, 0.25),  
    }

    # X center residual fiber errors.    
    plot_data = arcData.query('status_name.str.contains("RESERVED")')[['fiberId', 'dx', 'status_name']]
    plot_data = plot_data.groupby('fiberId').dx.agg(['median', iqr_sigma]).reset_index()    
    label = f'USED\n'
    label += f'median={plot_data["median"].median():>13.03e}\n'
    label += f'sigma  ={plot_data["iqr_sigma"].median():>13.03e}'
    br_ax.errorbar(
        plot_data.fiberId,
        plot_data['median'],
        plot_data['iqr_sigma'],
        ls='',
        marker='.',
        mec='k',
        label=label
    )
    # br_ax.legend(bbox_to_anchor=(1.3, 1), fontsize='small', shadow=True)
    
    
    # Wavelength residual fiber errors.
    plot_data = arcData.query('isTrace == False and status_name.str.contains("RESERVED")')[['fiberId', 'dy_nm', 'status_name']]
    plot_data = plot_data.groupby('fiberId').dy_nm.agg(['median', iqr_sigma]).reset_index()    
    label = f'USED\n'
    label += f'median={plot_data["median"].median():>13.03e}\n'
    label += f'sigma  ={plot_data["iqr_sigma"].median():>13.03e}'
    tr_ax.errorbar(
        plot_data.fiberId,
        plot_data['median'],
        plot_data['iqr_sigma'],
        ls='',
        marker='.',
        mec='k',
        label=label
    )
    # tr_ax.legend(bbox_to_anchor=(1.3, 1), fontsize='small', shadow=True)    
    
    bl_ax.legend(fontsize='small', shadow=True)
    bl_ax.set_ylabel("X residual (pix)")
    bl_ax.set_xlabel("Wavelength (nm)")
    bl_ax.set_xlim(minw - bufw, maxw + bufw)
    bl_ax.set_ylim(yxmin, yxmax)
    bl_ax.set_title("X center residual (unit=pix)")

    bl_hist_ax.set_ylim(yxmin, yxmax)
    bl_hist_ax.set_yticklabels([])

    tl_ax.legend(fontsize='small', shadow=True)
    tl_ax.set_ylabel("Wavelength residual (nm)")
    tl_ax.set_xlabel("Wavelength (nm)")
    tl_ax.set_xlim(minw - bufw, maxw + bufw)
    tl_ax.set_ylim(ywmin, ywmax)
    tl_ax.set_title("Wavelength residual (unit=nm)")

    tl_hist_ax.set_ylim(ywmin, ywmax)
    tl_hist_ax.set_yticklabels([])

    br_ax.set_xlabel("fiberId")
    br_ax.set_ylim(yxmin, yxmax)
    br_ax.set_title("X center residual of each fiber\n(point=median, errbar=1sigma scatter, unit=pix)")
    tr_ax.set_xlabel("fiberId")
    # tr_ax.set_ylim(ywmin, ywmax)
    tr_ax.set_title("Wavelength residual of each fiber\n(point=median, errbar=1sigma scatter, unit=nm)")

    return fig1