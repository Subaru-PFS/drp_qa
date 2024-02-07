import pandas as pd
import numpy as np

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
