from typing import Dict

import numpy as np
from pfs.drp.stella.fitReference import FilterCurve, TransmissionCurve


def get_filter_curves(filter_set="ps1", include_fake_j=False) -> Dict[str, TransmissionCurve]:
    """Get the filter curves for the given filter set.

    Parameters
    ----------
    filter_set : str
        Filter set to use.
    include_fake_j : bool
        Whether to include a fake narrow J filter. Defaults to False.

    Returns
    -------
    filter_curves : dict[str, TransmissionCurve]
        Filter curves.
    """
    filter_curves = dict()
    for filter_name in FilterCurve.filenames.keys():
        if not filter_name.endswith(filter_set.lower()):
            continue

        fc = FilterCurve(filter_name)
        fc.filter_name = filter_name
        filter_curves[filter_name] = fc

    if include_fake_j:
        # Add a fake narrow J filter that is a copy of the PS1 y filter.
        fakeJ_wl = np.arange(1082, 1202)

        # Fake J filter is 1 everywhere except for the edges.
        fakeJ_transmission = np.ones_like(fakeJ_wl)
        fakeJ_transmission[:1] = 0.0
        fakeJ_transmission[-2:] = 0.0

        fakeJ = TransmissionCurve(fakeJ_wl, fakeJ_transmission)
        fakeJ.filter_name = "fakeJ_ps1"
        filter_curves[fakeJ.filter_name] = fakeJ

    return filter_curves
