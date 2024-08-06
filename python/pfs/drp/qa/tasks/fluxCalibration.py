from typing import Dict

import lsstDebug
import numpy as np
from lsst.pex.config import Field
from lsst.pipe.base import Struct, Task
from pfs.datamodel import PfsConfig, PfsSingle, TargetType
from pfs.drp.qa.tasks.fluxCalibration import get_filter_curves, get_flux_info
from pfs.drp.qa.utils.plotting import plot_flux_cal_mag_diff


class FluxCalibrationConfig(Config):
    """Configuration for FluxCalibrationTask"""

    filterSet = Field(dtype=str, default="ps1", doc="Filter set to use, e.g. 'ps1'")
    includeFakeJ = Field(dtype=bool, default=False, doc="Include the fake narrow J filter")
    diffFilter = Field(dtype=str, default="g_ps1", doc="Filter to use for the color magnitude difference")


class FluxCalibrationTask(Task):
    """Task for QA of detectorMap."""

    ConfigClass = FluxCalibrationConfig
    _DefaultName = "fluxCalibration"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, pfsConfig: PfsConfig, pfsSingles: Dict[str, PfsSingle]) -> Struct:
        """QA plots for flux calibration.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration.
        pfsSingles : dict of `pfs.datamodel.PfsSingle`
            Flux-calibrated, single epoch spectra.

        Returns
        -------
        outputs : `Struct`
            QA outputs.
        """
        self.log.info(f"Flux Calibration QA for {pfsConfig}")

        # Get the filter curves.
        filter_curves = get_filter_curves(
            filter_set=self.config.filterSet, include_fake_j=self.config.includeFakeJ
        )
        self.log.info(f"Filter curves: {list(filter_curves.keys())}")

        pfsConfigFluxStd = pfsConfig.select(targetType=TargetType.FLUXSTD)

        # Get the flux information.
        if self.config.includeFakeJ:
            # Add a fake narrow J filter that is a copy of the PS1 y filter to the
            # psfFlux and filterNames
            pfsConfigFluxStd.psfFlux = [np.append(a, a[-1] - 0.054) for a in pfsConfigFluxStd.psfFlux]
            pfsConfigFluxStd.filterNames = [fn + ["fakeJ_ps1"] for fn in pfsConfigFluxStd.filterNames]

        diff_filter = self.config.diffFilter

        flux_info = get_flux_info(pfsConfigFluxStd, pfsSingles, filter_curves, diff_filter=diff_filter)

        # Make plots.
        self.log.info("Making magnitude difference plot")
        title = f"Magnitude Difference for v{pfsConfig.visit} designId={hex(pfsConfig.pfsDesignId)}"
        mag_diff_plot = plot_flux_cal_mag_diff(flux_info, filter_curves, pfsConfig, title=title)

        # Make a new set of filters with fake names, which makes plotting easy.
        fcs = {f"{diff_filter}-{k}": fc for k, fc in filter_curves.items()}
        title = f"Color Magnitude Difference for v{pfsConfig.visit} designId={hex(pfsConfig.pfsDesignId)}"
        color_diff_plot = plot_flux_cal_mag_diff(
            flux_info, fcs, pfsConfig, title=title, column_prefix="single-single"
        )

        return Struct(
            fluxCalStats=flux_info, fluxCalMagDiffPlot=mag_diff_plot, fluxCalColorDiffPlot=color_diff_plot
        )


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


def get_flux_info(
    pfs_config: PfsConfig,
    pfs_singles: Dict[str, PfsSingle],
    filter_curves: Dict[str, TransmissionCurve],
    diff_filter: str = "g_ps1",
) -> pd.DataFrame:
    """Get the flux information for the given pfsConfig and pfsSingles.

    Parameters
    ----------
    pfs_config : `pfs.datamodel.PfsConfig`
        Top-end configuration.
    pfs_singles : dict of `pfs.datamodel.PfsSingle`
        Flux-calibrated, single epoch spectra, with the fiberId as the key.
    filter_curves : dict[str, TransmissionCurve]
        Filter curves, with the filter name as the key.
    diff_filter : str
        Filter to use for the magnitude difference. Defaults to 'g_ps1'.

    Returns
    -------
    flux_info : `pandas.DataFrame`
        Flux information.
    """
    design_flux_df = get_design_flux_table(pfs_config)

    comparison_flux_df = get_comparison_flux_table(filter_curves, pfs_singles)

    # Merge comparison flux with design.
    design_flux_df = design_flux_df.merge(comparison_flux_df, left_index=True, right_index=True)

    # Clean up inf.
    design_flux_df = design_flux_df.replace([np.inf, -np.inf], np.nan)

    get_magnitude_diffs(design_flux_df, filter_curves, diff_filter=diff_filter)

    return design_flux_df


def get_magnitude_diffs(
    design_flux_df: pd.DataFrame, filter_curves: Dict[str, TransmissionCurve], diff_filter: str = "g_ps1"
) -> None:
    """Get the magnitude differences for the given design fluxes.

    Parameters
    ----------
    design_flux_df : `pandas.DataFrame`
        Design flux table.
    filter_curves : dict[str, TransmissionCurve]
        Filter curves, with the filter name as the key.
    diff_filter : str
        Filter to use for the magnitude difference. Defaults to 'g_ps1'.
    """
    # Get the magnitude differences for each filter.
    for fc_name in filter_curves.keys():
        design_mag = design_flux_df[f"design.ABMag.{fc_name}"]
        single_mag = design_flux_df[f"single.ABMag.{fc_name}"]
        design_flux_df[f"single-design.ABMag.{fc_name}"] = single_mag - design_mag

    # Get the color difference for given filter.
    if diff_filter is not None:
        for filter_name, fc in filter_curves.items():
            design_flux_df[f"single-single.ABMag.{diff_filter}-{filter_name}"] = (
                design_flux_df[f"single.ABMag.{diff_filter}"] - design_flux_df[f"single.ABMag.{filter_name}"]
            )


def get_comparison_flux_table(
    filter_curves: Dict[str, TransmissionCurve], pfs_singles: Dict[str, PfsSingle]
) -> pd.DataFrame:
    """Get the comparison filter fluxes for the given pfsSingles.

    Parameters
    ----------
    filter_curves : dict[str, TransmissionCurve]
        Filter curves, with the filter name as the key.
    pfs_singles : dict of `pfs.datamodel.PfsSingle`
        Flux-calibrated, single epoch spectra, with the fiberId as the key.

    Returns
    -------
    flux_df : `pandas.DataFrame`
        Comparison filter fluxes.
    """
    # Get the comparison filter fluxes.
    fluxes = list()
    for fiberId, pfs_single in pfs_singles.items():
        # Remove masked values.
        good_index = pfs_single.fluxTable.mask == 0
        pfs_single.fluxTable.wavelength = pfs_single.fluxTable.wavelength[good_index]
        pfs_single.fluxTable.flux = pfs_single.fluxTable.flux[good_index]
        pfs_single.fluxTable.mask = pfs_single.fluxTable.mask[good_index]
        pfs_single.fluxTable.error = pfs_single.fluxTable.error[good_index]

        # Get the magnitude for the filter.
        for i, fc in enumerate(filter_curves.values()):
            try:
                # Get the flux and magnitude for the filter.
                filter_flux = fc.photometer(pfs_single.fluxTable) * u.nJy
                filter_mag = filter_flux.to(u.ABmag, equivalencies=u.spectral_density(fc.wavelength))

                # Store values.
                fluxes.append([fiberId, f"single.ABMag.{fc.filter_name}", filter_mag.value])
            except Exception as e:
                print(f"Error for fiberId {fiberId} and filter {fc.filter_name}: {e}")

    if not fluxes:
        return pd.DataFrame()

    # Put the comparison filter fluxes into a table.
    flux_df = pd.DataFrame(fluxes, columns=["fiberId", "filter_name", "filter_flux"])
    flux_df = flux_df.pivot(index="fiberId", columns=["filter_name"], values="filter_flux")

    return flux_df


def get_design_flux_table(pfs_config: PfsConfig) -> pd.DataFrame:
    """Get the PSF (point spread function) flux, magnitude, and id information.

    Parameters
    ----------
    pfs_config : `pfs.datamodel.PfsConfig`
        Top-end configuration.

    Returns
    -------
    info_table : `pandas.DataFrame`
        PSF flux, magnitude, and id information.
    """
    # Put the basic id information into a table.
    info_table = pd.DataFrame(
        {
            "visit": pfs_config.visit,
            "ra": pfs_config.ra,
            "dec": pfs_config.dec,
            "x": pfs_config.extractCenters(pfs_config.fiberId)[:, 1].byteswap().newbyteorder(),
            "y": pfs_config.extractCenters(pfs_config.fiberId)[:, 0].byteswap().newbyteorder(),
            "spectrograph": pfs_config.spectrograph,
        },
        index=pd.Index(pfs_config.fiberId, name="fiberId"),
    )

    # Get the identity information.
    identity_table = pd.DataFrame(
        pfs_config.getIdentity(pfs_config.fiberId), index=pd.Index(pfs_config.fiberId, name="fiberId")
    )

    # Merge the identity information into the info table.
    info_table = info_table.merge(identity_table, left_index=True, right_index=True)

    # Get the broad-band flux values from the PfsConfig.
    col_names = [f"design.flux.{n}" for n in pfs_config.filterNames[0]]
    flux_table = pd.DataFrame(pfs_config.psfFlux, columns=col_names, index=pfs_config.fiberId)

    # Get the AB magnitudes for each broad-band filter.
    for col in flux_table.columns:
        filter_name = col.split(".")[-1]
        flux_table[f"design.ABMag.{filter_name}"] = pfs_config.getPhotometry(filter_name, asABMag=True)

    # Merge the broad-band flux table into the info table.
    info_table = info_table.merge(flux_table, left_index=True, right_index=True)

    return info_table
