import eups
from lsst.daf.butler import Butler
from pfs.datamodel import FiberStatus, TargetType
from pfs.drp.stella.datamodel import PfsConfig, PfsFiberNorms, PfsArm
from pfs.utils.fiberids import FiberIds

import matplotlib
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pytz
from scipy.signal import medfilt

import argparse
import csv
import dataclasses
import datetime
import functools
import json
import os
import random
import re
import shutil
import textwrap
import typing

from collections.abc import Callable
from typing import Final, Literal, TypeVar

SpectrographId = Literal[1, 2, 3, 4]
"""
List of spectrographs in PFS.
"""


def main() -> None:
    """Takahashi's fiberNormsQa.

    Amateracu kaminomiyoyori yacunokapa nakanipedatete mukapitati
    codepurikapaci ikinowoni nagekacukora watarimori punemomaukeju
    pacidanimo watacitearaba conopeyumo iyukiwataraci taducapari
    unagakeriwite omopociki kotomokatarapi nagucamuru kokoropaaramuwo
    nanicikamo akiniciaraneba kotodopino tomocikikora utucemino
    yonopitowaremo kokowocimo ayanikucucimi yukikaparu tocinopagotoni
    amanopara puricakemitutu ipituginicure.
    """
    parser = argparse.ArgumentParser(
        description=layout_paragraphs(typing.cast(str, main.__doc__)),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--butler-config",
        metavar="PATH",
        required=True,
        help="""
        Path to a butler config file or its parent directory.
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="COLLECTION[,...]",
        required=True,
        type=argtype_comma_separated(str),
        help="""
        Input collections.
        """,
    )
    parser.add_argument(
        "-a",
        "--arm",
        metavar="CHAR[,...]",
        dest="arms",
        required=True,
        type=argtype_armlist,
        help="""
        Arms. "b", "r", "m", or "n".
        """,
    )
    parser.add_argument(
        "-r",
        "--reference",
        metavar="VISIT",
        required=True,
        type=int,
        help="""
        Reference visit.
        """,
    )
    parser.add_argument(
        "visits",
        metavar="visit",
        nargs="+",
        type=int,
        help="""
        Visit to assess.
        """,
    )
    args = parser.parse_args()

    butler = Butler.from_config(args.butler_config, collections=args.input)

    for visit in args.visits:
        for arm in args.arms:
            task = TakahashiFiberNormsQa(
                butler,
                args.reference,
                visit,
                arm,
            )
            task.run()


T = TypeVar("T")


def argtype_comma_separated(type: Callable[[str], T]) -> Callable[[str], list[T]]:
    """
    Comma-separated list

    The return value of this function is intended to be used as the ``type``
    argument of `argtype.ArgumentParser.add_argument`

    Parameters
    ----------
    type : `Callable` [[`str`], `T`]
        Type of elements, or converter to the type.

    Returns
    -------
    func : `Callable`
        This function takes a string and returns a list of elements of type
        ``typ``.
    """

    def _argtype_comma_separated(s: str) -> list[T]:
        """
        Comma-separated list

        This function is intended to be used as the ``type`` argument of
        `argtype.ArgumentParser.add_argument`

        Parameters
        ----------
        s : `str`
            Comma-separated list.

        Returns
        -------
        elements : `list`
            List of elements.
        """
        elements: list[T] = []
        for elem in s.split(","):
            elem = elem.strip()
            if not elem:
                continue
            try:
                t = type(elem)
            except Exception:
                typename = getattr(type, "__name__", "")
                if typename:
                    typename += " "
                raise argparse.ArgumentTypeError(f"invalid {typename}value: '{elem}'.") from None

            elements.append(t)

        return elements

    return _argtype_comma_separated


def argtype_armlist(s: str) -> list[str]:
    """
    List of arms

    This function is intended to be used as the ``type`` argument of
    `argtype.ArgumentParser.add_argument`

    Parameters
    ----------
    s : `str`
        List of arms, optionally separated by ",".

    Returns
    -------
    elements : `list`
        List of elements.
    """
    ignored_chars = {",", " "}
    orders = {
        "b": 0,
        "r": 1,
        "m": 2,
        "n": 3,
    }
    elements = {c for c in s if c not in ignored_chars}
    tagged_elements = []
    for c in elements:
        try:
            tagged_elements.append((orders[c], c))
        except KeyError:
            raise argparse.ArgumentTypeError(f"invalid arm name '{c}'.") from None

    tagged_elements.sort()
    return [elem for tag, elem in tagged_elements]


def layout_paragraphs(text: str) -> str:
    """
    Lay out paragraphs so that they will fill the columns of the terminal.

    Parameters
    ----------
    text
        Text consisting of paragraphs separated by two linebreaks.

    Returns
    -------
    newtext
        Layed-out text.
    """
    width = max(1, shutil.get_terminal_size().columns - 2)
    return (
        "\n\n".join(
            textwrap.fill(re.sub(r"\s+", " ", paragraph.strip()), width=width)
            for paragraph in text.strip().split("\n\n")
        )
        + "\n"
    )


vmin: Final[float] = 0.97
vmax: Final[float] = 1.03
vmin2: Final[float] = 0.0
vmax2: Final[float] = 0.05

wavelength_ranges: Final[dict[str, tuple[float, float]]] = {
    "b": (450, 650),
    "r": (650, 950),
    "m": (650, 950),
    "n": (950, 1250),
}


@dataclasses.dataclass
class TakahashiFiberNormsQaStat:
    hst: str
    insrot: float
    azimuth: float
    sigma_std_median: float
    sigma_std_95: float
    twosigma_std: float
    twosigma_std_95: float
    sigma_iqr: float
    df_mtp: pd.DataFrame


@dataclasses.dataclass
class TakahashiFiberNormsQaPlotConfig:
    fontsize: float = 20
    figsize: tuple[float, float] = (30, 55)

    @property
    def fontsize_large(self) -> float:
        return self.fontsize * 1.25

    @property
    def fontsize_small(self) -> float:
        return self.fontsize * 0.85

    @property
    def fontsize_x_small(self) -> float:
        return self.fontsize * 0.75

    @property
    def fontsize_xx_small(self) -> float:
        return self.fontsize * 0.7


class TakahashiFiberNormsQa:
    def __init__(
        self,
        butler: Butler,
        ref_visit: int,
        visit: int,
        arm: str,
        *,
        plot_config=TakahashiFiberNormsQaPlotConfig(),
    ) -> None:
        self.ref_visit = ref_visit
        self.visit = visit
        self.arm = arm
        self.plot_config = plot_config

        self.pfsConfig = butler.get("pfsConfig", visit=visit)
        self.fiberNorms = butler.get("fiberNorms", visit=visit, arm=arm)

        self.pfsArmDict = {
            spec: butler.get("pfsArm", visit=visit, arm=arm, spectrograph=spec)
            for spec in typing.get_args(SpectrographId)
        }

        self.pfsArmRefDict = {
            spec: butler.get("pfsArm", visit=ref_visit, arm=arm, spectrograph=spec)
            for spec in typing.get_args(SpectrographId)
        }

        # Threshold for odd fiber
        self.sigma_flag_entire = {
            "b": 0.01090,
            "r": 0.00799,
            "m": 0.00799,
            "n": 0.00799,
        }[arm]

        # These two are used in captions of figures.
        self.datastore = str(butler._config.configDir)
        self.collections = str(butler.collections.defaults)

        self._cache_pfsArmRatio: dict[SpectrographId, PfsArmRatio] = {}
        self._cache_cleaned_fiberNorms: PfsFiberNorms | None = None

    def run(self) -> None:
        outPath = f"compare_quartz_result/{self.arm}-arm"
        outputfigurePath = f"output/fiberNormsQA/{self.arm}-arm"

        stat = self.sigma_perFiber_perMTP()

        os.makedirs(outPath, exist_ok=True)
        stat.df_mtp.to_csv(
            f"{outPath}/sigma_perFiber_perMTP_{self.visit}over{self.ref_visit}.csv", index=False
        )

        # save estimated value
        jsonobj = dataclasses.asdict(stat)
        jsonobj.pop("df_mtp")
        with open(f"{outPath}/sigma_perFiber_perMTP_{self.visit}over{self.ref_visit}.json", "w") as fcomp:
            json.dump(jsonobj, fcomp, indent="  ", ensure_ascii=False)

        fig = self.make_plot(stat)

        os.makedirs(outputfigurePath, exist_ok=True)
        fig.savefig(f"{outputfigurePath}" + f"/{self.visit}_over{self.ref_visit}.png", bbox_inches="tight")
        plt.close()

    def sigma_perFiber_perMTP(self) -> TakahashiFiberNormsQaStat:
        df_fibid = pd.DataFrame(FiberIds().data)
        df_mtp = df_fibid[["fiberId", "mtp_A"]]
        df_mtp = df_mtp.apply(get_mtpgroup, axis=1, result_type="expand")
        df_mtp["arm"] = self.arm
        df_mtp["SM"] = int(0)
        df_mtp["sigma"] = np.nan

        arr = np.zeros((1))

        for spec in typing.get_args(SpectrographId):
            try:
                pfsArmRatio = self._get_pfsArmRatio(spec)

                pfsArm = pfsArmRatio.pfsArm
                ratio = pfsArmRatio.ratio

                df_mtp.loc[df_mtp["fiberId"].isin(pfsArm.fiberId), "SM"] = spec

                # Calculate standard deviation for both raw data
                sigma = np.nanstd(ratio, axis=1)
                arr = np.append(arr, sigma)

                # Calculate IQR_sigma for both raw data
                ## Q75-Q25
                c = 0.741
                q3, q1 = 75, 25
                sigma_iqr4 = func_sigma_iqr(q3, q1, ratio, c)

            except LookupError:
                sigma_iqr4 = np.full(len(pfsArmRatio.pfsConfig.fiberId), np.nan)

            df_mtp.loc[df_mtp["fiberId"].isin(pfsArm.fiberId), ["sigma"]] = sigma_iqr4

            ################## Observing parameters #############
            ### For target visit
            inr = pfsArm.metadata["INSROT"]
            azi = pfsArm.metadata["AZIMUTH"]
            hst = pfsArm.metadata["HST"]

        arr = arr[1:]

        return TakahashiFiberNormsQaStat(
            hst=hst,
            insrot=inr,
            azimuth=azi,
            sigma_std_median=np.median(arr),
            sigma_std_95=np.nanpercentile(arr, 95),
            twosigma_std=np.median(arr * 2),
            twosigma_std_95=np.nanpercentile(arr * 2, 95),
            sigma_iqr=np.nanmedian(df_mtp["sigma"]),
            df_mtp=df_mtp,
        )

    def _get_pfsArmRatio(self, spec: SpectrographId) -> "PfsArmRatio":
        pfsArmRatio = self._cache_pfsArmRatio.get(spec)
        if pfsArmRatio is None:
            pfsArmRatio = self._cache_pfsArmRatio[spec] = PfsArmRatio(
                self.pfsArmRefDict[spec], self.pfsArmDict[spec], self.pfsConfig
            )

        return pfsArmRatio

    def _get_cleaned_fiberNorms(self) -> PfsFiberNorms:
        """Clean ``self.fiberNorms`` and get it.

        Returns
        -------
        fiberNorms : `PfsFiberNorms`
            Cleaned fiberNorms.
        """
        cache = self._cache_cleaned_fiberNorms
        if cache is not None:
            return cache

        wmin, wmax = wavelength_ranges[self.arm]
        pfsConfig = self.pfsConfig.select(fiberStatus=FiberStatus.GOOD)

        fiberNorms = self.fiberNorms
        fiberNorms = fiberNorms[np.isin(fiberNorms.fiberId, pfsConfig.fiberId)]
        fiberNorms.values = np.where(
            (wmin <= fiberNorms.wavelength) & (fiberNorms.wavelength <= wmax), fiberNorms.values, np.nan
        )
        fiberNorms.values = np.where(np.isinf(fiberNorms.values), np.nan, fiberNorms.values)

        self._cache_cleaned_fiberNorms = fiberNorms
        return fiberNorms

    def make_plot(self, stat: TakahashiFiberNormsQaStat) -> plt.Figure:
        df = stat.df_mtp
        df = df[df.mtpGroup.str.contains("U") | df.mtpGroup.str.contains("D")]
        df = df[df.SM != 0]
        df = df.reset_index()
        ins = stat.insrot
        azi = np.round(stat.azimuth, 2)

        fig = plt.figure(
            num="fiberNormsQA", figsize=self.plot_config.figsize, clear=True, facecolor="w", edgecolor="k"
        )

        arr = np.array(df.sigma)
        df_odd = df[(arr > self.sigma_flag_entire)]  # extract odd MTPs/fibers
        df_odd = df_odd.sort_values("sigma", ascending=False)

        if len(df_odd) > 10:
            df_odd = df_odd[0:15]
        else:
            df_odd = df_odd.head(int(len(df_odd) * 0.25))

        df_odd = df_odd.reset_index()
        n_odd = len(df_odd)

        ncols_odd = 5
        nrows_odd = int(n_odd / ncols_odd)  ### the number of rows for plot of odd fibers

        nrows = 4 + 4  # later four rows are for showing spectra

        gs_master = gridspec.GridSpec(
            nrows=nrows, ncols=6, height_ratios=np.full(nrows, 1), wspace=0.7, hspace=0.5
        )

        ax5 = fig.add_subplot(gs_master[1, 0], aspect="equal")  # --For PFI image with pfsArm
        ax6 = fig.add_subplot(gs_master[1, 1], aspect="equal")  # --For PFI image with pfsArm
        ax7 = fig.add_subplot(gs_master[2, 0], aspect="equal")  # --For PFI image with pfsArm
        ax8 = fig.add_subplot(gs_master[2, 1], aspect="equal")  # --For PFI image with pfsArm
        ax9 = fig.add_subplot(gs_master[3, 0], aspect="equal")  # --For PFI image with fiberNorms
        ax10 = fig.add_subplot(gs_master[3, 1])  # , aspect='equal') #--For PFI image with fiberNorms

        # For 2D spectral image
        ax13 = fig.add_subplot(gs_master[1:3, 2:5])

        ## Something
        ax14 = fig.add_subplot(gs_master[3, 2:5])

        ## MTP group vs. spectrograph ID
        ax15 = fig.add_subplot(gs_master[1:4, 5])

        ############################################### PLOT DATA ###############################################################

        # [5]
        self._make_plot_sigma_per_fiber(ax14, df)

        # [3]
        self._make_plot_used_mtps(ax15, df)

        # [2]
        self._make_plot_quartz_ratio_by_wavelength(ax13, df)

        # [1] PFI IMAGES (pfsArm.flux/pfsArm.norm)
        fig.text(0.15, 0.78, "[1] Quartz ratio measured from pfsArm", fontsize=self.plot_config.fontsize)
        self._make_plot_quartz_ratio_by_position_n_sigma(ax5, df, n=2)
        self._make_plot_quartz_ratio_by_position_median(ax6, df)
        self._make_plot_quartz_ratio_by_position_at_pixel(ax7, df, pixel_index=1500)
        self._make_plot_quartz_ratio_by_position_at_pixel(ax8, df, pixel_index=3500)

        # [4] PFI IMAGES (fiberNorms)
        fig.text(0.15, 0.59, "[4] fiberNorms.values of target quartz", fontsize=self.plot_config.fontsize)
        self._make_plot_fiberNorms_by_position(ax9)
        self._make_plot_fiberNorms_by_wavelength(ax10)

        if len(df_odd) >= 15:
            # [6] 1D spectra measured from pfsArm
            gs_spec = gridspec.GridSpecFromSubplotSpec(
                nrows=nrows_odd, ncols=ncols_odd, subplot_spec=gs_master[4:7, :], wspace=0.3, hspace=0.5
            )
            fig.text(
                0.35,
                0.49,
                f"[6] Example spectra for flagged fibers with large flux scatters (red marked in [2])",
                fontsize=self.plot_config.fontsize,
            )
            for i in range(n_odd):
                axs = fig.add_subplot(gs_spec[i // ncols_odd, i % ncols_odd])
                self._make_plot_1d_spectra(axs, df_odd, i)

            # [7] 1D spectra measured from fiberNorms
            gs_spec2 = gridspec.GridSpecFromSubplotSpec(
                nrows=1, ncols=ncols_odd, subplot_spec=gs_master[7:, :], wspace=0.3, hspace=0.5
            )
            fig.text(
                0.35,
                0.19,
                f"[7] Randomly selected spectra obtained by fiberNorms.values",
                fontsize=self.plot_config.fontsize,
            )
            # extract fiberIds for plotting example spectra
            list_ex_fibernorms = random.sample(list(df.fiberId), ncols_odd)
            for i in range(ncols_odd):
                axs = fig.add_subplot(gs_spec2[i // ncols_odd, i % ncols_odd])
                self._make_plot_example_spectra(axs, df, list_ex_fibernorms[i])

        ### end plotting ###

        sigma_typ = np.nanmedian(df.sigma)

        if sigma_typ >= self.sigma_flag_entire:
            flag = "Yes"
            strcolor = "red"
        else:
            flag = "No"
            strcolor = "black"

        obsdate = utc2hst(self.fiberNorms.metadata["DATEOBS"])

        fig.text(0.1, 0.85, f"fiberNormsQA ver. 1.0", fontsize=self.plot_config.fontsize_x_small)
        fig.text(
            0.7,
            0.85,
            f"Flag for fiber throughput variation={flag}",
            color=strcolor,
            fontsize=self.plot_config.fontsize_large,
            bbox=dict(facecolor="yellow", alpha=1.0),
        )
        fig.text(
            0.1,
            0.83,
            f"visit_target={self.visit}",
            fontweight="bold",
            fontsize=self.plot_config.fontsize_large,
        )
        fig.text(
            0.1,
            0.82,
            f"visit_reference={self.ref_visit}, arm={self.arm}, obsdate={obsdate} ,insrot={ins}deg, azimuth={azi}deg",
            fontsize=self.plot_config.fontsize_large,
        )
        pfs_pipe2d = eups.getSetupVersion("pfs_pipe2d")
        fig.text(
            0.1,
            0.81,
            f"datastore={self.datastore}, collections={self.collections}, pfs_pipe2d={pfs_pipe2d}",
            fontsize=self.plot_config.fontsize_large,
        )

        if (self.visit == self.ref_visit) or (len(df_odd) == 0):
            plt.annotate(
                f"fiberNormsQA is to monitor fiber throughput variation. We took quartz flux ratios for some of the figures to investigate how much quartz flux\n varies with time."
                + "i.e., quartz ratio = pfsArm.flux(visit_target)/pfsArm.flux(visit_reference). "
                + "Referenced quartz is the first one in the data set \n basically (see visit number above).\n"
                + "Sigma is measured from 0.741*(Q75-Q25). "
                + "Descriptions for each sub-fig component are as follows.\n"
                + "[1] PFI image of quartz ratios. 2 sigma per fier and median flux ratio per fiber are represent in the upper left and in the upper right figure, \n respectively."
                + " We also represent median flux ratios at two different wavelength point via PFI images.\n"
                + "[2] 2D spectrun of quartz ratio. If measure sigma of a fiber is larger then"
                + f" {self.sigma_flag_entire}, red closs marks fiber Id in the left side.\n"
                + "[3] Used MTPs compared with corresponding spectrograph ID. The MTP groups in which all MTPs were used were represented in green, \n and partially unused MTPs are represented in grey.\n"
                + "[4] fiberNorms.values of target quartz. Plotting bounds are "
                + r"2.5$\sigma$"
                + "\n"
                + "[5] Measured sigma for each fibers. The median values in all MTP groups represent in blue dashed line and 4 sigma \nrepresent in red dashed line. This figure is to check flux variation with MTP unit. "
                + "MTP groups above the red dashed line, \n indicating that the MTP group has a large flux variation. \n",
                (-95, -1.5),
                fontsize=self.plot_config.fontsize_large,
                xycoords="axes fraction",
            )
        else:
            plt.annotate(
                f"fiberNormsQA is to monitor fiber throughput variation. We took quartz flux ratios for some of the figures to investigate how much quartz flux\n varies with time."
                + "i.e., quartz ratio = pfsArm.flux(visit_target)/pfsArm.flux(visit_reference). "
                + "Referenced quartz is the first one in the data set \n basically (see visit number above).\n"
                + "Sigma is measured from 0.741*(Q75-Q25). "
                + "Descriptions for each sub-fig component are as follows.\n"
                + "[1] PFI image of quartz ratios. 2 sigma per fier and median flux ratio per fiber are represent in the upper left and in the upper right figure, \n respectively."
                + " We also represent median flux ratios at two different wavelength point via PFI images.\n"
                + "[2] 2D spectrun of quartz ratio. If measure sigma of a fiber is larger then"
                + f" {self.sigma_flag_entire}, red closs marks fiber Id in the left side.\n"
                + "[3] Used MTPs compared with corresponding spectrograph ID. The MTP groups in which all MTPs were used were represented in green.\n"
                + "[4] fiberNorms.values of target quartz. Plotting bounds are "
                + r"2.5$\sigma$"
                + "\n"
                + "[5] Measured sigma for each fibers. The median values in all MTP groups represent in blue dashed line and 4 sigma \nrepresent in red dashed line. This figure is to check flux variation with MTP unit. "
                + "MTP groups above the red dashed line, \n indicating that the MTP group has a large flux variation. \n"
                + "[6] Spectra of FIBER with large sigma with a maximum of 15. Blue plots represent normalized spectra obtained by quartz ratio \n (i.e. pfsArm.flux(target)/pfsArm.norm(target)/pfsArm.flux(reference)/pfsArm.norm(reference)) and light green plots represent median filtered \n spectra."
                + "\nFiberId are shown in upper left. Mesured sigma per fiber are shown in upper right. Red dashed lines represent 2sigma lines of median filtered spectra.\n"
                + "[7] Randomly selected spectra obtained from fiberNorms.values of target quartz",
                (-5.5, -2.5),
                fontsize=self.plot_config.fontsize_large,
                xycoords="axes fraction",
            )

        fig.tight_layout()
        return fig

    def _make_plot_sigma_per_fiber(self, ax: matplotlib.axes.Axes, df: pd.DataFrame) -> None:
        ax.set_title("[5] Sigma per fiber", fontsize=self.plot_config.fontsize)
        ax.scatter(df.mtpGroup, df.sigma, c="black")

        sigma_median = np.nanmedian(df.sigma)
        ax.axhline(sigma_median, ls="dashed", c="blue", zorder=0)
        ax.annotate(
            "median",
            (0.9, 0.1),
            c="blue",
            fontsize=self.plot_config.fontsize_xx_small,
            xycoords="axes fraction",
        )
        clip = 4
        sigma_of_sigma = sigma_median + clip * np.std(df.sigma)
        ax.axhline(sigma_of_sigma, ls="dashed", c="red", zorder=0)
        ax.text(
            "D2-1-4", 1.2 * sigma_of_sigma, "4 sigma", c="red", fontsize=self.plot_config.fontsize_xx_small
        )
        ax.text(
            0.85,
            0.9,
            f"${clip}\sigma$ = {sigma_of_sigma:.4f}",
            color="red",
            transform=ax.transAxes,
            fontsize=self.plot_config.fontsize_small,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax.text(
            0.85,
            0.8,
            f"median = {sigma_median:.4f}",
            color="blue",
            transform=ax.transAxes,
            fontsize=self.plot_config.fontsize_small,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        ax.grid(axis="both", which="both", linestyle="--", linewidth=0.5)
        ax.set_xlabel("MTP group", self.plot_config.fontsize_x_small)
        ax.set_ylabel("$\sigma$ (0.741*(Q75-Q25))", self.plot_config.fontsize_x_small)
        ax.tick_params("x", labelrotation=90)

    def _make_plot_used_mtps(self, ax: matplotlib.axes.Axes, df: pd.DataFrame) -> None:
        ax.set_title("[3] Used MTPs", fontsize=self.plot_config.fontsize)
        xdata, ydata = df.SM[df.SM != 0.0], df.mtpGroup[df.SM != 0.0]
        ax.scatter(xdata, ydata, c="olivedrab")
        ## The following MTPs had not been estimated sigma due to some error (e.g., could not read butler)
        xdata, ydata = df.SM[df.SM == 0.0], df.mtpGroup[df.SM == 0.0]
        ax.scatter(xdata, ydata, c="gray")

        ax.grid(axis="both", which="both", linestyle="--", linewidth=0.5)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlabel("spectrograph ID", fontsize=self.plot_config.fontsize)
        ax.set_ylabel("MTP group", fontsize=self.plot_config.fontsize_x_small)

    def _make_plot_quartz_ratio_by_wavelength(self, ax: matplotlib.axes.Axes, df: pd.DataFrame) -> None:
        # [2] 2D spec of quartz ratio
        ax.set_title(
            f"[2] 2D spectrum of quartz ratio measured from pfsArm.flux", fontsize=self.plot_config.fontsize
        )
        ax.set_xlabel("wavelength [nm]", fontsize=self.plot_config.fontsize_large)
        ax.set_ylabel("fiberId", fontsize=self.plot_config.fontsize_large)

        wmin, wmax = wavelength_ranges[self.arm]

        for spec in typing.get_args(SpectrographId):
            pfsArmRatio = self._get_pfsArmRatio(spec)
            pfsArm = pfsArmRatio.pfsArm
            ratio = pfsArmRatio.ratio
            fibs = np.array(
                [
                    np.full_like(pfsArm.wavelength[np.where(pfsArm.fiberId == f)[0][0]], f)
                    for f in pfsArm.fiberId
                ]
            )

            # plot quartz ratio as contour (fiberId vs wavelength)
            sc = ax.scatter(
                pfsArm.wavelength, fibs, c=ratio, vmin=vmin, vmax=vmax, s=0.6, alpha=1.0, label="quartz"
            )

            oddfib = df.fiberId[(df.SM == spec) & (df.sigma > self.sigma_flag_entire)].values
            for i_fib in oddfib:
                ax.scatter(wmin, i_fib, marker="+", color="red", s=150, alpha=0.8)

        divider = make_axes_locatable(ax)  # AxesDivider related to ax
        cax = divider.append_axes("right", size="2%", pad=0.1)  # create new axes

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, cax=cax)

    def _make_plot_quartz_ratio_by_position_n_sigma(
        self, ax: matplotlib.axes.Axes, df: pd.DataFrame, n: float
    ) -> None:
        for spec in typing.get_args(SpectrographId):
            pfsconfig = self._get_pfsArmRatio(spec).pfsConfig
            sc = ax.scatter(
                pfsconfig.pfiCenter[:, 0],
                pfsconfig.pfiCenter[:, 1],
                c=np.array(df.sigma[df["SM"] == spec]) * n,
                vmin=vmin2,
                vmax=vmax2,
                s=30.0,
                alpha=1.0,
                label=f"{n}sigma",
            )
            ax.set_xlim(xmin=-250, xmax=250)
            ax.set_ylim(ymin=-250, ymax=250)
            ax.yaxis.set_ticks_position("left")
            ax.set_title(f"{n}sigma", fontsize=self.plot_config.fontsize)
            ax.set_xlabel("X(PFI) [mm]", fontsize=self.plot_config.fontsize)
            ax.set_ylabel("Y(PFI) [mm]", fontsize=self.plot_config.fontsize)

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, ax=ax, location="right", fraction=0.04, alpha=1.0)

    def _make_plot_quartz_ratio_by_position_median(self, ax: matplotlib.axes.Axes, df: pd.DataFrame) -> None:
        for spec in typing.get_args(SpectrographId):
            pfsArmRatio = self._get_pfsArmRatio(spec)
            ratio = pfsArmRatio.ratio
            pfsconfig = pfsArmRatio.pfsConfig

            # Plot median
            median_array = np.array([np.nanmedian(ratio[i_fiber]) for i_fiber in range(len(ratio))])
            sc = ax.scatter(
                pfsconfig.pfiCenter[:, 0],
                pfsconfig.pfiCenter[:, 1],
                c=median_array,
                vmin=vmin,
                vmax=vmax,
                s=30.0,
                alpha=1.0,
                label=f"median per fiber",
            )
            ax.set_xlim(xmin=-250, xmax=250)
            ax.set_ylim(ymin=-250, ymax=250)
            ax.yaxis.set_ticks_position("left")
            ax.set_title(f"median per fiber", fontsize=self.plot_config.fontsize)
            ax.set_xlabel("X(PFI) [mm]", fontsize=self.plot_config.fontsize)
            ax.set_ylabel("Y(PFI) [mm]", fontsize=self.plot_config.fontsize)

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, ax=ax, location="right", fraction=0.04, alpha=1.0)

    def _make_plot_quartz_ratio_by_position_at_pixel(
        self, ax: matplotlib.axes.Axes, df: pd.DataFrame, pixel_index: int
    ) -> None:
        for spec in typing.get_args(SpectrographId):
            pfsArmRatio = self._get_pfsArmRatio(spec)
            ratio = pfsArmRatio.ratio
            pfsconfig = pfsArmRatio.pfsConfig
            pfsArm = pfsArmRatio.pfsArm

            ratio_lam = np.array([ratio[i_fiber][pixel_index] for i_fiber in range(len(ratio))])
            sc = ax.scatter(
                pfsconfig.pfiCenter[:, 0],
                pfsconfig.pfiCenter[:, 1],
                c=ratio_lam,
                vmin=vmin,
                vmax=vmax,
                s=30.0,
                alpha=1.0,
            )
            ax.set_xlim(xmin=-250, xmax=250)
            ax.set_ylim(ymin=-250, ymax=250)
            ax.yaxis.set_ticks_position("left")
            lam_point = np.round(pfsArm.wavelength[0][pixel_index], 3)
            ax.set_title(f"at {lam_point} [nm]", fontsize=self.plot_config.fontsize)
            ax.set_xlabel("X(PFI) [mm]", fontsize=self.plot_config.fontsize)
            ax.set_ylabel("Y(PFI) [mm]", fontsize=self.plot_config.fontsize)

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, ax=ax, location="right", fraction=0.04, alpha=1.0)

    def _make_plot_fiberNorms_by_position(self, ax: matplotlib.axes.Axes) -> None:
        pfsConfig = self.pfsConfig.select(fiberStatus=FiberStatus.GOOD)
        fiberNorms = self._get_cleaned_fiberNorms()
        fiberNorms.plot(pfsConfig, axes=ax, lower=2.5, upper=2.5)

        ax.set_title(f"median per fiber", fontsize=self.plot_config.fontsize)
        ax.set_xlabel("X(PFI) [mm]", fontsize=self.plot_config.fontsize)
        ax.set_ylabel("Y(PFI) [mm]", fontsize=self.plot_config.fontsize)

    def _make_plot_fiberNorms_by_wavelength(self, ax: matplotlib.axes.Axes) -> None:
        # [4] 2D image of fiberNorms.values
        pfsConfig = self.pfsConfig.select(fiberStatus=FiberStatus.GOOD)
        fiberNorms = self._get_cleaned_fiberNorms()

        values = np.nanmedian(fiberNorms.values, axis=1)
        good = np.isfinite(values)
        median = np.median(values[good])
        rms = np.sqrt(np.mean(np.square(values[good])))
        lower = median - 2.5 * np.std(values)
        upper = median + 2.5 * np.std(values)
        norm = Normalize(vmin=lower, vmax=upper)
        fib_norm = np.array(
            [
                np.full_like(fiberNorms.wavelength[np.where(fiberNorms.fiberId == f)[0][0]], f)
                for f in fiberNorms.fiberId
            ]
        )
        sc = ax.scatter(
            fiberNorms.wavelength,
            fib_norm,
            c=fiberNorms.values,
            vmin=lower,
            vmax=upper,
            s=10,
            cmap="coolwarm",
        )
        divider = make_axes_locatable(ax)  # AxesDivider related to ax
        cax = divider.append_axes("right", size="5%", pad=0.1)  # create new axes

        ax.set_ylabel("fiber index", fontsize=self.plot_config.fontsize_small)
        ax.set_xlabel("wavelength", fontsize=self.plot_config.fontsize_small)
        ax.set_title(f"2D spectrum", fontsize=self.plot_config.fontsize)

        fig = ax.figure
        if fig is not None:
            fig.colorbar(sc, cax=cax)

    def _make_plot_1d_spectra(self, ax: matplotlib.axes.Axes, df_odd: pd.DataFrame, i_odd: int) -> None:
        spec = df_odd["SM"].loc[i_odd]

        pfsArmRatio = self._get_pfsArmRatio(spec)
        pfsArm = pfsArmRatio.pfsArm

        ratio = pfsArmRatio.normalized_ratio
        ratio_mf = pfsArmRatio.filtered_normalized_ratio

        fib = df_odd.fiberId[i_odd]
        inx = np.where(pfsArm.fiberId == fib)[0]
        n_sigma = 2

        xdata = pfsArm.wavelength[inx]
        ydata = ratio[inx]
        ax.scatter(xdata, ydata, color="royalblue", s=10, label=f"{fib}")

        xdata_m = pfsArm.wavelength[inx]
        ydata_m = ratio_mf[inx]
        std = np.nanstd(ydata_m)
        ax.scatter(xdata_m, ydata_m, color="limegreen", s=10, label=f"{fib}(mf)")

        ax.text(
            0.68,
            0.9,
            f"$\sigma$ =" + f"{df_odd.sigma.iloc[i_odd]:.4f}",
            transform=ax.transAxes,
            fontsize=self.plot_config.fontsize_small,
            bbox=dict(facecolor="yellow", alpha=1.0),
        )
        ax.axhline(
            y=np.nanmedian(ydata_m) + n_sigma * df_odd.sigma.iloc[i_odd],
            color="tomato",
            linestyle="dashed",
        )
        ax.axhline(
            y=np.nanmedian(ydata_m) - n_sigma * df_odd.sigma.iloc[i_odd],
            color="tomato",
            linestyle="dashed",
        )

        ax.set(xlabel="wavelength [nm]", ylabel="normalized flux")

        ax.set_ylim(
            ymin=np.nanmedian(ydata_m) - (n_sigma + 6) * std,
            ymax=np.nanmedian(ydata_m) + (n_sigma + 6) * std,
        )
        ax.legend(loc="upper left", fontsize=self.plot_config.fontsize_x_small)
        ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5)

        ax.set_title(f"{df_odd.mtp_A[i_odd]}", fontsize=self.plot_config.fontsize)

    def _make_plot_example_spectra(self, ax: matplotlib.axes.Axes, df: pd.DataFrame, fiber_id: int) -> None:
        fiberNorms = self._get_cleaned_fiberNorms()

        inx_nrm = np.where(fiberNorms.fiberId == fiber_id)[0]
        ax.scatter(
            fiberNorms.wavelength[inx_nrm],
            fiberNorms.values[inx_nrm],
            color="navy",
            s=10,
            label=f"{fiber_id}",
        )

        xdata_m = fiberNorms.wavelength[inx_nrm]
        xdata_m = xdata_m[~np.isnan(fiberNorms.values[inx_nrm])]
        ydata_m = medfilt(fiberNorms.values[inx_nrm][~np.isnan(fiberNorms.values[inx_nrm])], kernel_size=15)

        ydata_m_ = medfilt(fiberNorms.values[inx_nrm][~np.isnan(fiberNorms.values[inx_nrm])], kernel_size=15)

        std = np.nanstd(ydata_m)
        ax.scatter(xdata_m, ydata_m, color="limegreen", s=10, label=f"{fiber_id}(mf)")

        n_sigma = 2

        ax.axhline(y=np.nanmedian(ydata_m) + n_sigma * std, color="red", linestyle="dashed")
        ax.axhline(y=np.nanmedian(ydata_m) - n_sigma * std, color="red", linestyle="dashed")

        ax.set(xlabel="wavelength [nm]", ylabel="normalized flux")
        ax.set_ylim(
            ymin=np.nanmedian(ydata_m) - (n_sigma + 2) * std,
            ymax=np.nanmedian(ydata_m) + (n_sigma + 2) * std,
        )
        ax.legend(loc="upper left", fontsize=self.plot_config.fontsize_x_small)
        ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5)

        ax.set_title(
            f"{df.mtp_A[df.fiberId==fiber_id].to_string(index=False)}",
            fontsize=self.plot_config.fontsize,
        )


# routine to get MTP group
def get_mtpgroup(df):
    df["mtpGroup"] = df.mtp_A[0:6]
    return df


def func_sigma_iqr(Q3, Q1, data, coff):
    q3, q1 = np.nanpercentile(data, [Q3, Q1], axis=1)
    sigma_iqr_perfib = coff * (q3 - q1)

    return sigma_iqr_perfib


class PfsArmRatio:
    def __init__(self, pfsArmRef: PfsArm, pfsArm: PfsArm, pfsConfig: PfsConfig) -> None:
        pfsConfig = pfsConfig.select(targetType=~TargetType.ENGINEERING)
        pfsConfig = pfsConfig.select(fiberStatus=FiberStatus.GOOD)

        pfsArmRef = pfsArmRef[np.isin(pfsArmRef.fiberId, pfsConfig.fiberId)]
        pfsArm = pfsArm[np.isin(pfsArm.fiberId, pfsConfig.fiberId)]

        wmin, wmax = wavelength_ranges[pfsArm.identity.arm]

        self.pfsConfig = pfsConfig
        self.pfsArmRef = pfsArmRef
        self.pfsArm = pfsArm
        self.wmin = wmin
        self.wmax = wmax

    @functools.cached_property
    def ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        ratio = self._naive_ratio
        xx, yy = np.meshgrid(np.nanmedian(ratio, axis=0), np.nanmedian(ratio, axis=1) / np.nanmedian(ratio))
        return ratio / xx / yy

    @functools.cached_property
    def normalized_ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        ratio = self._naive_normalized_ratio
        xx, yy = np.meshgrid(np.nanmedian(ratio, axis=0), np.nanmedian(ratio, axis=1) / np.nanmedian(ratio))
        return ratio / xx / yy

    @functools.cached_property
    def filtered_normalized_ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        ratio = self._naive_normalized_ratio
        ratio = medfilt(ratio, kernel_size=(1, 15))
        xx, yy = np.meshgrid(np.nanmedian(ratio, axis=0), np.nanmedian(ratio, axis=1) / np.nanmedian(ratio))
        return ratio / xx / yy

    @functools.cached_property
    def _naive_ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        ratio = self.pfsArm.flux / self.pfsArmRef.flux
        # discard around the edge of wavelength (noisy)
        ratio = np.where(
            (self.wmin <= self.pfsArm.wavelength) & (self.pfsArm.wavelength <= self.wmax), ratio, np.nan
        )
        ratio = np.where(np.isinf(ratio), np.nan, ratio)
        return ratio

    @functools.cached_property
    def _naive_normalized_ratio(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        ratio = (self.pfsArm.flux / self.pfsArm.norm) / (self.pfsArmRef.flux / self.pfsArmRef.norm)
        # discard around the edge of wavelength (noisy)
        ratio = np.where(
            (self.wmin <= self.pfsArm.wavelength) & (self.pfsArm.wavelength <= self.wmax), ratio, np.nan
        )
        ratio = np.where(np.isinf(ratio), np.nan, ratio)
        return ratio


def utc2hst(utc_str: str) -> str:
    """
    Convert UTC to HST

    Parameters
    ----------
    utc_str : `str`
        UTC in ISO format.

    Returns
    -------
    hst : `str`
        HST in ISO format (``YYYY-mm-ddTHH:MM:SS.S``)
    """
    # ISO to datetime
    utc_dt = datetime.datetime.fromisoformat(utc_str.replace("Z", "+00:00"))

    # set time zone
    utc_dt = pytz.utc.localize(utc_dt) if utc_dt.tzinfo is None else utc_dt

    # convert to the hst time zone（UTC-10:00）
    hst_tz = pytz.timezone("US/Hawaii")
    hst_dt = utc_dt.astimezone(hst_tz)

    hst_formatted = hst_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    hst_formatted = hst_formatted[:-5]

    return hst_formatted


if __name__ == "__main__":
    main()
