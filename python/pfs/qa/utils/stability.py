import logging
from pathlib import Path

import lsst.daf.persistence as dafPersist
import pandas as pd
from dataclasses import dataclass, field, InitVar
from pfs.drp.stella import ArcLineSet, DetectorMap
from pfs.qa.utils import helpers


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

    def __post_init__(self, loadData):
        self.rerun = self.repoDir / 'rerun' / self.rerunName

        if self.butler is None:
            logging.info('Creating a new butler')
            self.butler = dafPersist.Butler(self.rerun.as_posix(), calibRoot=self.calibDir.as_posix())

        self.arcLines = self.butler.get('arcLines', self.dataId)
        self.detectorMap = self.butler.get('detectorMap_used', self.dataId)

        if loadData is True:
            self.arcData = self.loadData()

    @property
    def dataId(self):
        return dict(visit=self.visit, arm=self.arm, spectrograph=self.spectrograph, label=self.label)

    @property
    def ccd(self):
        return self.arm + str(self.spectrograph)

    @property
    def uid(self):
        return f'v{self.visit}-{self.ccd}-{self.label}'

    def loadData(self, dropNaColumns: bool = True) -> pd.DataFrame:
        """Looks up the data in butler and returns a dataframe with the arcline data.

        The arcline data includes basic statistics, such as the median and sigma of the residuals.

        This method is called on init.

        Parameters
        ----------
        dropNaColumns : `bool`, optional
            Drop columns where all values are NaN. Default is True.

        """

        # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
        self.arcData = helpers.getArclineData(self.arcLines, dropNaColumns=dropNaColumns)
        self.arcData = helpers.addTraceLambdaToArclines(self.arcData, self.detectorMap)
        self.arcData = helpers.addResidualsToArclines(self.arcData)

        # Add dataId info to dataframe.
        for col in ['arm', 'spectrograph', 'visit', 'rerun', 'label']:
            self.arcData[col] = getattr(self, col)

        self.arcData.reset_index(drop=True, inplace=True)

        return self.arcData
