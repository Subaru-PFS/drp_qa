# Note: This test requires the LSST/PFS stack and cannot run in CI without it.
import logging

import lsst.log
import lsst.utils.tests


class TestDetectorMapResiduals(lsst.utils.tests.TestCase):
    def setUp(self):
        self.log = logging.getLogger("TestDetectorMapResiduals")
        self.log.setLevel(logging.DEBUG)
        lsst.log.setLevel("", lsst.log.DEBUG)

    def testResiduals(self):
        """Test the residuals of the detector map.

        Note: This test requires the LSST/PFS stack and cannot run in CI without it.
        When fully implemented with data, it would verify that the extended QA metrics
        (lineYieldFrac, spatialRms, wavelengthRms, velocityRms, medResolution,
        minFiberPitch, maxCrossTalk, and qaStatus) are correctly calculated and
        appended to the dmQaResidualStats DataFrame.
        """
        pass
