import logging

import lsst.log
import lsst.utils.tests


class TestDetectorMapResiduals(lsst.utils.tests.TestCase):
    def setUp(self):
        self.log = logging.getLogger("TestDetectorMapResiduals")
        self.log.setLevel(logging.DEBUG)
        lsst.log.setLevel("", lsst.log.DEBUG)

    def testResiduals(self):
        """Test the residuals of the detector map."""
        pass
