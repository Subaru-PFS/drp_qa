# -*- python -*-
import os
from lsst.sconsUtils import scripts
scripts.BasicSConstruct(
    "pfs_qa",
    versionModuleName="python/%s/version.py",
    subDirList=[path for path in os.listdir(".") if os.path.isdir(path) and not path.startswith(".")] +
        ["bin"],
)
