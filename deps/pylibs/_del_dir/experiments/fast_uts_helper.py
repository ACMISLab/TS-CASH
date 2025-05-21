import os
import sys

sys.path.append(os.path.abspath("../libs/datasets"))
sys.path.append(os.path.abspath("../libs/py-search-lib"))
sys.path.append(os.path.abspath("../libs/timeseries-models"))

from pylibs.utils.util_log import get_logger

log = get_logger()
from pylibs.utils.util_joblib import JLUtil

mem = JLUtil.get_memory()
