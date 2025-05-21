import time
import unittest

from pylibs.utils.util_log import get_logger
from pylibs.utils.util_system import UtilSys

log = get_logger()

from pylibs.utils.util_joblib import cache_
from pylibs.utils.util_joblib import JLUtil


def dsf():
    return "sdf"


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        print("fun result: ", fun(1))

    def test_demo2(self):
        print("fun result: ", fun(1))

    def test_demo3g(self):
        UtilSys.is_macos()
        UtilSys.is_macos()
        UtilSys.is_macos()
        UtilSys.is_macos()


@cache_
def fun(a=1):
    print("a long computation s")
    time.sleep(3)
    print("11")
    return a ** 3
