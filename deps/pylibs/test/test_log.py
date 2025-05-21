from unittest import TestCase

from pylibs.utils.util_log import get_logger, get_mysql_logger


class TestLogger(TestCase):
    def testMysqlLogger(self):
        log=get_mysql_logger()
        UtilSys.is_debug_mode() and log.info("aaa")
