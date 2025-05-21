from unittest import TestCase

from pylibs.application.datatime_helper import DateTimeHelper


class TestDTHelper(TestCase):
    def test_datetime_helper(self):
        dt = DateTimeHelper()
        dt.start_or_restart()
        dt.end()
        print(dt.collect_metrics())
