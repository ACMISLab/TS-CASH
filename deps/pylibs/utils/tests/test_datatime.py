#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/10/7 19:42
# @Author  : gsunwu@163.com
# @File    : test_datatime.py
# @Description:
import unittest

from pylibs.utils.util_datetime import convert_datatime_to_timestamp, convert_timestamp_to_datetime
from pylibs.utils.util_log import get_logger

log = get_logger()


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        print(convert_datatime_to_timestamp("2023-10-07 17:48:06"))
        print(convert_datatime_to_timestamp("2023-10-07 17:48:07"))
        assert convert_datatime_to_timestamp("2023-10-07 17:48:06") == 1696672086.0
        assert convert_datatime_to_timestamp("2023-10-07 17:48:07") == 1696672087.0

    def test_demo2(self):
        assert convert_timestamp_to_datetime(1696672086.0) == "2023-10-07 17:48:06"
        assert convert_timestamp_to_datetime(1696672087.0) == "2023-10-07 17:48:07"
        print(convert_timestamp_to_datetime(1696672086.0))
        print(convert_timestamp_to_datetime(1696672087.0))
