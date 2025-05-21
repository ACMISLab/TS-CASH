#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/8/21 19:05
# @Author  : gsunwu@163.com
# @File    : test_cyber_db.py
# @Description:

import unittest

from pylibs.utils.util_log import get_logger
from pylibs.utils.util_mem_db import CyberDB

log = get_logger()


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        cb = CyberDB()
        cb.save_dict("1", {"a": 1, "b": 1, "c": 1, "d": 1})

    def test_demo_1(self):
        cb = CyberDB()
        print(cb.get_dict("a2d274362e739f2c9dbc9213656dbf24d53c213d"))
