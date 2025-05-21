#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/21 07:26
# @Author  : gsunwu@163.com
# @File    : test_logging.py
# @Description:
import traceback
import unittest
import logging

from pylibs.utils.util_logging import UtilLogger


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        # 创建logger对象
        log=UtilLogger.get_logger()
        log2=UtilLogger.get_logger()
        log2.info("lkjldf")
