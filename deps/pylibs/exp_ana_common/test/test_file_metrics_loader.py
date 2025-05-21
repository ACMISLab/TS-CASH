#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/14 16:36
# @Author  : gsunwu@163.com
# @File    : test_file_metrics_loader.py
# @Description:

import unittest

from pylibs.exp_ana_common.uts_results_tools import FileMetricsLoader
from pylibs.utils.util_log import get_logger

log = get_logger()


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        home = "/Users/sunwu/Downloads/download_metrics/v4002_00_observation_sup1_vus_roc_0.001_random/ocsvm/Daphnet"
        fm = FileMetricsLoader(home)
        fm.load_metrics()
