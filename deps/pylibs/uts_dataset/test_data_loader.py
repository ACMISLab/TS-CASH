#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/3 17:39
# @Author  : gsunwu@163.com
# @File    : test_data_loader.py
# @Description:
from pylibs.uts_dataset.dataset_loader import UTSDataset

import unittest


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        assert UTSDataset.get_best_sample_rate("NASA-MSL", "C-2.test.out") == 0.51
        self.assertEqual(UTSDataset.get_best_sample_rate("YAHOO", "Yahoo_A1real_9_data.out"), 0.44)
