#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/13 18:46
# @Author  : gsunwu@163.com
# @File    : test_ali_pan.py
# @Description:
from pylibs.utils.util_ali_pan import AliPanUtil

file = "/Users/sunwu/SW-Research/AutoML/runs/automl/all_debug2_effect_n_trials_20240313_1845.csv"
import unittest
import logging

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        AliPanUtil.upload_files(file)
