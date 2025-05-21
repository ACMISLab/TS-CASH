#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/7 09:39
# @Author  : gsunwu@163.com
# @File    : test_models.py
# @Description:

import unittest

from ConfigSpace import Configuration
import logging

from pylibs.smac3.search_pace import objective_function

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        cfg = {
            'classifier': 'hbos',
            'hbos_alpha': 0.1780238738283515,
            'hbos_n_bins': 32,
            'hbos_tol': 0.6765088625252247,
        }
        other = {'is_auto_data_scaling': True, 'model_name': 'hbos', 'dataset_name': 'NASA-SMAP',
                 'data_id': 'T-2.test.out'}
        objective_function(cfg, **other)
