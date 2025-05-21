#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/22 09:04
# @Author  : gsunwu@163.com
# @File    : test_black_function.py
# @Description:
import unittest
import logging

import numpy as np

from pylibs.hpo.black_box_function import black_function_1

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        from matplotlib import pyplot as plt
        np.random.seed(42)
        xs = np.linspace(-2, 10, 10000)
        plt.plot(xs, black_function_1(xs))
        plt.show()
