#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/8 19:10
# @Author  : gsunwu@163.com
# @File    : tests.py
# @Description:
import unittest

import logging

import pandas as pd

from pylibs.gunplot.histogram import Histogram

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        df = pd.read_csv("/Users/sunwu/SW-Research/AutoML/runs/automl/process_data.csv", sep="")
        hg = Histogram()
        # dataset_name	data_id	facade	VUS_ROC_mean_disable_autoscale	VUS_ROC_mean_enable_autoscale	used_target_function_walltime_mean_disable_autoscale	used_target_function_walltime_mean_enable_autoscale	VUS_ROC_std_disable_autoscale	VUS_ROC_std_enable_autoscale	used_target_function_walltime_std_disable_autoscale	used_target_function_walltime_std_enable_autoscale
        df = df[df['facade'] == "hyperparameter_optimization"]
        # df = df[df['VUS_ROC_mean_disable_autoscale'] >= 0]
        hg.two_comp(df.loc[:, ["data_id",
                               "VUS_ROC_mean_disable_autoscale",
                               "VUS_ROC_std_disable_autoscale",
                               "VUS_ROC_mean_enable_autoscale",
                               "VUS_ROC_std_disable_autoscale"
                               ]])
#         hyperparameter_optimization
