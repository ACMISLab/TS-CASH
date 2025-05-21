#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/11 10:45
# @Author  : gsunwu@163.com
# @File    : observation_algo.py
# @Description:
from pylibs.uts_models.benchmark_models.a01_observation.util_observation import ObservationsUtil
from pylibs.utils.util_gnuplot import UTSViewGnuplot

values, labels = ObservationsUtil.load_observation_data_origin()

gp = UTSViewGnuplot()
gp.plot_uts_data_without_score(values, labels, w=30, h=2, fname="original")
