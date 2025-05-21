#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/2/27 13:40
# @Author  : gsunwu@163.com
# @File    : main_dask.py
# @Description:
from cash_benchmark_models.pyod.models.auto_encoder import AutoEncoder
from pylibs.utils.util_gnuplot import UTSViewGnuplot
from pylibs.utils.util_ust_test import UtilUTSTest

train_x, train_y, test_x, test_y = UtilUTSTest.load_csv()
model = AutoEncoder(batch_size=128, epochs=20, dropout_rate=0, hidden_neurons=[128, 64, 2, 2, 64, 128])
model.fit(train_x)
score = model.score(train_x)
ug = UTSViewGnuplot()
ug.plot_uts_data_v3(train_x, train_y, score)
