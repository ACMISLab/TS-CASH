#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/12 19:34
# @Author  : gsunwu@163.com
# @File    : example_tf.py
# @Description:


# !/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/12 09:37
# @Author  : gsunwu@163.com
# @File    : example_pyod.py
# @Description:
"""
exp_index	sr	VUS_ROC
0	8	0.616815804
1	16	0.578078863
2	32	0.60975608
3	64	0.725382617
4	128	0.738366176
5	256	0.70718911
6	512	0.814988886
7	1024	0.812873488


"""

import sys

import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.dagmm.dagmm_master_tf2.dagmm import DAGMM
from pylibs.uts_models.benchmark_models.pyod.models.vae import VAE
from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.experiments.exp_helper import JobConfV1
from pylibs.utils.util_common import UC
from pylibs.utils.util_gnuplot import UTSViewGnuplot
from pylibs.utils.util_pandas import PDUtil

from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.utils.util_joblib import JLUtil

window_size = 64
_dataset, _data_id = ["ECG", "MBA_ECG805_data.out"]
from pylibs.utils.util_log import get_logger

log = get_logger()
mem = JLUtil.get_memory()

from pylibs.experiments.exp_config import ExpConf

outs = []

for _sr in ["2", "4", "8", "16", "32", "64", "128"]:
    # conf = ExpConf(dataset_name="ECG",data_id="MBA_ECG14046_data.out",data_sample_rate=_sr,fold_index=1)
    # conf = ExpConf(model_name="VAE",dataset_name="ECG",data_id="MBA_ECG14046_data_46.out",data_sample_rate=_sr,fold_index=1)
    conf = ExpConf(model_name="VAE", dataset_name="SVDB", data_id="801.test.csv@1.out", data_sample_rate=_sr,
                   fold_index=1)
    train_x, train_y, test_x, test_y = conf.load_csv()

    cuda = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cuda: {cuda}")

    # Initialize
    model = DAGMM(xdim=1)
    model.fit(train_x)
    score = model.predict(test_x)
    # Post-processing

    ugp = UTSViewGnuplot()
    ugp.plot_uts_data_v3(test_x[:, -1], test_y, score, file_name=f"dagmm_tf_{_sr}")
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=window_size, metric_type="vus")
    print(metrics)
    outs.append([_sr, metrics[EK.VUS_ROC]])

pdf = pd.DataFrame(outs)
pdf.columns = ["sr", EK.VUS_ROC]
PDUtil.save_to_excel(pdf, "res", home=UC.get_entrance_directory())

#
#
#
# # Fit the training data to model
# model.fit(x_train)
#
# # Evaluate energies
# # (the more the energy is, the more it is anomary)
# energy = model.predict(x_test)
