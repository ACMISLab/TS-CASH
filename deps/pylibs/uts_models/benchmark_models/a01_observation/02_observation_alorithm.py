#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/11 10:54
# @Author  : gsunwu@163.com
# @File    : observation.py
# @Description:
import sys

from pylibs.uts_models.benchmark_models.a01_observation.util_observation import ObservationsUtil
from pylibs.experiments.exp_config import ExpConf
from pylibs.experiments.exp_helper import load_model
from pylibs.utils.util_gnuplot import UTSViewGnuplot
from pylibs.utils.util_joblib import cache_

window_size = 100
epoch = 50

train_x, true_label, test_x, test_y, _ = ObservationsUtil.load_observation_data_splits(original_labels=True,
                                                                                       window_size=window_size)
_, train_y, _, _, _ = ObservationsUtil.load_observation_data_splits(original_labels=False)


@cache_
def _train(ef, train_x, train_y):
    model = load_model(ef)
    model.fit(train_x, train_y)
    score = model.score(train_x)
    return score


for _model_name in ['lof', 'hbos', 'ocsvm', 'iforest', 'ae', 'vae', 'lstm', 'cnn', 'svm']:
    ef = ExpConf(model_name=_model_name, window_size=window_size, epoch=epoch)
    score = _train(ef, train_x, train_y)
    gp = UTSViewGnuplot()
    gp.plot_uts_data_v2(train_x[:, -1], true_label, score, file_name=_model_name, h=6)
