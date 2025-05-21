#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/23 11:55
# @Author  : gsunwu@163.com
# @File    : example.py
# @Description:
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_metrics.vus.uts_metric_helper import UTSMetricHelper, UTSMetrics
from pylibs.uts_dataset.dataset_loader import DatasetLoader, DataDEMOKPI
from pylibs.uts_models.benchmark_models.xgboost.uts_xgboost import XGBoost

window_size = 99
for dataset, data_id in DataDEMOKPI.DEMO_KPIS:
    dl = DatasetLoader(dataset, data_id, window_size=window_size, max_length=10000)
    train_x, train_y = dl.get_sliding_windows()

    modelName = 'IForest'
    clf = XGBoost()
    clf.fit(train_x,train_y)
    score = clf.score(train_x)

    # Post-processing
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

    uv = UnivariateTimeSeriesView(name=data_id, is_save_fig=True)
    # uv.plot_x_label_score_row2(train_x[:, -1], train_y, score)

    metrics = UTSMetricHelper.get_metrics_all(score=score,labels=train_y, window_size=window_size)
    uv.plot_x_label_score_metrics_row2(train_x[:, -1], train_y, score, metrics)
    print(metrics)
