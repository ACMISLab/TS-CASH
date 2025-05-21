import os

from pylibs.evaluation.metric_process import MetricHelper
from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.experiments.exp_config import ExpConf
from pylibs.experiments.exp_helper import JobConfV1
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.utils.util_merlion import UtilMerlion

from merlion.models.forecast.lstm import LSTM

from merlion.models.anomaly.vae import VAE, VAEConfig

from merlion.models.anomaly.dagmm import DAGMM, DAGMMConfig

from merlion.utils import UnivariateTimeSeries

from merlion.models.anomaly.isolation_forest import IsolationForestConfig, IsolationForest
import pandas as pd
from merlion.utils import TimeSeries
from ts_datasets.anomaly import NAB

from pylibs.uts_models.benchmark_models.lstm_ed.lstm_ed import LSTMED, LSTMEDConfig

for _sr in JobConfV1.SAMPLE_RATES_DEBUGS:
    conf = ExpConf(dataset_name="ECG", data_id="MBA_ECG14046_data_46.out", data_sample_rate=_sr, fold_index=1)
    conf.device = "mps"
    train_x, train_y, test_x, test_y = conf.load_dataset_at_fold_k()

    window_size = 64
    # model=DAGMM(DAGMMConfig(sequence_len=window_size))
    # model=VAE(VAEConfig(sequence_len=window_size))
    conf = LSTMEDConfig(sequence_len=window_size)
    conf.device = "cpu"

    model = LSTMED(conf)

    model.fit(X=train_x)

    score = model.score(test_x)
    metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=window_size, metric_type="vus")

    print(metrics[EK.VUS_ROC])
#
# from merlion.plot import plot_anoms
# import matplotlib.pyplot as plt
# fig, ax = model.plot_anomaly(time_series=test_data)
# plot_anoms(ax=ax, anomaly_labels=test_labels)
# plt.show()
#
#
# from merlion.evaluate.anomaly import TSADMetric
# p = TSADMetric.Precision.value(ground_truth=test_labels, predict=test_pred)
# r = TSADMetric.Recall.value(ground_truth=test_labels, predict=test_pred)
# f1 = TSADMetric.F1.value(ground_truth=test_labels, predict=test_pred)
# mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=test_pred)
# print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}\n"
#       f"Mean Time To Detect: {mttd}")
