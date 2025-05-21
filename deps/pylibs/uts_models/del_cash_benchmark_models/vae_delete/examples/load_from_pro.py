# todo: this code may be error.
import os

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.vae.vae import VAEModel
from pylibs.uts_models.benchmark_models.vae.vae_conf import VAEConfig
from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_keras import UtilKeras
from pylibs.uts_dataset.dataset_loader import DatasetLoader, DataDEMOKPI
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

window_size = 99
max_length = 1000
dataset, data_id = "ECG", "MBA_ECG805_data.out"
# 获取训练数据，只有异常

dl_train = DatasetLoader(dataset, data_id, window_size=window_size,
                         is_include_anomaly_window=False,
                         max_length=max_length)
train_x, train_y = dl_train.get_sliding_windows()

# 获取测试数据，包含异常和正常
dl_test = DatasetLoader(dataset, data_id, window_size=window_size,
                        is_include_anomaly_window=True,
                        max_length=max_length)
test_x, test_y = dl_test.get_sliding_windows()
conf = VAEConfig()
conf.window_size = window_size
conf.epochs = 1

# load
new_vae = VAEModel(conf).load_model(UtilComm.get_file_name(
    "/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/runtime/deep_vae_v1/vae/ECG/models_checkpoints/f8d685829f0d472cf492fd4826b6a6e7cad7aa76"))
new_score = new_vae.score(test_x)

print(new_score.sum())
