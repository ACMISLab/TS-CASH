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

from pylibs.uts_models.benchmark_models.lstm_ed.lstm_ed import LSTMED, LSTMEDConfig
from pylibs.uts_models.benchmark_models.vae_ed.vae_model import VAEEDConfig, VAEED

sys.path.append("../../../../timeseries-models")
sys.path.append("../../../../py-search-lib")
sys.path.append("../../../../datasets")
from sklearn.preprocessing import MinMaxScaler
from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.experiments.exp_helper import JobConfV1

from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.utils.util_joblib import JLUtil

window_size = 64
_dataset, _data_id = ["ECG", "MBA_ECG805_data.out"]
from pylibs.utils.util_log import get_logger

log = get_logger()
mem = JLUtil.get_memory()

from pylibs.experiments.exp_config import ExpConf

outs = []

# conf = ExpConf(dataset_name="ECG",data_id="MBA_ECG14046_data.out",data_sample_rate=_sr,fold_index=1)
# conf = ExpConf(model_name="VAE",dataset_name="ECG",data_id="MBA_ECG14046_data_46.out",data_sample_rate=_sr,fold_index=1)
_sr = 10
conf = ExpConf(model_name="VAE", dataset_name="SVDB", data_id="801.test.csv@1.out", data_sample_rate=_sr, fold_index=1)
train_x, train_y, test_x, test_y = conf.load_dataset_at_fold_k()

test_x = test_x[:1]
test_y = test_y[:1]

cuda = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda: {cuda}")
conf = VAEEDConfig()
conf.num_epochs = 50
conf.latent_size = 5
conf.batch_size = 1
conf.sequence_len = window_size
model = VAEED(conf)
model.fit(train_x)
score = model.score(test_x)

# Post-processing
score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=window_size, metric_type="vus")
print(metrics[EK.VUS_ROC])
outs.append([_sr, metrics[EK.VUS_ROC]])
