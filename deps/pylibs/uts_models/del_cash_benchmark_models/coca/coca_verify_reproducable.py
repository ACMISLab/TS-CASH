"""
I do not change hyperparameters and configs except the printing and some unrelated statements.

"""

import random
import sys

import numpy
import torch

from pylibs.uts_models.benchmark_models.coca.coca_config import COCAConf
from pylibs.uts_models.benchmark_models.coca.coca_model import COCAModel

sys.path.append("./")
from merlion.utils import TimeSeries

from dataloader.dataset import data_generator1, data_generator4
from pylibs.utils.util_pytorch import summary_dataset
from ts_datasets.anomaly import IOpsCompetition


def enable_numpy_reproduce(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)


def enable_pytorch_reproduciable(seed=0):
    """
    For reproducible results.

    Parameters
    ----------
    seed :

    Returns
    -------

    """
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


enable_numpy_reproduce(1)
enable_pytorch_reproduciable(1)
conf = COCAConf()
conf.batch_size = 512
idx = -3
dt = IOpsCompetition()

time_series, meta_data = dt[idx]
train_data = TimeSeries.from_pd(time_series[meta_data.trainval])
test_data = TimeSeries.from_pd(time_series[~meta_data.trainval])
train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator4(train_data, test_data, train_labels, test_labels,
                                                                     conf)
summary_dataset(train_dl)
summary_dataset(val_dl)
summary_dataset(test_dl)
coca = COCAModel(conf)
coca.fit(train_dl)
score = coca.score_coca(val_dl)
metrics = coca.report_metrics_coca(val_dl, test_dl)
# Assure the metrics is absolutely equal to COCA source: https://github.com/ruiking04/COCA
assert metrics["valid_affiliation_precision"] == 0.9882697947214076
assert metrics["valid_affiliation_recall"] == 0.49824046920821113
assert metrics["test_affiliation_recall"] == 0.41551757819907786
assert metrics["test_affiliation_precision"] == 0.6605439430607345

coca.report_metrics(val_dl, test_dl)
print("✅ Passed")
