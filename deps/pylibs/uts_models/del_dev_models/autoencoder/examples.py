import numpy as np
import pandas as pd

from dev_models.autoencoder.model import AutoEn
from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.experiments.exp_config import ExpConf
from pylibs.experiments.exp_helper import JobConfV1
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.utils.util_common import UC
from pylibs.utils.util_pandas import PDUtil
from pylibs.uts_dataset.dataset_loader import KFoldDatasetLoader

window_size = 64

from pylibs.utils.util_joblib import cache_

target_metrics = []
for sr in JobConfV1.SAMPLE_RATES_OBSERVATION_V4[:20:2]:
    print(f"++++++++++{sr}++++++++++++++++")
    ec = ExpConf(data_sample_rate=sr, kfold=2, fold_index=1)
    train_x, train_y, test_x, test_y = ec.load_dataset_at_fold_k()
    model = AutoEn()
    model.fit(train_x)
    pred = model.autoencoder.predict(test_x)
    score = np.mean(np.abs(pred - test_x), axis=1)
    metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=window_size, metric_type="vus")
    vr = metrics[EK.VUS_ROC]
    target_metrics.append([sr, vr])

PDUtil.save_to_excel(pd.DataFrame(target_metrics), "relation_acc_and_sr", home=UC.get_entrance_directory())
