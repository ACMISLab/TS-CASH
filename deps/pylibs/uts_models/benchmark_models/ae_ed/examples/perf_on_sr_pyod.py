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

from sklearn.preprocessing import MinMaxScaler

from pylibs._del_dir.experiments.exp_config import ExpConf
from pylibs._del_dir.experiments.exp_helper import JobConfV1
from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.utils.util_common import UC
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_pandas import PDUtil

from pylibs.utils.util_joblib import JLUtil
from pylibs.uts_metrics.vus.uts_metric_helper import UTSMetricHelper

window_size = 64
_dataset, _data_id = ["ECG", "MBA_ECG805_data.out"]
from pylibs.utils.util_log import get_logger

log = get_logger()
mem = JLUtil.get_memory()


outs = []

enable_numpy_reproduce()
for _sr in JobConfV1.SAMPLE_RATES_DEBUGS:
    # conf = ExpConf(dataset_name="ECG",data_id="MBA_ECG14046_data.out",data_sample_rate=_sr,fold_index=1)
    # conf = ExpConf(dataset_name="ECG",model_name="AE",data_id="MBA_ECG14046_data_46.out",data_sample_rate=_sr,fold_index=1)
    conf = ExpConf(model_name="VAE", dataset_name="SVDB", data_id="801.test.csv@1.out", data_sample_rate=_sr,
                   fold_index=1, anomaly_window_type='all')
    train_x, train_y, test_x, test_y = conf.load_data()

    cuda = "mps"
    # cuda = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cuda: {cuda}")

    from pylibs.uts_models.benchmark_models.pyod.models.auto_encoder import AutoEncoder

    model = AutoEncoder(batch_size=128, epochs=50)
    model.fit(train_x)
    score = model.score(test_x)
    # Post-processing
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=window_size, metric_type="vus")
    print(metrics[EK.VUS_ROC])
    outs.append([_sr, metrics[EK.VUS_ROC]])

pdf = pd.DataFrame(outs)
pdf.columns = ["sr", EK.VUS_ROC]
PDUtil.save_to_excel(pdf, "res", home=UC.get_entrance_directory())
