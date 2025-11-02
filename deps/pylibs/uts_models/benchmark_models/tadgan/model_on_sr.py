from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.experiments.exp_config import ExpConf
from pylibs.experiments.exp_helper import JobConfV1
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.utils.util_array import ArrSaver

from pylibs.uts_models.benchmark_models.tadgan.tadgan_model import TadGanEd, TadGanEdConf

ar = ArrSaver()
for _sr in JobConfV1.SAMPLE_RATES_DEBUGS:
    # conf = ExpConf(dataset_name="ECG",data_id="MBA_ECG14046_data.out",data_sample_rate=_sr,fold_index=1)
    conf = ExpConf(dataset_name="ECG", data_id="MBA_ECG14046_data_46.out", data_sample_rate=_sr, fold_index=1)
    # conf = ExpConf( data_sample_rate=_sr, fold_index=1)
    train_x, train_y, test_x, test_y = conf.load_dataset_at_fold_k()

    model_conf = TadGanEdConf()
    model_conf.signal_shape = conf.window_size
    model_conf.batch_size = 128
    model_conf.num_epochs = 50
    tg = TadGanEd(model_conf)
    tg.fit(train_x)

    score = tg.score(test_x)
    metric = UTSMetricHelper.get_metrics_all(test_y, score, window_size=conf.window_size)
    ar.append(_sr, metric[EK.VUS_ROC])

ar.save_to_excel()
