#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/12 14:58
# @Author  : gsunwu@163.com
# @File    : example_helper.py
# @Description:
import os.path

import pandas as pd

from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.experiments.exp_config import ExpConf
from pylibs.experiments.exp_helper import JobConfV1, _train_model_kfold_v2
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.utils.util_common import UC
from pylibs.utils.util_pandas import PDUtil
from pylibs.utils.util_torch import is_mps_available
from pylibs.utils.util_gnuplot import UTSViewGnuplot, Gnuplot
from pylibs.exp_ana_common.ExcelOutKeys import EK


class ExampleHelper:
    EPOCH = 50
    WINDOW_SIZE = 64
    BATCH_SIZE = 128

    @staticmethod
    def get_exp_conf_ecg(sample_rate=-1) -> ExpConf:
        #  dataset_name="SVDB", data_id="801.test.csv@1.out"
        conf = ExpConf(dataset_name="SVDB",
                       data_id="801.test.csv@1.out", data_sample_rate=sample_rate,
                       fold_index=1)
        return conf

    @staticmethod
    def observation_model(model, sample_rates=None):
        if sample_rates is None:
            sample_rates = JobConfV1.SAMPLE_RATES_DEBUGS
        outs = []
        for _sr in sample_rates:
            dt = DateTimeHelper()

            conf = ExampleHelper.get_exp_conf_ecg(_sr)

            dirty_train_sample_x, dirty_train_sample_y, clean_train_sample_x, clean_train_sample_y, \
                test_x, test_y = conf.load_data_for_lstm_and_cnn()

            cuda = "mps" if is_mps_available() else "cpu"
            print(f"Cuda: {cuda}")
            dt.train_start()
            model.fit(clean_train_sample_x)
            dt.train_end()
            score = model.score(test_x)
            from sklearn.preprocessing import MinMaxScaler
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

            from pylibs.utils.util_gnuplot import UTSViewGnuplot
            ugp = UTSViewGnuplot()
            ugp.plot_uts_data_v3(test_x[:, -1], test_y, score, file_name=f"{model.name}_{_sr}")
            metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=ExampleHelper.WINDOW_SIZE,
                                                      metric_type="vus")
            from pylibs.exp_ana_common.ExcelOutKeys import EK
            outs.append([_sr, metrics[EK.VUS_ROC], dt.get_elapse_train()])

        pdf = pd.DataFrame(outs)
        pdf.columns = [EK.DATA_SAMPLE_RATE, EK.VUS_ROC, EK.ELAPSED_TRAIN]
        PDUtil.save_to_excel(pdf, "res", home=UC.get_entrance_directory())

    @staticmethod
    def observation_model_by_conf(conf: ExpConf, sample_rates=JobConfV1.SAMPLE_RATES_DEBUGS):
        save_home = os.path.join(UC.get_entrance_directory(), "output", conf.model_name)
        outs = []
        print(f"Loading model {conf.model_name}")
        for _sr in sample_rates:
            conf.data_sample_rate = _sr
            train_x, train_y, test_x, test_y = conf.load_csv()

            cuda = "mps" if is_mps_available() else "cpu"
            print(f"Cuda: {cuda}")

            score, dth = _train_model_kfold_v2(conf.encode_key_params_bs64())

            from sklearn.preprocessing import MinMaxScaler
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

            ugp = UTSViewGnuplot(save_home)
            ugp.plot_uts_data_v3(test_x[:, -1], test_y, score, file_name=f"{conf.model_name}_{_sr}")
            metrics = UTSMetricHelper.get_metrics_all(test_y,
                                                      score,
                                                      window_size=conf.window_size,
                                                      metric_type="vus")
            outs.append([_sr, metrics[EK.VUS_ROC]])

        pdf = pd.DataFrame(outs)
        pdf.columns = ["sr", EK.VUS_ROC]
        gp = Gnuplot(home=save_home)
        gp.set_output_jpg(conf.model_name + "_training_samples_vs_sr")
        gp.add_data(pdf)
        gp.set(f'set title "{conf.model_name}"')
        gp.set('set xlabel "Training samples"')
        gp.set('set ylabel "VUS ROC"')
        gp.plot("plot $df using 0:2:xticlabels(1) with lp")
        gp.show()
        # PDUtil.save_to_excel(pdf, conf.model_name + "_training_samples_vs_sr", home=save_home)
