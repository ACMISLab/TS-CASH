import time
import unittest

import numpy as np

from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.experiments.exp_config import ExpConf, _cal_metrics_from_id
from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_log import get_logger

log = get_logger()


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        conf = np.load(
            "/Users/sunwu/SW-Research/runtime/exp_name/model_name1691919077.297033/dataset_name1691919077.297036/metrics/ae9104b26bc684dda81034d5d6b2cfe2ccd4d5bb_score.npz",
            allow_pickle=True)
        print(conf)

    def test_metrics(self):
        score = "/Users/sunwu/Downloads/download_metrics/V472_observation_v6_VUS_ROC_0.001_random/lstm/Daphnet/metrics/0f59a954e1dbb5764eb9d0b14a7da7ab70342203_score.npz"
        metrics = ExpConf.get_metrics_from_npz(score)
        print(metrics)

    def test_save_to_mysql(self):
        conf = ExpConf(model_name="as", dataset_name="IOPS",
                       data_id="KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd.train.out",
                       data_sample_method="random",
                       data_sample_rate=time.time(),
                       exp_index=1, exp_total=10,
                       exp_name="djls",
                       metrics_save_home=UtilComm.get_system_runtime(),
                       test_rate=time.time_ns()
                       )
        dth = DateTimeHelper()
        conf.save_score_dth_exp_conf_to_redis(score=np.random.normal(size=(3096,)), dt=dth)

    def test_cal_score(self):
        conf = ExpConf(model_name="as", dataset_name="IOPS",
                       data_id="KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd.train.out",
                       data_sample_method="random",
                       data_sample_rate=time.time(),
                       exp_index=1, exp_total=10,
                       exp_name="djls",
                       metrics_save_home=UtilComm.get_system_runtime(),
                       test_rate=time.time_ns()
                       )
        npz_file = conf.save_score_dth_exp_conf(None, None)
        ExpConf.get_metrics_from_npz(npz_file)

    def test_anomaly_rate(self):
        conf = ExpConf(model_name="as", dataset_name="IOPS",
                       data_id="KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd.train.out",
                       data_sample_method="random",
                       data_sample_rate=time.time(),
                       exp_index=1, exp_total=10,
                       exp_name="djls",
                       metrics_save_home=UtilComm.get_system_runtime(),
                       test_rate=time.time_ns()
                       )
        print(conf.get_anomaly_rate_of_windows())
