#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/10/4 19:30
# @Author  : gsunwu@163.com
# @File    : hpo_helper.py
# @Description:
import sys

import nni

from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.experiments.exp_config import ExpConf
from pylibs.utils.util_log import get_logger

log = get_logger()


class HPOKeys:
    EXPERIMENT_WORKING_DIRECTORY = "experiment_working_directory"
    DATA_SAMPLE_RATE = "data_sample_rate"
    DATA_SAMPLE_METHOD = "data_sample_method"
    DATASET_NAME = "dataset_name"
    DATA_ID = "data_id"

    def __init__(self):
        pass


class CommanderGenerater():

    def __init__(self):
        pass


class NNITrainerHelper:
    def __init__(self, cf: ExpConf):
        self.conf = cf
        self.model = None
        self._load_model()

    def _get_model_parameters(self):
        return nni.get_next_parameter()

    def _load_model(self):
        params = self._get_model_parameters()
        log.info(f"nni parameters: {params}")
        if self.conf.model_name == "iforest":
            from pylibs.uts_models.benchmark_models.iforest.iforest import IForest
            cf = IForest(**params)
            self.model = cf
        else:
            raise ValueError("Unexpected model name: %s" % self.conf.model_name)

        log.info(f"model parameters: {self.model.__dict__}")

        return self.model

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def score(self, test_x):
        return self.model.score(test_x)

    def report_results(self):
        results = {}
        nni.report_final_result(results)

    def run(self, ):
        dth = DateTimeHelper()
        dth.start_or_restart()
        train_x, train_y, test_x, test_y = self.conf.load_csv()
        dth.train_start()
        self.train(train_x, train_y)
        dth.train_end()

        dth.evaluate_start()
        score = self.score(test_x)
        dth.evaluate_end()

        metrics = self.conf.calculate_metrcs_by_conf_score_dth(self.conf, score, dth)
        metrics['default'] = metrics[EK.VUS_ROC]
        nni.report_final_result(metrics)
