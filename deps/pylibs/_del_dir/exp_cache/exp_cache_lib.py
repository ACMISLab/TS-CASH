#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/15 09:23
# @Author  : gsunwu@163.com
# @File    : exp_cache_lib.py
# @Description:
import json
import warnings

import pandas as pd
from tqdm import tqdm

from pylibs.config import GCFV3
from pylibs.experiments.exp_config import ExpConf
from pylibs.utils.util_base64 import UtilBase64
from pylibs.utils.util_file import FileUtils
from pylibs.utils.util_redis import RedisUtil
from pylibs.utils.util_log import get_logger
from typing import Union

log = get_logger()


class ExpCache:
    REDIS_CONN = None

    def __init__(self,
                 home="/Users/sunwu/Downloads/download_metrics/v4020_03_fastuts_sup1_vus_roc_0.5_random/vae/YAHOO/metrics",
                 redis_conf=GCFV3.get_server_conf("redis_111", "lan")):
        self.home = home
        self.redis_conf = redis_conf

    def _load_files(self) -> list:
        uf = FileUtils()
        files = uf.get_all_files(self.home, ext="metrics.bz2")
        return files

    def _get_redis_connection(self) -> RedisUtil:
        if self.REDIS_CONN is None:
            self.REDIS_CONN = RedisUtil(self.redis_conf)

        return self.REDIS_CONN

    def save_to_redis(self, home=None) -> None:
        if home is not None:
            self.home = home
        for _f in tqdm(self._load_files()):
            _df = pd.read_pickle(_f)
            if _df.shape[1] == 0:
                continue
            metrics = pd.Series(data=_df.iloc[:, 1].values, index=_df.iloc[:, 0].values)
            r_k = ExpConf.get_series_values_id(metrics)
            if self.conn.exist(r_k):
                log.warning(f"Key already exists...")
            else:
                r_val = metrics.to_json()
                self.conn.set(r_k, r_val)

    @property
    def conn(self) -> RedisUtil:
        return self._get_redis_connection()

    def is_exist(self, conf: ExpConf) -> bool:
        return self.conn.exist(conf.get_redis_key())

    def get(self, conf: ExpConf) -> Union[dict, None]:
        if not self.is_exist(conf):
            return None
        else:
            return json.loads(self.conn.get(conf.get_redis_key()))


if __name__ == '__main__':
    ec = ExpCache()
    ec.save_to_redis()
    debug_data = {
        "data_processing_time": 0.00249891,
        "window_size": 64,
        "model_name": "hbos",
        "data_sample_method": "random",
        "data_sample_rate": 512,
        "dataset_name": "IOPS",
        "data_id": "KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.train.out",
        "exp_total": 460,
        "exp_index": 149,
        "exp_name": "v4020_03_fastuts_sup1_vus_roc_0.001_random",
        "anomaly_window_type": "all",
        "metrics_save_home": "/remote-home/sunwu/cs_acmis_sunwu/runtime",
        "test_rate": 0.3,
        "seed": 0,
        "job_id": "8fe5a1d1887bbefabafc3a090592b26df449a37a",
        "kfold": 5,
        "is_send_message_to_feishu": True,
        "gpu_index": 33,
        "fold_index": 3,
        "stop_alpha": 0.001,
        "epoch": 50,
        "gpu_memory_limit": 1024,
        "batch_size": 128,
        "data_scale_beta": None,
        "verbose": 0,
        "optimal_target": "VUS_ROC",
        "time_app_start": 1694619445009106895,
        "time_app_end": -1,
        "time_train_start": 1694619445144130173,
        "time_train_end": 1694619445172698552,
        "time_eval_start": 1694619445172730476,
        "time_eval_end": 1694619445176729940,
        "elapsed_train": 0.03,
        "elapsed_between_train_start_and_eval_end": 0.032599767,
        "train_len": 512,
        "test_len": 1744,
        "Precision": 0,
        "Recall": 0,
        "F": 0,
        "AUC_ROC": 0.7043332611,
        "AUC_PR": 0.0591761621,
        "Precision_at_k": 0,
        "Rprecision": 0,
        "Rrecall": 0,
        "RF": 0,
        "R_AUC_ROC": 0.6213405211,
        "R_AUC_PR": 0.064528548,
        "VUS_ROC": 0.6347484608,
        "VUS_PR": 0.0659640888,
        "debug": False
    }
    data = ExpConf(**debug_data)
    # True
    print(ec.is_exist(data))

    # False
    print(ec.is_exist(ExpConf()))
