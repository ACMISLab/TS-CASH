#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/9 13:59
# @Author  : gsunwu@163.com
# @File    : test_mysql.py
# @Description:
import copy
import traceback
import unittest
import logging

import rich

from exps.search_pace import ModelResultCache, ModelResultCacheV2
from pylibs.utils.util_mysql import UtilMysql

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        d = UtilMysql(database="p2_automl_model_cache_v6_2080ti")
        results = d.exec("select * from default_table limit 1;")
        print(results)
        for r in results:
            _cache = ModelResultCache(**r)

            config_ids = _cache.config_id.split(";")

            _tmp_r=copy.deepcopy(r)
            del _tmp_r['config_id']
            del _tmp_r['function_loss']
            del _tmp_r['function_parameter']
            _new_cache = ModelResultCacheV2(**_tmp_r)
            for _id in config_ids:
                # data_id=NAB_data_KnownCause_3.out;dataset_name=NAB;is_auto_data_scaling=False;is_debug=False;seed=0;test_rate=0.4;
                if _id=="":
                    continue
                try:
                    key, val = _id.split("=")
                    if key == "data_id":
                        _new_cache.data_id = val
                    elif key == "dataset_name":
                        _new_cache.dataset_name = val
                    elif key == "is_debug":
                        _new_cache.is_debug = val
                    elif key == "seed":
                        _new_cache.seed = int(val)
                    elif key == "test_rate":
                        _new_cache.test_rate = float(val)
                    else:
                        if key not in ["is_auto_data_scaling"]:
                            raise ValueError(f"Unknown key {key}")
                except:
                    traceback.print_exc()
                # 设置默认值
                _new_cache.data_sample_method=""
                _new_cache.data_sample_rate=-1
                _new_cache.save()

            rich.print(_cache)
            rich.print(_new_cache)
