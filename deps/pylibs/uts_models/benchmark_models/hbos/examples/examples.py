#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/12 17:00
# @Author  : gsunwu@163.com
# @File    : examples.py
# @Description:
from pylibs._del_dir.experiments.example_helper import ExampleHelper
from pylibs._del_dir.experiments.exp_config import ExpConf

_c = ExpConf(model_name="hbos", dataset_name="SVDB", data_id="801.test.csv@1.out", fold_index=1, data_sample_rate=2)
model = _c.load_model()

from pylibs.uts_models.benchmark_models.tsbuad.models.hbos import HBOS
hbos = HBOS()
ExampleHelper.observation_model(hbos)
