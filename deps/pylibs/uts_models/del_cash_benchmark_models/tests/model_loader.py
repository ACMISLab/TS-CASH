#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/9/12 16:09
# @Author  : gsunwu@163.com
# @File    : model_loader.py
# @Description:


from tqdm import tqdm

from pylibs.experiments.example_helper import ExampleHelper
from pylibs.experiments.exp_config import ExpConf, ModelUtils

for _model_name in tqdm(ModelUtils.MODELS):
    _c = ExpConf(model_name=_model_name, dataset_name="SVDB", data_id="801.test.csv@1.out", fold_index=1)
    _c.load_model()
    ExampleHelper.observation_model_by_conf(_c, sample_rates=[2, 8, 32, 512, 1024, 2048])
