#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/15 07:00
# @Author  : gsunwu@163.com
# @File    : __init__.py.py
# @Description:
from dataclasses import dataclass


@dataclass
class UTSExpConf:
    model_name: str

    def is_semi_supervised_model(self):
        return self.model_name in ['ocsvm', 'ae', 'vae', 'lstm-ad', 'cnn']


def load_model(conf: UTSExpConf):
    if conf.model_name == "hbos":
        assert conf.is_semi_supervised_model() == False
        from pylibs.uts_models.benchmark_models.tsbuad.models.hbos import HBOS
        clf = HBOS()
    elif conf.model_name == "iforest":
        assert conf.is_semi_supervised_model() == False
        from pylibs.uts_models.benchmark_models.iforest.iforest import IForest
        clf = IForest()
    elif conf.model_name == "lof":
        assert conf.is_semi_supervised_model() == False
        from pylibs.uts_models.benchmark_models.lof.lof import LOF
        clf = LOF()
    elif conf.model_name == "ocsvm":
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.ocsvm.ocsvm import OCSVM
        clf = OCSVM()
    elif conf.model_name == "ae":
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.pyod.models.auto_encoder import AutoEncoder
        clf = AutoEncoder(batch_size=conf.batch_size, epochs=conf.epoch)
    elif conf.model_name == "vae":
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.pyod.models.vae import VAE
        clf = VAE(epochs=conf.epoch,
                  batch_size=conf.batch_size,
                  verbose=conf.verbose)
    elif conf.model_name == "lstm-ad":
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.tsbuad.models.lstm import lstm
        clf = lstm(slidingwindow=conf.window_size,
                   epochs=conf.window_size,
                   verbose=conf.verbose,
                   batch_size=conf.batch_size)
    elif conf.model_name == 'cnn':
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.tsbuad.models.cnn import cnn
        clf = cnn(slidingwindow=conf.window_size, epochs=conf.epoch,
                  batch_size=conf.batch_size)

    elif conf.model_name == "dagmm":
        from pylibs.uts_models.benchmark_models.dagmm.dagmm_model import DAGMMConfig, DAGMM
        cf = DAGMMConfig()
        cf.sequence_len = conf.window_size
        cf.batch_size = conf.batch_size
        cf.num_epochs = conf.epoch
        cf.device = conf.device
        clf = DAGMM(cf)
    elif conf.model_name == "coca":
        from pylibs.uts_models.benchmark_models.coca.coca_config import COCAConf
        from pylibs.uts_models.benchmark_models.coca.coca_model import COCAModel
        cf = COCAConf()
        cf.window_size = conf.window_size
        cf.batch_size = conf.batch_size
        cf.num_epoch = conf.epoch
        cf.device = conf.device
        clf = COCAModel(cf)
    elif conf.model_name == "tadgan":
        from pylibs.uts_models.benchmark_models.tadgan.tadgan_model import TadGanEd, TadGanEdConf
        cf = TadGanEdConf()
        cf.signal_shape = conf.window_size
        cf.batch_size = conf.batch_size
        cf.num_epochs = conf.epoch
        cf.device = conf.device
        clf = TadGanEd(cf)

    else:
        raise RuntimeError(f"Unknown model {conf.model_name}")
    return clf
