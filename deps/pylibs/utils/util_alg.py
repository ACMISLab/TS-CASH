#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/2/16 08:47
# @Author  : gsunwu@163.com
# @File    : util_alg.py
# @Description:

def is_semi_supervised_model(model_name):
    """
    如果是半监督，只能包含正常数据。

    监督和非监督按正常流程处理。监督包含label，非监督虽然包含label，但是没有用label。

    Returns
    -------

    """
    # fully unsupervised: IForest, IForest1, LOF, MP, NormA, PCA, HBOS, and POLY.
    # semi-supervised: OCSVM, AE, LSTM-AD, and CNN.
    #
    # ref: Theseus: Navigating the Labyrinth of Time-Series Anomaly Detection
    # We select 12 different AD methods, summarized in Table 1. Out of these, 8 are fully
    # unsupervised (i.e., they require no prior information on the anomalies to be detected): IForest, IForest1,
    # LOF, MP, NormA, PCA, HBOS, and POLY. The remaining 4 methods are semi-supervised (i.e., they require some
    # information related to the normal behavior): OCSVM, AE, LSTM-AD, and CNN.
    #
    return model_name in ['ocsvm', 'ae', 'vae', 'lstm-ad', 'cnn']