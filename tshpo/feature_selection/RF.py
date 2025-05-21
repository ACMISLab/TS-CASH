#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:31
# @Author  : gsunwu@163.com
# @File    : SP.py
# @Description:
"""
它的工作原理是根据特征的重要性评分选择前 50% 的特征

"""
import numpy as np
from tshpo.feature_selection.api import FS
from tshpo.lib_class import ExpConf
from sklearn.ensemble import RandomForestClassifier
from tshpo.automl_libs import load_dataset_at_fold, TSHPOFramework
class RF(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):

        model = RandomForestClassifier(n_estimators=100, random_state=econf.random_state)
        model.fit(X_train, y_train)

        # 计算阈值
        feature_importances = model.feature_importances_
        threshold = np.percentile(feature_importances, 100 * (1 - econf.feature_selec_rate))
        indices = np.where(feature_importances >= threshold)[0]

        # 最多允许30个特征
        indices = indices[: TSHPOFramework.max_n_features]
        return X_train[:, indices], y_train, X_test[:, indices], y_test


if __name__ == '__main__':


    econf = ExpConf(feature_selec_rate=1)

    X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=10,
                                                            fold_index=0, seed=42)

    _X_train, _y_train, X_test, y_test = RF().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train.shape[1] == _X_train.shape[1]
    print(X_train.shape, _X_train.shape)

    econf = ExpConf(feature_selec_rate=0.1)

    _X_train1, _y_train1, X_test1, y_test1 = RF().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train1.shape[1] < _X_train.shape[1]
    print(X_train.shape, _X_train1.shape)
