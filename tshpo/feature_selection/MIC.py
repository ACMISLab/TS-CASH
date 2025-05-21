#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 07:35
# @Author  : gsunwu@163.com
# @File    : MIC.py
# @Description: SelectKBest 允许你指定一个评分函数来评估特征的重要性。在你的例子中，使用的是 mutual_info_classif，这是一个用于分类问题的互信息评分函数。它衡量了每个特征与目标变量之间的依赖关系。
# 互信息：信息论中的一个重要概念，用于量化两个随机变量之间的依赖关系。
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from tshpo.feature_selection.api import FS
from tshpo.lib_class import ExpConf

class MIC(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        k = self.check_n_feature(X_train.shape[1] * econf.feature_selec_rate)
        clf = SelectKBest(score_func=mutual_info_classif, k=k)
        clf.fit(X_train, y_train)
        X_train_new = clf.transform(X_train)
        X_test_new = clf.transform(X_test)
        return X_train_new, y_train, X_test_new, y_test


if __name__ == '__main__':
    from automl_libs import load_dataset_at_fold, TSHPOFramework

    econf = ExpConf(feature_selec_rate=1)

    X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=10,
                                                            fold_index=0, seed=42)

    _X_train, _y_train, X_test, y_test = MIC().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train.shape[1] == _X_train.shape[1]
    print(X_train.shape, _X_train.shape)

    econf = ExpConf(feature_selec_rate=0.1)

    _X_train1, _y_train1, X_test1, y_test1 = MIC().transform(X_train, y_train, X_test, y_test, econf)
    assert _X_train1.shape[1] < _X_train.shape[1]
    print( X_train.shape, _X_train1.shape)
