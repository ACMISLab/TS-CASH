#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/23 11:36
# @Author  : gsunwu@163.com
# @File    : xgboost.py
# @Description:
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
import xgboost as xgb
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from ..utils.detectorB import DetectorB
from pylibs.uts_models.benchmark_models.utils.utility import invert_order


@dataclass
class XGBoost(DetectorB):
    model_name: str = "xgboost"
    # 使用多少棵树来拟合，也可以理解为多少次迭代。默认100；
    n_estimators: float = 100

    # 树相关参数：
    learning_rate: float = 0.3
    # max_depth：每一棵树最大深度，默认6；
    max_depth: int = 6
    # min_child_weight：可以理解为叶子节点最小样本数，默认1；
    min_child_weight: int = 1
    # gamma：叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
    gamma: float = 0
    # 抽样：
    # subsample：训练集抽样比例，每次拟合一棵树之前，都会进行该抽样步骤。默认1，取值范围(0, 1]
    subsample: float = 1
    # 列抽样：
    # colsample_bytree：每次拟合一棵树之前，决定使用多少个特征。
    colsample_bytree: float = 0.5
    # colsample_bylevel：每层树节点，可以使用多少个特征。
    colsample_bylevel: float = 0.5
    # colsample_bynode：每个节点分裂之前，决定使用多少个特征。
    colsample_bynode: float = 0.5
    # 这三个参数默认都是1，取值范围(0, 1]，列抽样也可以理解为特征抽样，注意这三个参数是同时作用的，比如训练集总共有64个特征，参数
    # {‘colsample_bytree’:0.5, ‘colsample_bylevel’:0.5, ‘colsample_bynode’:0.5}，则每次拟合一棵树之前，在64个特征中随机抽取其中32个特征，然后在树的每一层，在32个特征中随机抽取16个特征，然后每次节点分裂，从16个特征中随机抽取8个特征。

    # tree_method：默认是auto，会自动选择最保守的方式。这个是决定训练速度的关键超参数。一般有三种树方法：exact（精确方法），approx（近似方法），hist（直方图方法），其中hist就是LightGBM中的直方图方法，速度最快，approx速度次之，exact最慢。
    tree_method: str = "auto"
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    seed: int = 42
    n_jobs:int =1
    def __post_init__(self):
        self.detector_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            tree_method=self.tree_method,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            use_label_encoder=False
        )

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        try:
            X = check_array(X)
        except:
            X = X.reshape(-1, 1)

        X = check_array(X)
        self.detector_.fit(X, y)
        # invert decision_scores_. Outliers comes with higher outlier scores.
        # self.decision_scores_ = -self.detector_.score_samples(X)
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        # invert outlier scores. Outliers comes with higher outlier scores
        return invert_order(self.detector_.decision_function(X))

    def estimators_(self):
        """The collection of fitted sub-estimators.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.detector_.estimators_

    def estimators_samples_(self):
        """The subset of drawn samples (i.e., the in-bag samples) for
        each base estimator.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.detector_.estimators_samples_

    def max_samples_(self):
        """The actual number of samples.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.detector_.max_samples_

    def score(self, X):
        return self.detector_.predict_proba(X)[:,1]
