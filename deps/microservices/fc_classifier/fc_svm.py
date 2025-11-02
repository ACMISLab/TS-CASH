#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM分类器 - 支持ConfigurationSpace的AutoSklearn兼容分类器

依赖安装:
pip install scikit-learn
pip install ConfigSpace
pip install dgl
pip install torch
"""

import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array

from pyutils.util_sys import is_macos

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath("../../"))
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
from fcvgae.kvdb_json import KVDBJson
from fcvgae.libs import eval_predict_failure_type, D1, D2, load_pkl
import dgl.nn.pytorch

# 忽略警告
warnings.filterwarnings("ignore")

from fc_config import get_classic_data_home
DATA_HOME = get_classic_data_home()

def load_classic_fault_type_data(file):
    """
    加载经典故障类型数据
    """
    pkil_data = load_pkl(file)
    # 最后一列是failure_type_id
    feas = []
    labels = []
    ts_arr = []
    for d in pkil_data:
        ts, graph, fea, fault_ms_id, failure_type_id = d
        pooler = dgl.nn.pytorch.AvgPooling()
        fea = pooler(graph, fea)
        feas.append(fea[0].numpy())
        labels.append(failure_type_id)
        ts_arr.append(ts)

    return np.asarray(feas), np.asarray(labels), np.asarray(ts_arr)


class SVMClassifierAutoSklearn(AutoSklearnClassificationAlgorithm):
    """
    SVM分类器 - 支持ConfigurationSpace的AutoSklearn兼容分类器
    """

    def __init__(
            self,
            dataset_name="D1",
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            max_iter=-1,
            random_state=None,
            debug=False,
            train_ratio=1.0
    ):
        """
        初始化SVM分类器
        
        Args:
            dataset_name: 数据集名称，"D1" 或 "D2"
            C: 正则化参数
            kernel: 核函数类型
            degree: 多项式核函数的度数
            gamma: 核函数系数
            coef0: 核函数中的独立项
            shrinking: 是否使用shrinking启发式
            probability: 是否启用概率估计
            tol: 停止准则的容忍度
            cache_size: 核缓存大小（MB）
            max_iter: 最大迭代次数
            random_state: 随机种子
            debug: 调试模式
            train_ratio: 训练数据比例
        """
        self.dataset_name = dataset_name
        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.shrinking = bool(shrinking)
        self.probability = bool(probability)
        self.tol = float(tol)
        self.cache_size = float(cache_size)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.debug = debug
        self.train_ratio = float(train_ratio)

        # 初始化组件
        self.loader = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.ts_train = None
        self.X_test = None
        self.y_test = None
        self.ts_test = None

        # 设置随机种子
        if self.random_state is not None:
            np.random.seed(self.random_state)
            pl.seed_everything(self.random_state)

    def fit(self, X=None, y=None, sample_weight=None):
        """
        训练SVM分类器
        
        Args:
            X: 特征数据（在此实现中不直接使用，而是从数据加载器获取）
            y: 标签数据（在此实现中不直接使用，而是从数据加载器获取）
            sample_weight: 样本权重
            
        Returns:
            self: 返回自身实例
        """
        # 设置随机种子
        pl.seed_everything(42 if self.random_state is None else self.random_state)

        # 初始化数据加载器
        if self.dataset_name.upper() == "D1":
            self.loader = D1()
        else:
            self.loader = D2()

        # 加载数据
        self.X_train, self.y_train, self.ts_train = load_classic_fault_type_data(
            f"{DATA_HOME}/{self.loader.DATA_NAME}/chunk_train.pkl"
        )
        self.X_test, self.y_test, self.ts_test = load_classic_fault_type_data(
            f"{DATA_HOME}/{self.loader.DATA_NAME}/chunk_test.pkl"
        )

        # 应用训练数据比例
        if self.train_ratio < 1.0:
            np.random.shuffle(self.X_train)
            n_samples = int(len(self.X_train) * self.train_ratio)
            self.X_train = self.X_train[:n_samples]
            self.y_train = self.y_train[:n_samples]
            self.ts_train = self.ts_train[:n_samples]

        if self.debug:
            print(f"训练样本数: {len(self.X_train)}, 测试样本数: {len(self.X_test)}")
            print(f"特征维度: {self.X_train.shape[1]}, 类别数: {len(np.unique(self.y_train))}")

        # 创建SVM模型
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        # 训练模型
        self.model.fit(self.X_train, self.y_train)

        if self.debug:
            print("SVM模型训练完成")

        return self.predict()

    def predict(self, X=None):
        """
        使用训练好的模型进行预测并返回评估指标
        
        Args:
            X: 特征数据（在此实现中不直接使用）
            
        Returns:
            tuple: (precision, recall, f1) 评估指标
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        try:
            # 进行预测
            predictions = self.model.predict(self.X_test)

            # 创建预测结果DataFrame
            predict_df = pd.DataFrame(
                list(zip(self.ts_test, predictions)),
                columns=["timestamp", "predict"]
            )

            # 评估预测结果
            prec, recall, f1 = eval_predict_failure_type(
                self.loader,
                predict_df,
                desc={
                    "dataset_name": self.dataset_name,
                    "model_name": "SVM"
                }
            )

            if self.debug:
                print(f"预测完成 - Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            return prec, recall, f1

        except Exception as e:
            if self.debug:
                print(f"预测过程中出现错误: {str(e)}")
            raise e

    def predict_proba(self, X=None):
        """
        预测类别概率
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 类别概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        try:
            if self.probability:
                # 如果启用了概率估计，直接获取概率
                probabilities = self.model.predict_proba(self.X_test)
            else:
                # 如果没有启用概率估计，使用决策函数创建伪概率
                decision_scores = self.model.decision_function(self.X_test)
                if len(decision_scores.shape) == 1:
                    # 二分类情况
                    probabilities = np.column_stack([-decision_scores, decision_scores])
                else:
                    # 多分类情况
                    probabilities = decision_scores

                # 应用softmax转换为概率
                from scipy.special import softmax
                probabilities = softmax(probabilities, axis=1)

            probabilities = convert_multioutput_multiclass_to_multilabel(probabilities)
            return probabilities

        except Exception as e:
            if self.debug:
                print(f"概率预测过程中出现错误: {str(e)}")
            raise e

    @staticmethod
    def get_properties(dataset_properties=None):
        """
        获取分类器属性
        
        Returns:
            dict: 分类器属性字典
        """
        return {
            "shortname": "SVM",
            "name": "Support Vector Machine Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
            feat_type: Optional[FEAT_TYPE_TYPE] = None,
            dataset_properties=None,
            seed=42
    ):
        """
        获取超参数搜索空间
        
        Args:
            feat_type: 特征类型
            dataset_properties: 数据集属性
            seed: 随机种子
            
        Returns:
            ConfigurationSpace: 超参数配置空间
        """
        cs = ConfigurationSpace(seed=seed)

        # 正则化参数
        C = UniformFloatHyperparameter(
            "C",
            lower=1e-4,
            upper=1e4,
            default_value=1.0,
            log=True
        )

        # 核函数类型
        kernel = CategoricalHyperparameter(
            "kernel",
            choices=["linear", "poly", "rbf", "sigmoid"],
            default_value="rbf"
        )

        # 多项式核函数的度数
        degree = UniformIntegerHyperparameter(
            "degree",
            lower=2,
            upper=5,
            default_value=3
        )

        # 核函数系数
        gamma = CategoricalHyperparameter(
            "gamma",
            choices=["scale", "auto"],
            default_value="scale"
        )

        # 核函数中的独立项
        coef0 = UniformFloatHyperparameter(
            "coef0",
            lower=-1.0,
            upper=1.0,
            default_value=0.0
        )

        # 是否使用shrinking启发式
        shrinking = CategoricalHyperparameter(
            "shrinking",
            choices=[True, False],
            default_value=True
        )

        # 是否启用概率估计
        probability = CategoricalHyperparameter(
            "probability",
            choices=[True, False],
            default_value=False
        )

        # 停止准则的容忍度
        tol = UniformFloatHyperparameter(
            "tol",
            lower=1e-5,
            upper=1e-1,
            default_value=1e-3,
            log=True
        )

        # 添加超参数到配置空间
        cs.add_hyperparameters([
            C, kernel, degree, gamma, coef0,
            shrinking, probability, tol
        ])

        return cs


if __name__ == "__main__":
    # 测试代码
    print("开始测试SVM分类器...")

    # 创建分类器实例
    classifier = SVMClassifierAutoSklearn(
        dataset_name="D1",
        C=1.0,
        kernel="rbf",
        probability=True,  # 启用概率估计以便测试predict_proba
        debug=True
    )

    try:
        # 训练模型
        print("开始训练...")
        classifier.fit()

        # 进行预测
        print("开始预测...")
        prec, recall, f1 = classifier.predict()

        print(f"评估结果 - Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # 测试概率预测
        print("测试概率预测...")
        probabilities = classifier.predict_proba()
        print(f"概率预测形状: {probabilities.shape}")

        print("测试完成！")

    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")

    # 测试超参数空间
    print("\n测试超参数搜索空间...")
    cs = SVMClassifierAutoSklearn.get_hyperparameter_search_space()
    print(f"超参数数量: {len(cs.get_hyperparameters())}")

    # 采样配置
    config = cs.sample_configuration()
    print(f"采样配置: {config}")

    # 测试属性
    properties = SVMClassifierAutoSklearn.get_properties()
    print(f"\n分类器属性: {properties}")

    # 测试数据库存储
    db = KVDBJson("test_svm_results.json")
    keys = "test_svm_config"

    if db.query(keys) is not None:
        print(f"\n已存在配置: {db.query(keys)}")
    else:
        print("\n保存测试配置...")
        test_config = {
            "dataset_name": "D1",
            "C": 1.0,
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "probability": True
        }
        db.insert(keys, test_config)
        print(f"配置已保存: {test_config}")
