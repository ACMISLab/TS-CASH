#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP分类器 - 支持ConfigurationSpace的AutoSklearn兼容分类器

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
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array

from fc_config import get_classic_data_home

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


class MLPClassifierAutoSklearn(AutoSklearnClassificationAlgorithm):
    """
    MLP分类器 - 支持ConfigurationSpace的AutoSklearn兼容分类器
    """

    def __init__(
            self,
            dataset_name="D1",
            hidden_layer_size=100,  # Changed from hidden_layer_sizes to hidden_layer_size
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=200,
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=10,
            debug=False,
            train_ratio=1.0
    ):
        """
        初始化MLP分类器
        
        Args:
            dataset_name: 数据集名称，"D1" 或 "D2"
            hidden_layer_size: 隐藏层大小  # Updated docstring
            activation: 激活函数
            solver: 优化器
            alpha: L2正则化参数
            batch_size: 批次大小
            learning_rate: 学习率策略
            learning_rate_init: 初始学习率
            power_t: 学习率衰减指数
            max_iter: 最大迭代次数
            shuffle: 是否打乱数据
            random_state: 随机种子
            tol: 优化容忍度
            verbose: 是否输出详细信息
            warm_start: 是否热启动
            momentum: 动量
            nesterovs_momentum: 是否使用Nesterov动量
            early_stopping: 是否早停
            validation_fraction: 验证集比例
            beta_1: Adam优化器参数
            beta_2: Adam优化器参数
            epsilon: Adam优化器参数
            n_iter_no_change: 早停容忍次数
            debug: 调试模式
            train_ratio: 训练数据比例
        """
        self.dataset_name = dataset_name
        self.hidden_layer_sizes = (hidden_layer_size,)  # Convert to tuple
        self.activation = activation
        self.solver = solver
        self.alpha = float(alpha)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = float(learning_rate_init)
        self.power_t = float(power_t)
        self.max_iter = int(max_iter)
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.warm_start = bool(warm_start)
        self.momentum = float(momentum)
        self.nesterovs_momentum = bool(nesterovs_momentum)
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(validation_fraction)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.n_iter_no_change = int(n_iter_no_change)
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
        训练MLP分类器
        
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

        # 创建MLP模型
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
            warm_start=self.warm_start,
            momentum=self.momentum,
            nesterovs_momentum=self.nesterovs_momentum,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            n_iter_no_change=self.n_iter_no_change
        )

        # 训练模型
        self.model.fit(self.X_train, self.y_train)

        if self.debug:
            print("MLP模型训练完成")
            print(f"训练迭代次数: {self.model.n_iter_}")
            print(f"训练损失: {self.model.loss_:.4f}")

        return  self.predict()

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
                    "model_name": "MLP"
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
            # 获取概率预测
            probabilities = self.model.predict_proba(self.X_test)
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
            "shortname": "MLP",
            "name": "Multi-layer Perceptron Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": False,
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
        

        # 隐藏层大小（简化为单层）
        hidden_layer_size = UniformIntegerHyperparameter(
            "hidden_layer_size", 
            lower=50, 
            upper=500, 
            default_value=100
        )
        
        # 激活函数
        activation = CategoricalHyperparameter(
            "activation", 
            choices=["identity", "logistic", "tanh", "relu"], 
            default_value="relu"
        )
        
        # 优化器
        solver = CategoricalHyperparameter(
            "solver", 
            choices=["lbfgs", "sgd", "adam"], 
            default_value="adam"
        )
        
        # L2正则化参数
        alpha = UniformFloatHyperparameter(
            "alpha", 
            lower=1e-7, 
            upper=1e-1, 
            default_value=1e-4, 
            log=True
        )
        
        # 学习率策略
        learning_rate = CategoricalHyperparameter(
            "learning_rate", 
            choices=["constant", "invscaling", "adaptive"], 
            default_value="constant"
        )
        
        # 初始学习率
        learning_rate_init = UniformFloatHyperparameter(
            "learning_rate_init", 
            lower=1e-5, 
            upper=1e-1, 
            default_value=1e-3, 
            log=True
        )
        
        # 最大迭代次数
        max_iter = UniformIntegerHyperparameter(
            "max_iter", 
            lower=100, 
            upper=1000, 
            default_value=200
        )
        
        # 早停
        early_stopping = CategoricalHyperparameter(
            "early_stopping", 
            choices=[True, False], 
            default_value=False
        )
        
        # 验证集比例
        validation_fraction = UniformFloatHyperparameter(
            "validation_fraction", 
            lower=0.1, 
            upper=0.3, 
            default_value=0.1
        )
        

        # 添加超参数到配置空间
        cs.add_hyperparameters([
             hidden_layer_size, activation, solver, alpha,
            learning_rate, learning_rate_init, max_iter, early_stopping,
            validation_fraction
        ])
        
        return cs


if __name__ == "__main__":
    # 测试代码
    print("开始测试MLP分类器...")
    
    # 创建分类器实例
    classifier = MLPClassifierAutoSklearn(
        dataset_name="D1",
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        max_iter=200,
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
    cs = MLPClassifierAutoSklearn.get_hyperparameter_search_space()
    print(f"超参数数量: {len(cs.get_hyperparameters())}")
    
    # 采样配置
    config = cs.sample_configuration()
    print(f"采样配置: {config}")
    
    # 测试属性
    properties = MLPClassifierAutoSklearn.get_properties()
    print(f"\n分类器属性: {properties}")
    
    # 测试数据库存储
    db = KVDBJson("test_mlp_results.json")
    keys = "test_mlp_config"
    
    if db.query(keys) is not None:
        print(f"\n已存在配置: {db.query(keys)}")
    else:
        print("\n保存测试配置...")
        test_config = {
            "dataset_name": "D1",
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200
        }
        db.insert(keys, test_config)
        print(f"配置已保存: {test_config}")