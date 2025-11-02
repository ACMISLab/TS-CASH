#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAT分类器模块 - 支持ConfigurationSpace的AutoSklearn兼容分类器
"""
import warnings
from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
)
from pytorch_lightning import seed_everything

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
from fc_gat import GATClassifierAutoSklearn
from pyutils.kvdb.kvdb_json import KVDBJson
from pyutils.util_pytorch_lighting import get_trainer

warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.abspath("../../"))

from fcvgae.libs import D1, D2, eval_predict_failure_type_v2
from gnn.gnn_lib import *
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class GCNClassifierAutoSklearn(AutoSklearnClassificationAlgorithm):
    """
    GAT分类器 - 支持ConfigurationSpace的AutoSklearn兼容分类器
    """

    def __init__(
            self,
            dataset_name="D1",
            batch_size=128,
            test_batch_size=None,
            max_epochs=1000,
            lr=0.01,
            min_delta=0.00,
            hidden_dim=64,
            dropout=0.1,
            patience=10,
            random_state=None,
            debug=False,
            train_ratio=1
    ):
        """
        初始化GAT分类器
        
        Args:
            dataset_name: 数据集名称，"D1" 或 "D2"
            batch_size: 训练批次大小
            test_batch_size: 测试批次大小，如果为None则使用全部测试数据
            max_epochs: 最大训练轮数
            patience: 早停耐心值
            lr: 学习率
            min_delta: 早停最小变化量
            num_heads: GAT注意力头数
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            random_state: 随机种子
            debug: 调试模式
        """
        self.dataset_name = dataset_name
        self.batch_size = int(batch_size)
        self.test_batch_size = int(test_batch_size) if test_batch_size is not None else None
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.min_delta = float(min_delta)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.random_state = random_state
        self.debug = debug

        # 初始化组件
        self.loader = None
        self.model = None
        self.trainer = None
        self.gconf = None
        self.train_samples = None
        self.test_samples = None
        self.train_ratio = train_ratio

    def fit(self, X=None, y=None, sample_weight=None):
        """
        训练模型
        
        Args:
            X: 特征数据（在此实现中不直接使用，而是从数据加载器获取）
            y: 标签数据（在此实现中不直接使用，而是从数据加载器获取）
            sample_weight: 样本权重
            
        Returns:
            self: 返回自身
        """
        # 设置随机种子
        pl.seed_everything(42 if self.random_state is None else self.random_state)

        # 初始化数据加载器
        if self.dataset_name.upper() == "D1":
            self.loader = D1()
        else:
            self.loader = D2()

        # 加载数据
        self.train_samples, self.test_samples = load_gnn_data(self.loader.DATA_NAME)
        if self.train_ratio < 1:
            # 随机从训练数据集中抽取 self.train_ratio 比例的数据
            np.random.shuffle(self.train_samples)
            self.train_samples = self.train_samples[:int(len(self.train_samples) * self.train_ratio)]
        print("train samples:", len(self.train_samples))
        print("test samples:", len(self.test_samples))
        # 配置模型参数
        self.gconf = GCNConf(
            n_node=self.train_samples[1][2].shape[0],
            n_fea=self.train_samples[1][2].shape[1],
            n_class=len(self.loader.TypeDict.keys()),
            lr=self.lr
        )

        # 创建GAT模型
        self.model = GCNClassifier(
            num_classes=self.gconf.n_class,
            in_channels=self.gconf.n_fea,
            hidden_channels=self.hidden_dim,
            dropout=self.dropout,
        )


        # 创建数据加载器
        train_loader = create_dataloader(self.train_samples, self.batch_size)
        test_loader = create_dataloader(self.test_samples, len(self.test_samples))

        print(f"train, test samples: {len(self.train_samples)}, {len(self.test_samples)}")
        # 创建训练器
        self.trainer = get_trainer(debug=self.debug)

        # 训练模型
        self.trainer.fit(model=self.model, train_dataloaders=train_loader)

        # # 进行预测和评估
        # predict = self.trainer.predict(model=self.model, dataloaders=test_loader)
        #
        # # 提取时间戳和预测结果
        # timestamps = np.concatenate([batch[0] for batch in predict], axis=0)
        # predictions = np.concatenate([batch[1] for batch in predict], axis=0)
        #
        # # 合并为DataFrame
        # predict_df = pd.DataFrame(np.column_stack((timestamps, predictions)), columns=['timestamp', 'predict'])
        #
        # prec, recall, f1 = eval_predict_failure_type_v2(self.loader, predict_df, desc={
        #     "dataset_name": self.dataset_name,
        #     "model_name": "GAT"
        # })
        #
        # print(f"Training completed - Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        # return prec, recall, f1
        return self.predict()

    def predict(self, X=None):
        """
        预测类别
        
        Args:
            X: 特征数据
            
        Returns:
            predictions: 预测的类别
        """
        # 进行预测和评估
        test_loader = create_dataloader(self.test_samples, len(self.test_samples))
        predict = self.trainer.predict(model=self.model, dataloaders=test_loader)

        # 提取时间戳和预测结果
        timestamps = np.concatenate([batch[0] for batch in predict], axis=0)
        predictions = np.concatenate([batch[1] for batch in predict], axis=0)

        # 合并为DataFrame
        predict_df = pd.DataFrame(np.column_stack((timestamps, predictions)), columns=['timestamp', 'predict'])

        prec, recall, f1 = eval_predict_failure_type_v2(self.loader, predict_df, desc={
            "dataset_name": self.dataset_name,
            "model_name": "GAT"
        })

        print(f"Training completed - Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return prec, recall, f1

    def predict_proba(self, X):
        """
        预测类别概率
        
        Args:
            X: 特征数据
            
        Returns:
            probabilities: 预测的类别概率
        """
        if self.model is None or self.trainer is None:
            raise NotImplementedError("模型必须先训练才能进行预测")

        # 简化的概率预测实现
        predictions = self.predict(X)
        n_classes = self.gconf.n_class if self.gconf else 3
        n_samples = len(predictions)

        # 创建one-hot编码的概率矩阵
        probas = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(predictions):
            if pred < n_classes:
                probas[i, pred] = 1.0
            else:
                # 如果预测值超出范围，使用均匀分布
                probas[i, :] = 1.0 / n_classes

        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        """
        获取分类器属性
        
        Returns:
            dict: 分类器属性字典
        """
        return {
            "shortname": "GAT",
            "name": "Graph Attention Network Classifier",
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
            feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None, seed=42
    ):
        """
        获取超参数搜索空间
        
        Returns:
            ConfigurationSpace: 超参数配置空间
        """
        cs = ConfigurationSpace(seed=seed)

        batch_size = UniformIntegerHyperparameter(
            "batch_size", 32, 256, default_value=128
        )
        test_batch_size = UniformIntegerHyperparameter(
            "test_batch_size", 64, 512, default_value=256
        )
        max_epochs = UniformIntegerHyperparameter(
            "max_epochs", 100, 2000, default_value=1000
        )

        lr = UniformFloatHyperparameter(
            "lr", 0.001, 0.1, default_value=0.01
        )
        min_delta = UniformFloatHyperparameter(
            "min_delta", 0.0, 0.01, default_value=0.00
        )
        hidden_dim = UniformIntegerHyperparameter(
            "hidden_dim", 32, 256, default_value=64
        )
        dropout = UniformFloatHyperparameter(
            "dropout", 0.0, 0.8, default_value=0.1
        )

        cs.add_hyperparameters([
            batch_size,
            test_batch_size,
            max_epochs,
            lr,
            min_delta,
            hidden_dim,
            dropout,
        ])

        return cs


if __name__ == '__main__':
    # 测试代码
    seed_everything(42)
    cs = GATClassifierAutoSklearn.get_hyperparameter_search_space()
    configs = cs.sample_configuration(50)
    print(configs[0].get_dictionary())
    db = KVDBJson("gat_pre.json")
    train_ratio = 0.1
    debug = False
    for dataset_name in ["D1", "D2"]:
        for config in configs:
            keys = config.get_dictionary()
            keys["dataset_name"] = dataset_name
            keys["train_ratio"] = train_ratio
            if db.is_exist(keys):
                continue
            # 创建分类器实例
            classifier = GCNClassifierAutoSklearn(
                debug=debug,
                dataset_name=dataset_name,
                train_ratio=train_ratio,
                **config
            )

            pre, recall, f1 = classifier.fit()

            db.add(keys, {"pre": pre, "recall": recall, "f1": f1})
