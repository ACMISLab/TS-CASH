#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/13 08:46
# @Author  : gsunwu@163.com
# @File    : best_sample_rate.py
# @Description:
import datetime
import time
from dataclasses import dataclass
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Enum,
    DECIMAL,
    DateTime,
    Boolean,
    UniqueConstraint,
    Index,
    Text,
    Float, Engine, and_, TIMESTAMP
)

from exps.search_pace import UtilModelCache

Base = declarative_base()

class BestSampleRateCache(Base):
    """ 必须继承Base """
    # 数据库中存储的表名
    __tablename__ = "best_sample_rate"
    # 对于必须插入的字段，采用nullable=False进行约束，它相当于NOT NULL

    # >>> time.time()
    # 1712791436.526145
    # DECIMAL(26,6)
    #                            1712791436.526145
    # 2099年12月31日23:59:59为止是 4102444799.
    # DECIMAL(32, 8)： 共 32 位，小数部分占8位
    id = Column(Integer, primary_key=True, autoincrement=True, comment="主键")
    create_time = Column(
        DECIMAL(32, 8),
        default=time.time(),
        comment="创建时间")
    update_time = Column(
        DECIMAL(32, 8),
        default=time.time(),
        comment="更新时间")
    time_app_start = Column(
        DECIMAL(32, 8),
        comment="实验开始时间")
    time_app_end = Column(
        DECIMAL(32, 8),
        comment="实验结束时间")
    time_train_start = Column(
        DECIMAL(32, 8),
        comment="训练开始时间")
    time_train_end = Column(
        DECIMAL(32, 8),
        comment="训练结束时间")
    time_eval_start = Column(
        DECIMAL(32, 8),
        comment="评估开始时间")
    time_eval_end = Column(
        DECIMAL(32, 8),
        comment="评估结束时间")
    elapsed_train_seconds = Column(
        DECIMAL(32, 8),
        comment="训练时间, 单位秒")
    elapsed_evaluate_seconds = Column(
        DECIMAL(32, 8),
        comment="模型评估时间（计算精度的时间），单位秒")
    # vus_roc = Column(
    #     DECIMAL(32, 8),
    #     comment="模型精度:VUS ROC")
    # vus_pr = Column(
    #     DECIMAL(32, 8),
    #     comment="模型精度:VUS PR")
    # best_f1_score = Column(
    #     DECIMAL(32, 8),
    #     comment="模型精度:Best F1 score")
    dataset_name = Column(
        Text,
        comment="训练数据的名称")
    data_id = Column(
        Text,
        comment="数据集中数据的id")
    data_sample_method = Column(
        Text,
        comment="训练数据抽样方法")

    best_sample_rate = Column(
        DECIMAL(32, 8),
        comment="训练数据的抽样比例")

    seed = Column(
        Integer,
        comment="随机种子")

    test_rate = Column(
        DECIMAL(32, 8),
        comment="测试集的比例")

    hp_cfg_id = Column(
        Text,
        comment="模型及其参数配置，多参数用;分割，如：classifier=lof;lof_n_neighbors=100")

    def __str__(self):
        return str(self.__dict__)


class BestSampleRate:
    def calc_best_sample_rate(self):
        UtilModelCache()
