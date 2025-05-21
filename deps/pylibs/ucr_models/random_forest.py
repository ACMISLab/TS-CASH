#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/9/3 17:19
# @Author  : gsunwu@163.com
# @File    : random_forest.py
# @Description:
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import tsaug
from sklearn.ensemble import RandomForestClassifier

from ucr.ucr_dataset_loader import calc_acc_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs)
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augament(ts):
    """
    input: 一条数据，[0.45,0.23,....,0.11], length=128
    output: 两天数据，[[0.45,0.23,....,0.11],[0.33,0.4,....,0.22]]


    Parameters
    ----------
    ts :

    Returns
    -------

    """

    def tail_padding_zero(l, length):
        ret = []
        ret.extend(l)
        if length - len(l) <= 0:
            return ret
        for i in range(length - len(l)):
            ret.append(0)
        return ret

    ret = []
    i_list = []
    for _ in range(1, 3):
        while True:
            i = random.randint(0, 100) % 9  # 随机选择一个增强操作
            if i in i_list:  # 确保每次选择的增强操作不同
                continue
            else:
                i_list.append(i)
                break

        X = np.array(ts)  # 将输入的时间序列转换为 NumPy 数组
        if i == 0:
            ret.append(tsaug.AddNoise(scale=0.1).augment(X))  # 添加噪声
        if i == 1:
            ret.append(tsaug.Convolve(window="flattop", size=20).augment(X))  # 卷积平滑
        if i == 2:
            term = tsaug.Crop(size=100).augment(X)  # 裁剪
            ret.append(tail_padding_zero(term, len(X)))  # 填充零
        if i == 3:
            ret.append(tsaug.Drift(max_drift=0.7, n_drift_points=20).augment(X))  # 漂移
        if i == 4:
            ret.append(tsaug.Pool(size=40).augment(X))  # 池化
        if i == 5:
            ret.append(tsaug.Quantize(n_levels=100).augment(X))  # 量化
        if i == 6:
            term = tsaug.Resize(size=100).augment(X)  # 重采样
            ret.append(tail_padding_zero(term, len(X)))  # 填充零
        if i == 7:
            ret.append(tsaug.Reverse().augment(X))  # 反转
        if i == 8:
            ret.append(tsaug.TimeWarp(n_speed_change=20, max_speed_ratio=6).augment(X))  # 时间扭曲

    return ret


class RF:
    def __init__(self):
        self.model_ = None

    def fit(self, X_train):
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        train_neg_data = []
        start_time = time.time()
        for train_pos in X_train:
            augamented_neg_list = augament(train_pos)
            for augamented_neg in augamented_neg_list:
                train_neg_data.append(augamented_neg)

        train_neg_data = np.asarray(train_neg_data)
        # 创建标签
        labels_X_train = np.zeros(X_train.shape[0], dtype=int)
        labels_train_neg_data = np.ones(train_neg_data.shape[0], dtype=int)

        # 合并数据和标签
        X_combined = np.vstack((X_train, train_neg_data))
        y_combined = np.concatenate((labels_X_train, labels_train_neg_data))

        # 打乱数据
        indices = np.arange(X_combined.shape[0])
        np.random.shuffle(indices)

        X_combined_shuffled = X_combined[indices]
        y_combined_shuffled = y_combined[indices]
        # 下面的数据，如何将
        print("aug_time:", (time.time() - start_time))
        clf.fit(X_combined_shuffled, y_combined_shuffled)
        self.model_ = clf

    def accuracy_score(self, X_test, anomaly_range):
        # 标签1所在的序号，用于获取predict_proba的结果
        anomaly_label_index = list(self.model_.classes_).index(1)
        scores = self.model_.predict_proba(X_test)[:, anomaly_label_index]

        self.anomaly_scores_ = scores
        self.anomaly_pos_ = np.argmax(scores)
        score = calc_acc_score(anomaly_range, self.anomaly_pos_, len(X_test))
        return score


if __name__ == "__main__":
    from libs import eval_model

    vae = RF()
    eval_model(vae,False)
