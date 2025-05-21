#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/9/3 12:38
# @Author  : gsunwu@163.com
# @File    : vae_test.py
# @Description:

import os
import numpy as np

from ucr.ucr_dataset_loader import load_ucr_by_number

HOME = "vae"
os.makedirs(HOME, exist_ok=True)
from ucr.ucr_dataset_loader import calc_acc_score
# 展示test
import matplotlib.pyplot as plt

WINDOW_SIZE = 128
for i in range(1, 250):
    X_train_window, x_test_window, anomaly_range = load_ucr_by_number(i, window_length=WINDOW_SIZE)
    vae = VAE(window_size=WINDOW_SIZE)
    vae.fit(X_train_window)

    acc_score = vae.accuracy_score(x_test_window, anomaly_range)
    score = vae.score_
    detect_pose = vae.detect_pos_

    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    fig.subplots_adjust(hspace=0.5)  # 0.5 是子图之间的垂直间距，可以根据需要调整
    axes[0].set_title("Red span: ground truth anomaly")
    ax = axes[0]
    ax.axvspan(anomaly_range[0], anomaly_range[1], color='red', alpha=0.8, label='红色区域')
    ax.plot(x_test_window[:, -1])

    acc = calc_acc_score(anomaly_range, np.argmax(score), len(x_test_window))
    axes[1].set_title(f"Vertical line: anomaly with maximum score, acc score: {acc}")
    axes[1].plot(score)
    axes[1].axvline(x=np.argmax(score), color='red', linestyle='--', label='垂直线 x=5')

    fig.savefig(f"{HOME}/{HOME}_{i}.png")
    # break
