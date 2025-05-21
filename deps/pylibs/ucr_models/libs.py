#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/9/3 14:02
# @Author  : gsunwu@163.com
# @File    : libs.py
# @Description:
import sys
from pathlib import Path

import numpy as np

import os


class UCRModelConfig:
    test_epoch = 30


def _train(clf, i, debug):
    HOME = os.path.join("figs", Path(sys.argv[0]).stem)
    os.makedirs(HOME, exist_ok=True)
    from ucr.ucr_dataset_loader import calc_acc_score, load_ucr_by_number
    # 展示test
    import matplotlib.pyplot as plt

    WINDOW_SIZE = 128
    # for i in range(4, 5):
    if debug and i != 4:
        return
    X_train_window, x_test_window, anomaly_range = load_ucr_by_number(i, window_length=WINDOW_SIZE)

    clf.fit(X_train_window)
    acc_score = clf.accuracy_score(x_test_window, anomaly_range)
    anomaly_scores_ = clf.anomaly_scores_
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    fig.subplots_adjust(hspace=0.5)  # 0.5 是子图之间的垂直间距，可以根据需要调整
    axes[0].set_title("Red span: ground truth anomaly")
    ax = axes[0]
    ax.axvspan(anomaly_range[0], anomaly_range[1], color='red', alpha=0.8, label='红色区域')
    ax.plot(x_test_window[:, -1])

    axes[1].set_title(f"Vertical line: anomaly with maximum score, acc score: {acc_score}")
    axes[1].plot(anomaly_scores_)
    axes[1].axvline(x=np.argmax(anomaly_scores_), color='red', linestyle='--', label='垂直线 x=5')
    print(f"acc score: {acc_score}, detect pos: {clf.anomaly_pos_}, anomaly range: {anomaly_range}")
    fig.savefig(f"{HOME}/data_{i}.png")
    # break


def eval_model(clf, debug=sys.platform=="darwin"):
    if debug:
        for i in range(1, 250):
            _train(clf, i, debug)
    else:
        from joblib import Parallel, delayed
        out_metric = Parallel(n_jobs=5, verbose=10)(delayed(_train)(clf, i, debug) for i in range(1,250))