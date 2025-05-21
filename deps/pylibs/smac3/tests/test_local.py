"""
实验1： 在4个数据集上，每个数据集选10个数据做代表，使用3种超参数优化算法优化模型的参数，每个实验运行优化200次，重复10次， 优化指标三个，
"""
import shutil

import numpy as np
from numpy.testing import assert_equal

from exps.search_pace import parse_and_run

if __name__ == '__main__':
    # random --parallel
    #  DEBUG2
    shutil.rmtree("/Users/sunwu/Documents/experiment_results/automl_results/debug_False/test_exp")
    options = f"--exp-name test_exp --datasets DEBUG1 --facade hyperparameter_optimization   --n-uts 2 --dask-client local --n-trials 10 --n-seed 2 --opt-metrics VUS_ROC VUS_PR BEST_F1_SCORE  --yes --parallel"
    res1 = parse_and_run(options)
    res2 = parse_and_run(options)

    (assert_equal(res2.iloc[:, 0].values, res1.iloc[:, 0].values))
    (assert_equal(res2.iloc[:, 1].values, res1.iloc[:, 1].values))
