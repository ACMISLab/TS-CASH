"""
实验1： 在4个数据集上，每个数据集选10个数据做代表，使用3种超参数优化算法优化模型的参数，每个实验运行优化200次，重复10次， 优化指标三个，
"""

import sys

LIB = "/tmp/pylibs-1.0.0-py3.10.egg"
sys.path.append(LIB)
from pylibs.utils.util_dask import DaskCluster
from pylibs.smac3.search_pace import start_experiments

if __name__ == '__main__':
    server_name = DaskCluster.TYPE_164
    # NASA-SMAP SMD NASA-MSL
    # random blackbox
    # VUS_PR
    options = f"--exp-name overall_results_acc_and_runtime_2024_03_17 --datasets  YAHOO  NASA-SMAP  SMD  NASA-MSL --facade hyperparameter_optimization  --n-uts 10 --dask-client {server_name} --n-trials 200 --n-seed 10 --opt-metrics VUS_ROC  BEST_F1_SCORE  --yes --parallel --data-autoscaling"
    start_experiments(server_name, options)
