"""
实验1： 在4个数据集上，每个数据集选10个数据做代表，使用3种超参数优化算法优化模型的参数，每个实验运行优化200次，重复10次， 优化指标三个，
"""
from pylibs.smac3.search_pace import start_experiments_v3
from pylibs.utils.util_servers import Servers

if __name__ == '__main__':
    server_name = Servers.DASK_SCHEDULER.name
    options = (f"--exp-name effect_n_trials_all_2024_03_19 --datasets YAHOO  NASA-SMAP  SMD  NASA-MSL --facade "
               f"hyperparameter_optimization  random --n-uts 5 --dask-client {server_name} --n-trials 10 20 50 100 "
               f"150 200 250 300  --n-seed 3 --opt-metrics VUS_ROC BEST_F1_SCORE  --yes --parallel")
    # --parallel
    start_experiments_v3(options)
