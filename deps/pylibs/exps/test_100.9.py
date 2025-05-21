"""
实验1： 在4个数据集上，每个数据集选10个数据做代表，使用3种超参数优化算法优化模型的参数，每个实验运行优化200次，重复10次， 优化指标三个，
"""

import sys

from pylibs.config import Env

LIB = "/tmp/pylibs-1.0.0-py3.10.egg"
sys.path.append(LIB)
from pylibs.smac3.search_pace import parse_and_run
from pylibs.utils.util_servers import Servers
if __name__ == '__main__':
    server = Servers.DASK_SCHEDULER
    server.upload_pylibs()
    options = f"--exp-name test_exp --datasets DEBUG1 DEBUG2  --facade hyperparameter_optimization  random blackbox --n-uts 1 --dask-client {server.name} --n-trials 10 --n-seed 3 --opt-metrics VUS_ROC VUS_PR BEST_F1_SCORE  --parallel -y"
    client = server.get_dask_client()
    # client.restart(wait_for_workers=False)
    # client.upload_file(Env.LOCAL_PYLIBS_WHEEL)
    parse_and_run(options)
