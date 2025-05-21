"""
@summary 找出每个数据集上最好的算法名称及其fscore(acc+f1, 来源于 auto-cash)

"""
import sys

import pandas as pd

sys.path.append("/Users/sunwu/SW-Research/AutoML-Benchmark")
sys.path.append("/Users/sunwu/SW-Research/AutoML-Benchmark/deps")
from tshpo.automl_libs import *

# data_meta = pd.read_csv("auto_cash_data_meta_fea.csv", index_col=0)
data_meta = pd.read_csv("auto_cash_data_meta_fea.csv", index_col=0)
# data_model_perf_meta = pd.read_csv("autocash_每个算法在每个数据集上的最好的fscore.csv", index_col=0)
data_model_perf_meta = pd.read_csv("autocash_每个数据集上最好的算法及其fscore.csv", index_col=0)
print(data_meta)

auto_cash_mata = data_model_perf_meta.merge(data_meta, left_on="id", right_on="id", how="left")
print(auto_cash_mata)

auto_cash_mata.to_csv("auto_cash_meta_data.csv", index=False)
