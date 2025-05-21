import os
import pandas as pd

tshpo_file = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/tshpo_accuracy_and_rocauc.csv"
auto_cash_metric_dir = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/auto_cash/auto_cash_perf"
df_tshpo = pd.read_csv(tshpo_file)

auto_cash = pd.concat(
    [pd.read_csv(os.path.join(auto_cash_metric_dir, f), index_col=0) for f in os.listdir(auto_cash_metric_dir) if
     f.endswith(".csv")])
auto_cash_df = auto_cash.groupby(by=['dataset', 'metric'])['value'].agg(['mean', 'std']).reset_index()
import numpy as np

auto_cash_df['value'] = auto_cash_df.apply(lambda x: f"{np.round(x['mean'], 3)}({np.round(x['std'], 3)})", axis=1)


def myfun(x):
    return x


res = pd.pivot_table(auto_cash_df, index="dataset", columns=['metric'], values='value', aggfunc=myfun)
res.columns = [("Auto-CASH", "Accuracy"), ("Auto-CASH", "ROC AUC")]
res.to_csv("../results/Auto-CASH.csv")
res
