import os
import pandas as pd

auto_cash_metric_dir = "./automhs_gpt_perf"
import numpy as np

df = pd.concat(
    [pd.read_csv(os.path.join(auto_cash_metric_dir, f), index_col=0) for f in os.listdir(auto_cash_metric_dir) if
     f.endswith(".csv")])
auto_cash_df = df.groupby(by=['dataset', "gpt_version", 'metric'])['value'].agg(['mean', 'std']).reset_index()

auto_cash_df['value'] = auto_cash_df.apply(lambda x: f"{np.round(x['mean'], 3)}({np.round(x['std'], 3)})", axis=1)


def myfun(x):
    return x


for _gpt_version in df.gpt_version.unique():
    _ana_df = auto_cash_df[auto_cash_df['gpt_version'] == _gpt_version]
    res = pd.pivot_table(_ana_df, index="dataset", columns=['metric'], values='value', aggfunc=myfun)
    res.columns = [(f"AutoMHS-GPT({_gpt_version})", "Accuracy"), (f"AutoMHS-GPT({_gpt_version})", "ROC AUC")]
    res.to_csv(f"../results/AutoMHS-GPT({_gpt_version}).csv")
