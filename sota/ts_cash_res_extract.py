import os

import numpy as np
import pandas as pd

tshpo_file = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/tshpo_accuracy_and_rocauc.csv"
df = pd.read_csv(tshpo_file)

df['value'] = df.apply(lambda x: f"{np.round(x['mean'], 4)}({np.round(x['std'], 4)})", axis=1)


def myfun(x):
    return x


res = pd.pivot_table(df, index="dataset", columns=['metric'], values='value', aggfunc=myfun)
res.columns = [("TS-CASH", "Accuracy"), ("TS-CASH", "ROC AUC")]
res.to_csv("./results/TS-CASH.csv")
