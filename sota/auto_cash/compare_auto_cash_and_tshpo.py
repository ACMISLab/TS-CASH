import os

import pandas as pd

tshpo_file = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/tshpo_accuracy_and_rocauc.csv"
auto_cash_metric_dir = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/auto_cash/auto_cash_perf"
df_tshpo = pd.read_csv(tshpo_file)

auto_cash = pd.concat(
    [pd.read_csv(os.path.join(auto_cash_metric_dir, f), index_col=0) for f in os.listdir(auto_cash_metric_dir) if
     f.endswith(".csv")])
auto_cash_df = auto_cash.groupby(by=['dataset', 'metric'])['value'].agg(['mean', 'std']).reset_index()
print(auto_cash)

datasets = auto_cash_df.dataset.unique()
metrics = auto_cash_df.metric.unique()
results = []

for _dataset in datasets:
    for _metric in metrics:
        tshpo = df_tshpo.loc[(df_tshpo.dataset == _dataset) & (df_tshpo.metric == _metric), ['mean']].values[0][0]
        auto_cash = \
            auto_cash_df.loc[(auto_cash_df.dataset == _dataset) & (auto_cash_df.metric == _metric), ['mean']].values[0][
                0]

        results.append(
            {
                "dataset": _dataset,
                "metric": _metric,
                "tshpo": tshpo,
                "name": "Auto-CASH",
                "sota": auto_cash,
                "improved": tshpo - auto_cash
            }
        )
os.makedirs("../results", exist_ok=True)
pd.DataFrame(results).to_csv("auto_cash_vs_tshpo.csv")
