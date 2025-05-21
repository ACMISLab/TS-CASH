import os

import pandas as pd

tshpo_file = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/tshpo_accuracy_and_rocauc.csv"
automhs_gpt_data_home = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/automhs_gpt/automhs_gpt_perf"
df_tshpo = pd.read_csv(tshpo_file)

autochms = pd.concat(
    [pd.read_csv(os.path.join(automhs_gpt_data_home, f), index_col=0) for f in os.listdir(automhs_gpt_data_home) if
     f.endswith(".csv")])
auto_chms_df = autochms.groupby(by=['dataset', "gpt_version", 'metric'])['value'].agg(['mean', 'std']).reset_index()
print(autochms)

datasets = auto_chms_df.dataset.unique()
metrics = auto_chms_df.metric.unique()
gpt_versions = auto_chms_df.gpt_version.unique()
results = []
for _gpt_version in gpt_versions:
    for _dataset in datasets:
        for _metric in metrics:
            tshpo = df_tshpo.loc[(df_tshpo.dataset == _dataset) & (df_tshpo.metric == _metric), ['mean']].values[0][0]
            autochms = \
                auto_chms_df.loc[
                    (auto_chms_df.dataset == _dataset) & (auto_chms_df.metric == _metric) & (
                            auto_chms_df.gpt_version == _gpt_version), ['mean']].values[0][
                    0]
            # dataset,metric,mean,std,name
            results.append(
                {
                    "dataset": _dataset,
                    "metric": _metric,
                    "name": f"AutoMHS-GPT({_gpt_version})",
                    "tshpo": tshpo,
                    "sota": autochms,
                    "improved": tshpo - autochms
                }
            )
    os.makedirs("../results", exist_ok=True)
    pd.DataFrame(results).to_csv(f"../results/automhs_gpt_{_gpt_version}_vs_tshpo.csv")
