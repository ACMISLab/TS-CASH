import pandas as pd

import os
from pathlib import Path

# 数据文件路径
DATA_HOME = Path("./results")

# 读取所有 CSV 文件并合并
DATA_LIST = [
    "TS-CASH.csv",
    "Auto-CASH.csv",
    # "AutoMHS-GPT(gpt-4-turbo).csv",
    "AutoMHS-GPT(gpt-4o).csv",
    # "AutoMHS-GPT(gpt-4o-mini).csv",
    "BO.csv",
    "HB.csv",
    "RS.csv",
]
df_list = []

# 遍历目录中的所有 CSV 文件
for file in DATA_LIST:
    if file.endswith(".csv"):
        # 读取 CSV 文件
        temp_df = pd.read_csv(DATA_HOME / file)

        # 确保 `dataset` 列是唯一的索引
        temp_df.set_index("dataset", inplace=True)

        # 将每个文件的数据加入列表
        df_list.append(temp_df)

# 按 `dataset` 合并所有数据，生成多列
df = pd.concat(df_list, axis=1)

# 重置索引（可选）
df.reset_index(inplace=True)

for _type_metric in ["Accuracy", "ROC AUC"]:
    _select_column = [i for i in df.columns if str(i).find(_type_metric) > -1]
    _select_column.insert(0, "dataset")
    # 打印结果
    # print(df)
    df.to_csv(f"compare_all_{_type_metric}.csv", index=False, columns=_select_column)
    df.to_latex(f"compare_all_{_type_metric}.tex", index=False, columns=_select_column)
