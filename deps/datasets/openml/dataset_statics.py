# ID2024102120171781367947
import json

import numpy as np
import pandas as pd
from tshpo.automl_libs import load_dataset_at_fold
from tshpo.lib_class import AnaHelper

# from automl_libs import load_dataset_at_fold, AnaHelper

with open("binary_classification_maps.json", "r") as f:
    datasets = json.load(f)


def data_type(size):
    if size > 10000:
        return "large"
    elif size > 2000:
        return "medium"
    else:
        return "small"


stats = []
for _index, (task_id, dataset_name) in enumerate(datasets.items()):
    # X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=dataset_name, fold_index=0)

    X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=dataset_name, n_fold=10,
                                                            fold_index=0, seed=42)
    n_instance = X_train.shape[0] + y_train.shape[0]
    positive_label_count = y_train.sum() + y_test.sum()
    labels_count = y_train.shape[0] + y_test.shape[0]
    if dataset_name in AnaHelper.SELECTED_DATASET:
        stats.append({
            "task_id": task_id,
            "dataset_name": dataset_name,
            "#instances": n_instance,
            "#Features": X_test.shape[1],
            "group": data_type(n_instance),
            "x_shape": X_train.shape,
            "Y_train_unique": np.unique(y_train),
            "Y_test_unique": np.unique(y_test),
            "positive_label": positive_label_count,
            "negative_label": labels_count - positive_label_count,
            "positive_rate": round(positive_label_count / labels_count, 2)
        })

df = pd.DataFrame(stats)
df = df.sort_values(by=['#instances'], ascending=True)
df.to_csv("stats.csv", index=False)
filter_df = df.loc[:, ["group", "task_id", "dataset_name", '#instances', "#Features", "positive_rate"]]
filter_df['task_id'] = filter_df['task_id'].astype(int)
filter_df = filter_df.sort_values(by=['group', 'task_id'], ascending=True).reset_index(drop=True).reset_index()
filter_df['index'] = filter_df['index'] + 1
print(filter_df.to_latex("data_stats.tex", index=False))
