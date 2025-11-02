import json
import re

import pandas as pd

from sota.auto_cash.auto_cash_helper import KVDB, get_model_args_from_dict_by_model_name, get_model_args_from_dict
from tshpo.automl_libs import get_auto_sklearn_classification_search_space

kvdb = KVDB("authmhs_gpt.dump", "./")


def gpt_response_to_dict(suggest_hpy):
    """suggest_hpy= ```json
    {
        "model": "random_forest",
        "hyperparameters": {
            "bootstrap": true,
            "criterion": "gini",
            "max_depth": 15,
            "max_features": "sqrt",
            "max_leaf_nodes": null,
            "min_impurity_decrease": 1e-7,
            "min_samples_leaf": 3,
            "min_samples_split": 5,
            "min_weight_fraction_leaf": 0.0
            }
    }
    ```
    """
    # 使用正则表达式匹配 {} 之间的内容
    pattern = r'\{[\s\S]*\}'  # 非贪婪匹配
    matches = re.findall(pattern, suggest_hpy)

    if matches:
        json_str = matches[0]  # 提取第一个匹配项
        try:
            replace_dict = {
                "true": "True",
                "false": "False",
                "null": "0"
            }
            for k, v in replace_dict.items():
                json_str = json_str.replace(k, v)
            return eval(json_str)
        except json.JSONDecodeError as e:
            print("JSON解析失败:", e)
    else:
        print("未找到JSON内容")


statics = []
kvdb_processed = KVDB("authmhs_gpt_processed.dump", "./")
for key in kvdb.keys():
    # print(key)
    # key: kr-vs-kp:::gpt-4-turbo
    dataset, gpt_version = key.split(":::")
    v = kvdb.query(key)
    suggest_hpy_str = v['suggest']

    suggest_hpy_dict = gpt_response_to_dict(suggest_hpy_str)
    suggest_model_name = suggest_hpy_dict['model']
    suggest_model_hpy = suggest_hpy_dict['hyperparameters']

    print(dataset, gpt_version, suggest_model_name, suggest_model_hpy)
    cs = get_auto_sklearn_classification_search_space(y_train=[0, 1], include=[suggest_model_name])
    default_hpy = cs.sample_configuration()
    default_hpy_dict = dict(default_hpy)
    default_hpy_dict_processed = get_model_args_from_dict(default_hpy_dict)

    # 使用AutoCMS-GPT的结果共享
    important_hpys = []
    for k, v in suggest_model_hpy.items():
        if k in default_hpy_dict_processed.keys():
            # print(f"update: {k}:{v}")
            default_hpy_dict_processed.update({k: v})
            important_hpys.append(k)
    statics.append({
        "dataset": dataset,
        "gpt_version": gpt_version,
        "suggest_model_name": suggest_model_name,
        "suggest_model_hpy": default_hpy_dict_processed,
    })
    kvdb_processed.add(key, {
        "suggest_model_name": suggest_model_name,
        "suggest_model_hpy": default_hpy_dict_processed,
        "import_hpys": important_hpys
    })
pd.DataFrame(statics).to_csv("statics.csv")
