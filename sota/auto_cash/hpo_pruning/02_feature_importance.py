# 随机森林回归
import pickle
import traceback

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sota.auto_cash.df_helper import DFHelper

select_features = {}
datafile = "../meta_datas/all_data.pkl"
all_data = pickle.load(open(datafile, "rb"))
for key in all_data.keys():
    dataset, model_name = key.split(":::")

    data = all_data[key]
    df = pd.DataFrame(data)

    x_train, y_train = DFHelper.pre_process_df(df)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    try:
        model.fit(x_train, y_train)
        # # 特征重要性
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=x_train.columns).sort_values(ascending=False)
        theta = feature_importance.max() * 0.3
        # Potential priority
        lambda_list = list(feature_importance[feature_importance > theta].keys())

        select_features[key] = lambda_list
    except:
        select_features[key] = {}

pickle.dump(select_features, open("../meta_datas/select_features.pkl", "wb"))
