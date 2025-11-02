# 总结
# - 下文中的性能指验证集上和测试集上的性能, 如 0.72 and 0.66 指验证集上的性能是 0.72, 测试集上的性能是 0.66
# - 当训练的 windows <2 时, 在验证集和测试集上的性能 (valid and test f1) 都是 0
# - 当训练的 windows =3 时, 模型的性能有明显的变化 0.72 and 0.66
# - 当训练的 windows=300 时, 模型的性能是 0.64 and 0.61
# - 当使用所有数据集时, 模型的性能是 0.71	and 0.63
# - 感觉模型性能和窗口长度关系更大
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from pylibs.utils.util_numpy import enable_numpy_reproduce
from timeseries_models.isolation_forest.isolation_forest_model import IsolationForestModel

# TRIALCMD_MODEL	DATA_SAMPLE_RATE	window_size	n_estimators	DATA_ID	METRIC_VALID_AFFILIATION_F1	METRIC_TEST_AFFILIATION_F1
# IsolationForest	1	12.66	319.29	42d6616d-c9c5-370a-a8ba-17ead74f3114	0.73	0.64
# IsolationForest	1	13.01	427.64	431a8542-c468-3988-a508-3afd06a218da	0.94	0.92
# IsolationForest	1	25.62	51.91	6d1114ae-be04-3c46-b5aa-be1a003a57cd	0.59	0.54
# IsolationForest	1	25.62	51.91	6efa3a07-4544-34a0-b921-a155bd1a05e8	0.6	    0.65
# IsolationForest	1	181.01	38.35	9c639a46-34c8-39bc-aaf0-9144b37adfc8	0.79	0.66
# IsolationForest	1	47.42	63.63	a8c06b47-cc41-3738-9110-12df0ee4c721	0.99	0.81
# IsolationForest	1	181.01	38.35	c02607e8-7399-3dde-9d28-8a8da5e5d251	0.86	0.49
# IsolationForest	1	204.35	274.56	e0747cad-8dc8-38a9-a9ab-855b61f5551d	0.72	0.47
# IsolationForest	1	13.01	427.64	f0932edd-6400-3e63-9559-0a9860a1baa9	0.78	0.75


# reproduce the experiment
# IsolationForest	1	12.66	319.29	42d6616d-c9c5-370a-a8ba-17ead74f3114	0.73	0.64
enable_numpy_reproduce(1)
ifc = IsolationForestConf()
ifc.update_parameters({
    "window_size": 12.66,
    "n_estimators": 319.29
})

da = DatasetAIOps2018(kpi_id=AIOpsKPIID.D5,
                      windows_size=ifc.window_size,
                      is_include_anomaly_windows=True,
                      sampling_rate=1,
                      valid_rate=0.5)
train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()

# 1 windows
# valid and test f1: 0.0 and 0.0
iforest_small_data = IsolationForestModel(ifc)
iforest_small_data.fit(train_x[0:1, :])
iforest_small_data.report_metrics(valid_x, valid_y, test_x, test_y)

# 2 windows
# valid and test f1: 0 and 0
iforest_small_data = IsolationForestModel(ifc)
iforest_small_data.fit(train_x[0:2, :])
iforest_small_data.report_metrics(valid_x, valid_y, test_x, test_y)

# 3 windows
# valid and test f1: 0.72 and 0.66
iforest_small_data = IsolationForestModel(ifc)
iforest_small_data.fit(train_x[0:3, :])
iforest_small_data.report_metrics(valid_x, valid_y, test_x, test_y)

# 300 windows
# valid and test f1: 0.64 and 0.61
iforest_small_data = IsolationForestModel(ifc)
iforest_small_data.fit(train_x[0:300, :])
iforest_small_data.report_metrics(valid_x, valid_y, test_x, test_y)

# All data
iforest_best = IsolationForestModel(ifc)
iforest_best.fit(train_x)
iforest_best.report_metrics(valid_x, valid_y, test_x, test_y)
# valid and test f1: 0.73	0.64
# reproduce: valid and test: 0.71	0.63
