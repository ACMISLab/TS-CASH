from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.isolation_forest.isolation_forest_conf import IsolationForestConf
from timeseries_models.isolation_forest.isolation_forest_model import IsolationForestModel

window_size = 10
da = DatasetAIOps2018(kpi_id=DatasetAIOps2018.KPI_IDS.TEST_PERIOD_OBVIOUS,
                      windows_size=window_size,
                      is_include_anomaly_windows=True,
                      valid_rate=0.5)

train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()
iforest = IsolationForestModel(IsolationForestConf(n_estimators=100))
iforest.fit(train_x)
iforest.report_metrics(valid_x, valid_y, test_x, test_y)
score = iforest.score(test_x)
uv = UnivariateTimeSeriesView()
uv.plot_kpi_with_anomaly_score_row2_with_best_threshold_numpy(test_x[:, -1], test_y, score)
