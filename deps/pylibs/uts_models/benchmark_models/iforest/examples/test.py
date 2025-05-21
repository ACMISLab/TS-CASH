from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.iforest.iforest import IForest
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.uts_dataset.dataset_loader import DatasetLoader
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

dl = DatasetLoader("DEBUG", "period_test.out", test_rate=0.3, anomaly_window_type="coca", window_size=30)
train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test()
model=IForest()
model.fit(train_x)
score=model.score(test_x)

# Post-processing
score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()


uv = UnivariateTimeSeriesView(name="period_test.out", is_save_fig=True)
# uv.plot_x_label_score_row2(train_x[:, -1], train_y, score)

metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=60)
uv.plot_x_label_score_metrics_row2(test_x[:, -1], test_y, score, metrics)
print(metrics)