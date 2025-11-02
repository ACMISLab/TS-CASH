from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.pca.pca import PCA
from pylibs.uts_dataset.dataset_loader import DatasetLoader, DataDEMOKPI, KFoldDatasetLoader
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper

window_size = 99
dataset, data_id = ["ECG", "MBA_ECG805_data.out"]
dl = KFoldDatasetLoader(dataset, data_id, window_size=window_size, is_include_anomaly_window=False, max_length=10000,
                        sample_rate=1)
train_x, train_y, test_x, test_y = dl.get_kfold_sliding_windows_train_and_test_by_fold_index(1)

clf = PCA()
clf.fit(train_x)
score = clf.score(test_x)

# Post-processing
score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

uv = UnivariateTimeSeriesView(name=data_id, is_save_fig=True)
# uv.plot_x_label_score_row2(train_x[:, -1], train_y, score)

metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=window_size)
uv.plot_x_label_score_metrics_row2(test_x[:, -1], test_y, score, metrics)
print(metrics)
