from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.vae.vae import VAEModel
from pylibs.uts_models.benchmark_models.vae.vae_conf import VAEConfig
from pylibs.uts_dataset.dataset_loader import DatasetLoader
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

window_size = 99
data_id = "NAB_data_CloudWatch_5.out"
dl = DatasetLoader("NAB", data_id, window_size=window_size, is_include_anomaly_window=False)
train_x, train_y, origin_train_x, origin_train_y = dl.get_sliding_windows(return_origin=True)
conf = VAEConfig()
conf.window_size = window_size
conf.epochs = 10
clf = VAEModel(conf)
clf.fit(train_x)
score = clf.score(origin_train_x)

# Post-processing
score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

uv = UnivariateTimeSeriesView(name=data_id, is_save_fig=True)

metrics = UTSMetricHelper.get_metrics_all(origin_train_y, score, window_size=window_size)
uv.plot_x_label_score_metrics_row2(origin_train_x[:, -1], origin_train_y, score, metrics)
print(metrics)
