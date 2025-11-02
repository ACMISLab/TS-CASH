from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.model_utils import get_all_models, load_model, ExpConf
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.uts_dataset.dataset_loader import DatasetLoader
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

models = get_all_models()
model_name="vae"
econf = ExpConf(
    model_name=model_name,
    dataset_name="DEBUG",
    data_id="period_test.out",
    epochs=64,
    anomaly_window_type="all",
    window_size=32)

dl = DatasetLoader(econf.dataset_name, econf.data_id, test_rate=0.3, anomaly_window_type="coca",
                   window_size=econf.window_size)
train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test()

clf = load_model(econf)
clf.fit(train_x, train_y)
score = clf.score(test_x)

# Post-processing
score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

uv = UnivariateTimeSeriesView(name=econf.model_name, dataset_name=econf.dataset_name, dataset_id=econf.data_id,is_save_fig=True,conf=econf)
# uv.plot_x_label_score_row2(train_x[:, -1], train_y, score)

metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=60)
uv.plot_x_label_score_metrics_row2(test_x[:, -1], test_y, score, metrics)
print(metrics)
