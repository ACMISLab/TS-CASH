# todo: this code may be error.
from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.vae.vae import VAEModel
from pylibs.uts_models.benchmark_models.vae.vae_conf import VAEConfig
from pylibs.uts_dataset.dataset_loader import DatasetLoader, DataDEMOKPI
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

window_size = 99
max_length = 10000
for dataset, data_id in DataDEMOKPI.DEMO_KPIS:
    # 获取训练数据，只有异常

    dl_train = DatasetLoader(dataset, data_id, window_size=window_size,
                             is_include_anomaly_window=False,
                             max_length=max_length)
    train_x, train_y = dl_train.get_sliding_windows()

    # 获取测试数据，包含异常和正常
    dl_test = DatasetLoader(dataset, data_id, window_size=window_size,
                            is_include_anomaly_window=True,
                            max_length=max_length)
    test_x, test_y = dl_test.get_sliding_windows()
    conf = VAEConfig()
    conf.window_size = window_size

    clf = VAEModel(conf)
    clf.fit(train_x)
    score = clf.score(test_x)

    # Post-processing
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

    uv = UnivariateTimeSeriesView(name=data_id, is_save_fig=True)
    # uv.plot_x_label_score_row2(train_x[:, -1], train_y, score)
    metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=window_size)
    uv.plot_x_label_score_metrics_row2(test_x[:, -1], test_y, score, metrics)
    print(metrics)
