from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.decision_tree.decision_tree import DecisionTree
from pylibs.uts_dataset.dataset_loader import DatasetLoader, DataDEMOKPI
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper

i = 99
window_size = i
for dataset, data_id in DataDEMOKPI.DEMO_KPIS:
    print(">" * 30)
    dl = DatasetLoader(dataset, data_id, window_size=window_size, max_length=10000)
    train_x, train_y = dl.get_sliding_windows()

    clf = DecisionTree()
    clf.fit(train_x, train_y)
    score = clf.score(train_x)

    # Post-processing
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

    uv = UnivariateTimeSeriesView(name=data_id, is_save_fig=True)
    # uv.plot_x_label_score_row2(train_x[:, -1], train_y, score)

    metrics = UTSMetricHelper.get_metrics_all(train_y, score, window_size=window_size)
    uv.plot_x_label_score_metrics_row2(train_x[:, -1], train_y, score, metrics)
    print(metrics)
