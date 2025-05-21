from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.random_forest.random_forest_conf import RandomForestConf
from pylibs.uts_models.benchmark_models.random_forest.random_forest_model import RandomForestModel
from pylibs.uts_dataset.dataset_loader import DatasetLoader, DataDEMOKPI
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper

window_size = 64

dataset = "Daphnet"
data_id = "S01R02E0.test.csv@6.out"
# data_id = "S03R03E4.test.csv@6.out"
dl = DatasetLoader(dataset, data_id,
                   window_size=window_size,
                   is_include_anomaly_window=True,
                   anomaly_window_type="all")
train_x, train_y = dl.get_sliding_windows()

modelName = 'IForest'
conf = RandomForestConf()
clf = RandomForestModel(conf)
clf.fit(train_x, train_y)
score = clf.score(train_x)
# Post-processing
score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

# uv.plot_x_label_score_row2(train_x[:, -1], train_y, score)

metrics = UTSMetricHelper.get_metrics_all(train_y, score, window_size=window_size)
metrics.update(
    {
        "dataset": dataset,
        "data_id": data_id
    }
)

uv = UnivariateTimeSeriesView(dataset_name=dataset, name=data_id, is_save_fig=True)
uv.plot_x_label_score_metrics_row2(train_x[:, -1], train_y, score, metrics)
print(metrics)
