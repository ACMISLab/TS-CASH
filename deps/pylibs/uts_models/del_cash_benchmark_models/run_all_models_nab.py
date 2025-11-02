from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.model_utils import get_all_models, load_model, ExpConf
from pylibs.metrics.uts.uts_metric_helper import UTSMetricHelper
from pylibs.uts_dataset.dataset_loader import DatasetLoader, DataDEMOKPI
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

test_kpis = [
["NAB","NAB_data_Traffic_3.out"],
["NAB","NAB_data_Traffic_6.out"],
["NAB","NAB_data_CloudWatch_14.out"],
["NAB","NAB_data_tweets_2.out"],
["NAB","NAB_data_CloudWatch_3.out"],
["NAB","NAB_data_CloudWatch_10.out"],
["NAB","NAB_data_tweets_7.out"],
["NAB","NAB_data_CloudWatch_13.out"],
["NAB","NAB_data_CloudWatch_12.out"],
["NAB","NAB_data_tweets_4.out"],
["NAB","NAB_data_KnownCause_4.out"],
["NAB","NAB_data_tweets_9.out"],
["NAB","NAB_data_tweets_8.out"],
["NAB","NAB_data_KnownCause_5.out"],
["NAB","NAB_data_art1_5.out"],
["NAB","NAB_data_KnownCause_6.out"],
["NAB","NAB_data_art1_2.out"],
["NAB","NAB_data_KnownCause_3.out"],
["NAB","NAB_data_art1_3.out"],
["NAB","NAB_data_art1_1.out"],
["NAB","NAB_data_KnownCause_1.out"],
["NAB","NAB_data_art1_0.out"],
["NAB","NAB_data_Exchange_6.out"],
["NAB","NAB_data_Exchange_5.out"],
["NAB","NAB_data_Exchange_4.out"],

]

test_kpis=test_kpis[:5]
models = get_all_models()
for model_name in models:
    for dataset_name, data_id in test_kpis:
        econf = ExpConf(
            model_name=model_name,
            dataset_name=dataset_name,
            data_id=data_id,
            epochs=8,
            window_size=32)

        dl = DatasetLoader(econf.dataset_name, econf.data_id, test_rate=0.3, anomaly_window_type="coca",
                           window_size=econf.window_size)
        train_x, train_y, test_x, test_y = dl.get_sliding_windows_train_and_test()

        clf = load_model(econf)
        clf.fit(train_x, train_y)
        score = clf.score(test_x)

        # Post-processing
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

        uv = UnivariateTimeSeriesView(name=econf.model_name, dataset_name=econf.dataset_name, dataset_id=econf.data_id,
                                      is_save_fig=True)
        # uv.plot_x_label_score_row2(train_x[:, -1], train_y, score)

        metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=60)
        uv.plot_x_label_score_metrics_row2(test_x[:, -1], test_y, score, metrics)
        print(metrics)
