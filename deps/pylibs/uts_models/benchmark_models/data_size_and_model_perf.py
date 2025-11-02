import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pylibs._del_dir.experiments.exp_config import ExpConf
from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_pandas import PDUtil
from pylibs.utils.util_picture import UtilPic
from pylibs.uts_dataset.dataset_loader import get_debug_datasize

log = get_logger()
window_size = 64
columns = ["model_name", "train_length", "train_x.mean", "train_y.mean", "test_x.mean", "test_y.mean",
           "parameters.mean", 'score.mean']
out_metrics = []

for _model_name in ["ocsvm", "pca", "iforest"]:

    conf = ExpConf(model_name=_model_name, window_size=window_size)
    is_include_anomaly_window = True
    if conf.is_semi_supervised_model():
        is_include_anomaly_window = False
    for train_x, train_y, test_x, test_y in get_debug_datasize(window_size, is_include_anomaly_window):
        _clf = load_model(conf)
        _clf.fit(train_x)
        score = _clf.score(test_x)
        out_metrics.append(
            [_model_name, train_x.shape[0], train_x.mean(), train_y.mean(), test_x.mean(), test_y.mean(),
             _clf.get_parameters_desc(), score.mean()])
        # Post-processing
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        from pylibs.utils.util_univariate_time_series_view import UTSPlot

        metrics = UTSMetricHelper.get_metrics_all(test_y, score, window_size=window_size)

        uv = UTSPlot(filename=f"model_{_model_name}_train_len_{train_x.shape[0]}_test_len_{test_x.shape[0]}",
                     is_save_fig=True)
        uv.plot_x_label_score_metrics_row2(test_x[:, -1], test_y, score, metrics)
        print(metrics)
        # UtilPic.merge_picture(UtilComm.get_runtime_directory(), "summ.pdf")

df = pd.DataFrame(out_metrics, columns=columns)
PDUtil.save_to_excel(df, 'model_hyperparameters_vs_datasize')
UtilPic.merge_picture(UtilComm.get_runtime_directory(), "summary.pdf")
