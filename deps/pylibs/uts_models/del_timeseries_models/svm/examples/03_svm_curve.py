from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.isolation_forest.examples.isolation_aiops_lib import get_best_conf
from timeseries_models.random_forest.random_forest_conf import RandomForestConf
from timeseries_models.random_forest.random_forest_model import RandomForestModel
from timeseries_models.svm.svm_conf import SVMConf
from timeseries_models.svm.svm_model import SVMModel

enable_numpy_reproduce(1)
#
rfc = SVMConf()
da = DatasetAIOps2018(kpi_id=AIOpsKPIID.TEST_PERIOD_OBVIOUS,
                      windows_size=rfc.window_size,
                      is_include_anomaly_windows=True,
                      sampling_rate=1,
                      valid_rate="0.2")
train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()
conf = get_best_conf()
iforest_best = SVMModel(rfc)
iforest_best.fit(train_x, train_y)
score = iforest_best.score(train_x)
iforest_best.report_metrics(valid_x, valid_y, test_x, test_y, DateTimeHelper())
av = UnivariateTimeSeriesView()
av.plot_kpi_with_anomaly_score_row2_with_best_threshold_numpy(train_x[:, -1], train_y, score)
