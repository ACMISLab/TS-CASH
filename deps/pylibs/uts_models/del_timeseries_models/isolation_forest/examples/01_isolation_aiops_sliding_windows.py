from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.isolation_forest.examples.isolation_aiops_lib import get_best_conf
from timeseries_models.isolation_forest.isolation_forest_model import IsolationForestModel

enable_numpy_reproduce(1)
#

da = DatasetAIOps2018(kpi_id=AIOpsKPIID.D7,
                      windows_size=get_best_conf().window_size,
                      is_include_anomaly_windows=True,
                      sampling_rate=1,
                      valid_rate="0.2")
train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_three_splits()
conf = get_best_conf()
iforest_best = IsolationForestModel(conf)
iforest_best.fit(train_x)
score = iforest_best.score(train_x)
iforest_best.report_metrics(valid_x, valid_y, test_x, test_y, DateTimeHelper())
av = UnivariateTimeSeriesView()
av.plot_kpi_with_anomaly_score_row2_with_best_threshold_numpy(train_x, train_y, score)
