from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.one_class_svm.one_svm_conf import OneClassSVMConf
from timeseries_models.one_class_svm.one_svm_model import OneClassSVMModel

enable_numpy_reproduce(1)
for _kid in [AIOpsKPIID.TEST_LINE, AIOpsKPIID.TEST_LINE_VAGUE, AIOpsKPIID.TEST_PERIOD_VAGUE,
             AIOpsKPIID.TEST_PERIOD_OBVIOUS, AIOpsKPIID.D27]:
    rfc = OneClassSVMConf()
    da = DatasetAIOps2018(kpi_id=_kid,
                          windows_size=rfc.window_size,
                          is_include_anomaly_windows=True,
                          sampling_rate=1,
                          valid_rate="0.2")
    train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()
    iforest_best = OneClassSVMModel(rfc)
    iforest_best.fit(train_x, train_y)
    score = iforest_best.score(train_x)
    iforest_best.report_metrics(valid_x, valid_y, test_x, test_y, DateTimeHelper())
    av = UnivariateTimeSeriesView(name=f"one_class_svm_{_kid}")
    av.plot_kpi_with_anomaly_score_row2_with_best_threshold_numpy(train_x[:, -1], train_y, score)
    print("\n" * 3)
