from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.coca.coca_factory import COCAFactory

enable_numpy_reproduce(1)

for _kid in [AIOpsKPIID.TEST_LINE, AIOpsKPIID.TEST_LINE_VAGUE, AIOpsKPIID.TEST_PERIOD_VAGUE,
             AIOpsKPIID.TEST_PERIOD_OBVIOUS, AIOpsKPIID.D27]:
    coca = COCAFactory(None, "cpu").get_model()
    da = DatasetAIOps2018(kpi_id=_kid,
                          windows_size=coca.config.window_size,
                          is_include_anomaly_windows=False,
                          sampling_rate=1,
                          valid_rate="0.2",
                          anomaly_window_type="coca"
                          )
    train_dataloader, valid_dataloader, test_dataloader = da.get_pydl_windows_3splits_with_origin_label_coca(
        coca.config.batch_size)
    coca.fit(train_dataloader)
    score = coca.score(test_dataloader)
    coca.report_metrics(valid_dataloader, test_dataloader)
    av = UnivariateTimeSeriesView(_kid)
    av.plot_kpi_with_anomaly_score_row2_with_best_threshold_torch_data_loader(test_dataloader, score)
