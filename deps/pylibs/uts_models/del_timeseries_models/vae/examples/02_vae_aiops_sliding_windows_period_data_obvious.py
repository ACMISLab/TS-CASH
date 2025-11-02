
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.vae.vae_confi import VAEConf
from timeseries_models.vae.vae_model import VAEModel

enable_numpy_reproduce(1)
#
conf = VAEConf()
da = DatasetAIOps2018(kpi_id=AIOpsKPIID.TEST_PERIOD_OBVIOUS,
                      windows_size=conf.window_size,
                      is_include_anomaly_windows=False,
                      sampling_rate=1,
                      valid_rate="0.2")
train_dataloader, valid_dataloader, test_dataloader = da.get_pydl_windows_3splits_with_origin_label(conf.batch_size)
vae = VAEModel(conf)
vae.fit(train_dataloader)
vae.report_metrics(valid_dataloader, test_dataloader)

score = vae.score(test_dataloader)
av = UnivariateTimeSeriesView()
av.plot_kpi_with_anomaly_score_row2_with_best_threshold_torch_data_loader(test_dataloader, score)
