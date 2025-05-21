from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018
from pylibs.dataset.UnivariateTSFake import generate_fake_period_anomaly_unclear
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.vae.vae_confi import VAEConf
from timeseries_models.vae.vae_model import VAEModel

kid = generate_fake_period_anomaly_unclear()
conf = VAEConf()
conf.latent_dim = 5
conf.window_size = 20
conf.epoch = 20
conf.n_neurons_layer3 = 128
conf.n_neurons_layer2 = 64
conf.n_neurons_layer1 = 32
conf.kl_weight = 500
da = DatasetAIOps2018(kpi_id=kid,
                      windows_size=conf.window_size,
                      is_include_anomaly_windows=False,
                      sampling_rate=1,
                      valid_rate="0.2",
                      anomaly_window_type="all")
train_dataloader, valid_dataloader, test_dataloader = da.get_pydl_windows_3splits_with_origin_label()
vae = VAEModel(conf)
vae.fit(train_dataloader)
score = vae.score(test_dataloader)
vae.report_metrics(valid_dataloader, test_dataloader)
av = UnivariateTimeSeriesView()
av.plot_kpi_with_anomaly_score_row2_with_best_threshold_torch_data_loader(test_dataloader, score)
