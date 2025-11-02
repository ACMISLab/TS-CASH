from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018
from pylibs.dataset.UnivariateTSFake import generate_fake_liner_anomaly_unclear
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.ae.ae_confi import AEConf
from timeseries_models.ae.ae_model import AEModel

av = UnivariateTimeSeriesView()
kid = generate_fake_liner_anomaly_unclear()
conf = AEConf()
conf.latent_dim = 8
conf.window_size = 32
conf.epoch = 30
conf.n_neurons_layer3 = 256
conf.n_neurons_layer2 = 128
conf.n_neurons_layer1 = 64
da = DatasetAIOps2018(kpi_id=kid,
                      windows_size=conf.window_size,
                      is_include_anomaly_windows=False,
                      sampling_rate=1,
                      valid_rate="0.2",
                      anomaly_window_type="all")
train_dataloader, valid_dataloader, test_dataloader = da.get_pydl_windows_3splits_with_origin_label(conf.batch_size)
vae = AEModel(conf)
vae.fit(train_dataloader)
x = test_dataloader.dataset.data[:, -1]
label = test_dataloader.dataset.label
x_bar = vae.reconstruct_x(test_dataloader)
score = vae.score(test_dataloader)
vae.report_metrics(valid_dataloader, test_dataloader)
av.plot_kpi_with_anomaly_score_row2_with_best_threshold_torch_data_loader(test_dataloader, score)
