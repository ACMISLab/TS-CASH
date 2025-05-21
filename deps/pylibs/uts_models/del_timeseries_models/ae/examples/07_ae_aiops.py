from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018
from pylibs.dataset.UnivariateTSFake import generate_fake_liner_anomaly_unclear
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.ae.ae_confi import AEConf
from timeseries_models.ae.ae_model import AEModel

av = UnivariateTimeSeriesView()
kid = generate_fake_liner_anomaly_unclear()
hps = {'window_size': 98.29372677316161, 'batch_size': 99.57330401987517, 'learning_rate': 0.01002269772896124,
       'dropout': 0.09194826217964122, 'epoch': 47.99751277475024, 'latent_dim': 19.979246118221596,
       'n_neurons_layer1': 65.4764202433906, 'n_neurons_layer2': 113.23011842524, 'n_neurons_layer3': 17.09960434187699}
conf = AEConf()
conf.update_parameters(hps)
da = DatasetAIOps2018(kpi_id=DatasetAIOps2018.KPI_IDS.D24,
                      windows_size=conf.window_size,
                      is_include_anomaly_windows=False,
                      sampling_rate=1,
                      valid_rate="0.5",
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
