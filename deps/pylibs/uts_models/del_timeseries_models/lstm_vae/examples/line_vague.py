from torch.optim import Adam

from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from pylibs.dataset.UnivariateTSFake import generate_fake_period_anomaly_unclear
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.ae.ae_confi import AEConf
from timeseries_models.ae.ae_model import AEModel

from timeseries_models.lstm_vae.lstm_model import LSTM_VAE
from timeseries_models.lstm_vae.lstm_config import LSTMConf

av = UnivariateTimeSeriesView()
kid = generate_fake_period_anomaly_unclear()
conf = LSTMConf()
conf.lstm_layers = 3
conf.epochs = 20
da = DatasetAIOps2018(kpi_id=kid,
                      windows_size=conf.window_size,
                      is_include_anomaly_windows=False,
                      sampling_rate=1,
                      valid_rate="0.2",
                      anomaly_window_type="all")
train_dataloader, valid_dataloader, test_dataloader = \
    da.get_pydl_windows_3splits_with_origin_label(conf.batch_size)
ae = LSTM_VAE(conf.window_size, conf.lstm_layers, conf.rnn_hidden_size,
              conf.latent_size, 1)
optimizer = Adam(ae.parameters(), lr=0.001)
ae.fit(optimizer, 1, train_dataloader)
score = ae.score(test_dataloader)
av = UnivariateTimeSeriesView()
av.plot_kpi_with_anomaly_score_row2(test_dataloader.dataset.data[:, -1], test_dataloader.dataset.label,
                                    score[conf.window_size - 1:])
