import lightning
import nni
import numpy as np
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning import Trainer, seed_everything
from torch import optim, nn
from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018
from pylibs.nni_report import report_nni_final_metric
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_timeseries_sliding_windows import unroll_ts
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from timeseries_models.ae.ae_confi import AEConf
from timeseries_models.base_model import PyTorchBase, _F1_INDEX, _PRECISION_INDEX, _RECALL_INDEX

log = get_logger()


# define any number of nn.Modules (or use your current ones)


class _Encoder(nn.Module):
    def __init__(self, conf: AEConf):
        super().__init__()
        modules = []
        in_channels = conf.window_size
        encoder_neurons = conf.encoder_neurons.copy()
        encoder_neurons.append(conf.latent_dim)
        for h_dim in encoder_neurons:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=h_dim),
                    nn.ReLU(),
                    nn.Dropout(conf.dropout)
                )
            )
            in_channels = h_dim

        self.l1 = nn.Sequential(*modules)

    def forward(self, x):
        return self.l1(x)


class _Decoder(nn.Module):
    def __init__(self, conf: AEConf):
        super().__init__()
        hidden_size = conf.decoder_neurons
        # Keep the same output size
        hidden_size.append(conf.window_size)
        modules = []
        in_channels = conf.latent_dim
        for h_dim in hidden_size:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=h_dim),
                    nn.ReLU(),
                    nn.Dropout(conf.dropout)
                )
            )
            in_channels = h_dim

        self.l1 = nn.Sequential(*modules)

    def forward(self, x):
        return self.l1(x)


# define the LightningModule
class _AEModel(lightning.pytorch.LightningModule):
    def __init__(self, conf: AEConf):
        super().__init__()
        self.encoder = _Encoder(conf)
        self.decoder = _Decoder(conf)
        self.conf = conf

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", test_loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        z = self.encoder(x)
        # hat
        x_hat = self.decoder(z)

        _score = self._score(x_hat, x)
        return x_hat, _score

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.conf.learning_rate)
        return optimizer

    @staticmethod
    def _score(recon_x, x):
        """
        Calculate score, which is defined (Cited by 902)  in https://link.springer.com/chapter/10.1007/3-540-46145-0_17

        The larger the score, the more likely it is an anomaly.

        Parameters
        ----------

        Returns
        -------

        """
        score = torch.mean(torch.square(recon_x - x), dim=1)
        return score


class AEModel(PyTorchBase):

    def __init__(self, conf: AEConf, device=None, seed=None):
        super().__init__(conf)
        self.conf = conf
        self.device = None
        if seed is None:
            self.seed = 0
        else:
            self.seed = seed

        # seeing https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html
        if device is not None:
            self.device = [device]
        UtilSys.is_debug_mode() and log.info(f"Used  device  {self.device} for vae")
        seed_everything(self.seed, workers=True)
        self.trainer = lightning.pytorch.Trainer(accelerator="gpu",
                                                 devices=self.device,
                                                 enable_progress_bar=True,
                                                 enable_checkpointing=False,
                                                 max_epochs=conf.epoch,
                                                 deterministic=True,
                                                 callbacks=[RichProgressBar(leave=True)])

        self.model = _AEModel(conf)

    def _predict_x_and_score(self, dl):
        predict_arr = self.trainer.predict(self.model, dataloaders=dl)
        predict_x_ = torch.concat([batch_pre[0] for batch_pre in predict_arr], dim=0)
        predict_score_ = torch.concat([batch_pre[1] for batch_pre in predict_arr], dim=0)
        return predict_x_, predict_score_

    def _gather_metrics(self, valid_dl, test_dl, app_dt: DateTimeHelper):
        UtilSys.is_debug_mode() and log.info("Calculating model metrics...")
        app_dt.evaluate_start()
        _valid_score = self.score(valid_dl)
        _test_score = self.score(test_dl)

        val_best_affiliation = self.ad_predict_v3(valid_dl.dataset.label, _valid_score)
        test_best_affiliation = self.ad_predict_v3(test_dl.dataset.label, _test_score)
        app_dt.evaluate_end()
        return {
            'valid_affiliation_f1': val_best_affiliation[_F1_INDEX],
            'valid_affiliation_precision': val_best_affiliation[_PRECISION_INDEX],
            'valid_affiliation_recall': val_best_affiliation[_RECALL_INDEX],
            'test_affiliation_f1': test_best_affiliation[_F1_INDEX],
            'test_affiliation_precision': test_best_affiliation[_PRECISION_INDEX],
            'test_affiliation_recall': test_best_affiliation[_RECALL_INDEX],
        }

    def report_metrics(self, valid_dl, test_dl, app_dt: DateTimeHelper = None):
        # fix: 'AssertionError: nni.get_next_parameter() needs to be called before '
        nni.get_next_parameter()
        if app_dt is None:
            app_dt = DateTimeHelper()
        metrics = self._gather_metrics(valid_dl, test_dl, app_dt)
        if app_dt is not None:
            metrics.update(app_dt.collect_metrics())
        metrics['default'] = metrics.get("test_affiliation_f1")
        # UtilSys.is_debug_mode()  and log.info(pprint.pformat(metrics))
        report_nni_final_metric(**metrics)

    def predict(self, data):
        _predict = self.trainer.predict(self.model, dataloaders=data)
        _predict_x = [batch_pre[0] for batch_pre in _predict]
        return torch.concat(_predict_x, dim=0)

    def fit(self, train_dl):
        self.trainer.fit(model=self.model, train_dataloaders=train_dl)

    def reconstruct_x(self, dl):
        predict_windows = self.predict(dl)
        x_bar = unroll_ts(predict_windows)
        return x_bar

    def score(self, dl):
        x_bar = self.reconstruct_x(dl)
        source_x = dl.dataset.data[:, -1]
        score = np.abs(np.subtract(source_x, x_bar))
        return score


if __name__ == '__main__':
    conf = AEConf()
    da = DatasetAIOps2018(kpi_id=DatasetAIOps2018.KPI_IDS.TEST_01,
                          windows_size=conf.window_size,
                          is_include_anomaly_windows=False,
                          valid_rate=0.5)
    train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()

    # init the autoencoder
    train_dataloader, valid_dataloader, test_dataloader = da.get_sliding_windows_three_splits_pytorch_dataloader(conf)
    autoencoder = _AEModel(conf)

    # Reproducibility
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # trainer = pl.Trainer(limit_train_batches=100, accelerator="gpu", devices=[0], max_epochs=1)
    trainer = lightning.pytorch.Trainer(accelerator="gpu", devices=[0], max_epochs=conf.epoch,
                                        deterministic=True)
    trainer.fit(model=autoencoder, train_dataloaders=train_dataloader)

    tester = Trainer()
    tester.test(autoencoder, dataloaders=valid_dataloader)
    predict = tester.predict(autoencoder, dataloaders=test_dataloader)

    _predict_x = [batch_pre[0] for batch_pre in predict]
    _predict_score = [batch_pre[1] for batch_pre in predict]
    predict_x = torch.concat(_predict_x, dim=0)
    predict_score = torch.concat(_predict_score, dim=0)

    uv = UnivariateTimeSeriesView()
    uv.plot_kpi_with_anomaly_score_row2(test_x[:, -1], test_y, predict_score)
