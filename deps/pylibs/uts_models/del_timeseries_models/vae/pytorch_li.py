import lightning.pytorch as pl
import torch
from lightning import Trainer
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
from pylibs.dataset.AIOpsDataset import AIOpsDataset
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018

WINDOW_SIZE = 20
BATCH_SIZE = 32
da = DatasetAIOps2018(kpi_id=DatasetAIOps2018.KPI_IDS.D1,
                      windows_size=WINDOW_SIZE,
                      is_include_anomaly_windows=True,
                      valid_rate=0.2)
train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()

train_dataset = AIOpsDataset(train_x, train_y)
valid_dataset = AIOpsDataset(valid_x, valid_y)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)


# define any number of nn.Modules (or use your current ones)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(WINDOW_SIZE, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, WINDOW_SIZE))

    def forward(self, x):
        return self.l1(x)


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    #
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# setup data

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
# trainer = pl.Trainer(limit_train_batches=100, accelerator="gpu", devices=[0], max_epochs=1)
trainer = pl.Trainer(limit_train_batches=100, accelerator="gpu", devices=[0], max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_dataloader)

tester = Trainer()
tester.test(autoencoder, dataloaders=valid_dataloader)
#
# checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
# autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)
#
# # choose your trained nn.Module
# encoder = autoencoder.encoder
# encoder.eval()
#
# # embed 4 fake images!
# fake_image_batch = Tensor(4, 28 * 28)
# embeddings = encoder(fake_image_batch)
# print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
