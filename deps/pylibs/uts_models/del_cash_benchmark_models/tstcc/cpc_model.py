import numpy as np
import torch
from pylibs.utils.util_pytorch import convert_to_dl
from pylibs.utils.util_system import UtilSys
from torch import nn

from pylibs.utils.util_log import get_logger

log = get_logger()


class TSTCCConf(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32

        self.num_classes = 2
        self.dropout = 0.35
        # 与window有关，window=20 取5 window=18取4
        self.features_len = 4
        self.window_size = 18
        self.time_step = 18

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        self.weight_decay = 5e-4
        # data parameters
        # True, False. training maybe report errors
        self.drop_last = True
        self.batch_size = 64

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        self.center_eps = 0.0005


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 2


class _TSTCC(torch.nn.Module):
    def __init__(self, conf: TSTCCConf, device):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(conf.input_channels, 32, kernel_size=conf.kernel_size,
                      stride=conf.stride, bias=False, padding=(conf.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(conf.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, conf.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(conf.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = conf.features_len
        self.logits = nn.Linear(model_output_dim * conf.final_out_channels, conf.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x


class TSTCCModel:
    """A modified version of the COCA: https://github.com/ruiking04/COCA"""

    def __init__(self, config: TSTCCConf, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        assert self.device is not None

        self.config = config
        self.num_epochs = config.num_epoch
        self.batch_size = config.batch_size

        self.model = _TSTCC(self.config, device=self.device).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay)

    def score_coca(self, data):
        return self._model_evaluate(data)

    def score(self, test_x):
        data = convert_to_dl(test_x, self.config.batch_size)
        return self.score_coca(data)

    def predict(self, test_dl):
        val_target, val_score_origin, val_loss, all_projection = self._model_evaluate(test_dl)
        return val_score_origin

    def _model_evaluate(self, data_loader):

        self.model.eval()
        # total_loss, total_f1, total_precision, total_recall = [], [], [], []
        # all_target, all_score = [], []
        # all_projection = []
        with torch.no_grad():
            scores = []
            size = 0
            for data, target in data_loader:
                data, target = data.float().to(self.device), target.long().to(self.device)
                # torch.Size([256, 1, 64])
                inputs = data.float().to(self.device)
                hidden = self.model.init_hidden(len(inputs))
                _, _, _, nce = self.model(inputs, hidden, return_nce=True)
                scores.append(nce)
                size += data.shape[0]
                # scores.append(nce)
            scores = np.concatenate(scores)
            lattice = np.full((self.sequence_length, size), np.nan)
            for i, score in enumerate(scores):
                lattice[i % self.sequence_length, i: i + self.sequence_length] = score
            scores = np.nanmean(lattice, axis=0)

        return scores

        # all_target: 数据标签 label
        # all_score: score, a float value = dist(project of z, center)+ dist(project of output, center)
        # total_loss: a sum of a set of loss in each batch
        # all_projection: project of z

    def fit(self, train_x, valid_dl=None, label=None):
        UtilSys.is_debug_mode() and log.info("Training started ....")

        self.model.train()
        train_dl = convert_to_dl(train_x, self.config.batch_size)

        center = self.center_c(train_dl)
        length = torch.tensor(0, device=device)  # radius R initialized with 0 by default.
        for epoch in range(1, self.config.num_epoch + 1):
            # train_loader, center, length, epoch
            train_loss = self._model_train(train_dl)
            # log.info(f"Length:{length}")
            # scheduler.step(train_loss)
            UtilSys.is_debug_mode() and log.info(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}')

            # Update the loss

    def _model_train(self, train_loader):
        self.model.train()
        total_loss = []
        # torch.autograd.set_detect_anomaly(True)
        for batch_idx, (data, target) in enumerate(train_loader):
            # send to device
            data, target = data.float().to(self.device), target.long().to(self.device)
            # optimizer
            self.optimizer.zero_grad()
            _, features = model(data)
            features = features.permute(0, 2, 1)
            features = features.reshape(-1, config.final_out_channels)

            loss, score = train(features, center, length, epoch, config, device)
            total_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()

            target = target.reshape(-1)
            predict = score.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            all_target.extend(target)
            all_predict.extend(predict)

        total_loss = torch.tensor(total_loss).mean()

        return all_target, all_predict, total_loss, length

    def center_c(self, train_loader):
        eps = self.config.center_eps
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(self.config.window_size, device=self.device)
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                data, _ = data
                data = data.float().to(self.device)
                _, features = self.model(data)
                features = features.permute(0, 2, 1)
                features = features.reshape(-1, self.config.final_out_channels)
                n_samples += features.shape[0]
                c += torch.sum(features, dim=0)

        c /= (n_samples)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
