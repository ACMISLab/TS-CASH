import numpy as np
import torch
from pylibs.utils.util_pytorch import convert_to_dl
from pylibs.utils.util_system import UtilSys
from torch import nn

from pylibs.uts_models.benchmark_models.cpc.CPC import CDCK2, CPC
from pylibs.uts_models.benchmark_models.cpc.cpc_conf import CPCConf
from pylibs.utils.util_log import get_logger

log = get_logger()


class _CPC(torch.nn.Module):
    def __init__(self, conf: CPCConf, device):

        super(_CPC, self).__init__()
        self.device = device
        self.conf = conf
        self.batch_size = self.conf.batch_size
        self.seq_len = self.conf.sequence_length
        self.timestep = self.conf.timestep
        self.hidden_size = self.conf.hidden_size
        self.input_channel = self.conf.input_channel
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_channel, self.hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(self.hidden_size, self.hidden_size // 2, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(self.hidden_size // 2, self.hidden_size) for _ in range(self.conf.timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size // 2).to(self.device)

    def forward(self, x, hidden, return_nce=False):
        batch = x.size()[0]
        # randomly pick time stamps
        t_samples = torch.randint(self.seq_len // 2 - self.timestep, size=(1,)).long().item()
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.hidden_size)).float()  # e.g. size 12*8*512
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, self.hidden_size)  # z_tk e.g. size 8*512
        forward_seq = z[:, :t_samples + 1, :]  # e.g. size 8*100*512
        output, hidden = self.gru(forward_seq, hidden)  # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, self.hidden_size // 2)  # c_t e.g. size 8*256
        pred = torch.empty((self.timestep, batch, self.hidden_size)).float()  # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512
        nce_list = []
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce_list.append(torch.sum(torch.diag(self.lsoftmax(total))).item())
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch * self.timestep
        accuracy = 1. * correct.item() / batch
        if return_nce:
            return accuracy, nce, hidden, nce_list
        else:
            return accuracy, nce, hidden

    def predict(self, x, hidden):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        output, hidden = self.gru(z, hidden)  # output size e.g. 8*128*256

        return output, hidden  # return every frame
        # return output[:,-1,:], hidden # only return the last frame per utt


class _CPC_BAK(torch.nn.Module):
    """
    The CPC-based multivariate time series anomaly detector.
    """

    def __init__(self, config: CPCConf, device):
        super(_CPC, self).__init__()
        self.config = config
        self.num_epochs = config.epochs
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.device = device
        self.model = CDCK2(self.config.timestep, self.config.batch_size, self.config.sequence_length,
                           config.window_size)

    def init_hidden(self, batch_size):
        return torch.zeros(2 * 1, batch_size, 40)


class LSTMEDModule(torch.nn.Module):
    """
    The LSTM-encoder-decoder module. Both the encoder and decoder are LSTMs.

    :meta private:
    """

    def __init__(self, n_features, hidden_size, n_layers, dropout, device):
        """
        :param n_features: The input feature dimension
        :param hidden_size: The LSTM hidden size
        :param n_layers: The number of LSTM layers
        :param dropout: The dropout rate
        :param device: CUDA or CPU
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.encoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[0],
            bias=True,
            dropout=self.dropout[0],
        )
        self.decoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[1],
            bias=True,
            dropout=self.dropout[1],
        )
        self.output_layer = nn.Linear(self.hidden_size, self.n_features)

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.n_layers[0], batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.n_layers[0], batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x, return_latent=False):
        # Encoder
        enc_hidden = self.init_hidden_state(x.shape[0])
        _, enc_hidden = self.encoder(x.float(), enc_hidden)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(x.shape).to(self.device)
        for i in reversed(range(x.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(x[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        return (output, enc_hidden[1][-1]) if return_latent else output


class CPCModel:
    """A modified version of the COCA: https://github.com/ruiking04/COCA"""

    def __init__(self, config, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        assert self.device is not None

        self.config = config
        self.num_epochs = config.epochs
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length

        self.model = _CPC(self.config, device=self.device).to(self.device)
        log.info("CPC is on device: [{}]".format(self.device))

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
            size=0
            for data, target in data_loader:
                data, target = data.float().to(self.device), target.long().to(self.device)
                # torch.Size([256, 1, 64])
                inputs = data.float().to(self.device)
                hidden = self.model.init_hidden(len(inputs))
                _, _, _, nce = self.model(inputs, hidden, return_nce=True)
                scores.append(nce)
                size+=data.shape[0]
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
        train_dl = convert_to_dl(train_x, self.config.batch_size)

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

            hidden = self.model.init_hidden(len(data))
            acc, loss, hidden = self.model(data, hidden)

            # Update hypersphere radius R on mini-batch distances

            total_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()

        total_loss = torch.tensor(total_loss).mean()
        return total_loss
