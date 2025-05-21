#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The LSTM-encoder-decoder-based anomaly detector for multivariate time series
"""
from typing import Sequence

import numpy as np
import pandas as pd

from pylibs.utils.util_pytorch import convert_to_dl
from pylibs.utils.util_system import UtilSys

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

from merlion.models.base import NormalizingConfig
from merlion.models.anomaly.base import DetectorConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.misc import initializer


class LSTMEDConfig(DetectorConfig, NormalizingConfig):
    """
    Configuration class for LSTM-encoder-decoder. The normalization is inherited from `NormalizingConfig`.
    The input data will be standardized automatically.
    """

    _default_threshold = AggregateAlarms(alm_threshold=2.5, abs_score=True)

    @initializer
    def __init__(
            self,
            hidden_size: int = 5,
            sequence_len: int = 20,
            n_layers: Sequence[int] = (1, 1),
            dropout: Sequence[int] = (0, 0),
            lr: float = 1e-3,
            batch_size: int = 256,
            num_epochs: int = 10,
            **kwargs
    ):
        """
        :param hidden_size: The hidden state size of the LSTM modules
        :param sequence_len: The input series length, e.g., input = [x(t-sequence_len+1)...,x(t-1),x(t)]
        :param n_layers: The number of layers for the LSTM encoder and decoder. ``n_layer`` has two values, i.e.,
            ``n_layer[0]`` is the number of encoder layers and ``n_layer[1]`` is the number of decoder layers.
        :param dropout: The dropout rate for the LSTM encoder and decoder. ``dropout`` has two values, i.e.,
            ``dropout[0]`` is the dropout rate for the encoder and ``dropout[1]`` is the dropout rate for the decoder.
        :param lr: The learning rate during training
        :param batch_size: The batch size during training
        :param num_epochs: The number of training epochs
        """
        super().__init__(**kwargs)


class LSTMEDConfig(DetectorConfig, NormalizingConfig):
    """
    Configuration class for LSTM-encoder-decoder. The normalization is inherited from `NormalizingConfig`.
    The input data will be standardized automatically.
    """

    _default_threshold = AggregateAlarms(alm_threshold=2.5, abs_score=True)

    @initializer
    def __init__(
            self,
            hidden_size: int = 5,
            sequence_len: int = 20,
            n_layers: Sequence[int] = (1, 1),
            dropout: Sequence[int] = (0, 0),
            lr: float = 1e-3,
            batch_size: int = 256,
            num_epochs: int = 10,
            **kwargs
    ):
        """
        :param hidden_size: The hidden state size of the LSTM modules
        :param sequence_len: The input series length, e.g., input = [x(t-sequence_len+1)...,x(t-1),x(t)]
        :param n_layers: The number of layers for the LSTM encoder and decoder. ``n_layer`` has two values, i.e.,
            ``n_layer[0]`` is the number of encoder layers and ``n_layer[1]`` is the number of decoder layers.
        :param dropout: The dropout rate for the LSTM encoder and decoder. ``dropout`` has two values, i.e.,
            ``dropout[0]`` is the dropout rate for the encoder and ``dropout[1]`` is the dropout rate for the decoder.
        :param lr: The learning rate during training
        :param batch_size: The batch size during training
        :param num_epochs: The number of training epochs
        """
        super().__init__(**kwargs)


class LSTMED:
    """
    The LSTM-encoder-decoder-based multivariate time series anomaly detector.
    The time series representation is modeled by an encoder-decoder network where
    both encoder and decoder are LSTMs. The distribution of the reconstruction error
    is estimated for anomaly detection.
    """

    def __init__(self, config: LSTMEDConfig):
        self.device = config.device
        self.conf = config
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.hidden_size = config.hidden_size
        self.sequence_length = config.sequence_len
        self.n_layers = config.n_layers
        self.dropout = config.dropout

        assert (
                len(self.n_layers) == 2
        ), "Param n_layers should contain two values: (num_layers for LSTM encoder, num_layers for LSTM decoder)"
        assert len(self.n_layers) == len(self.dropout), "Param dropout should contain two values"

        self.lstmed = None
        self.data_dim = None

    def _build_model(self, dim):
        return LSTMEDModule(dim, self.hidden_size, self.n_layers, self.dropout, self.device)

    def fit(self, X, y=None):
        train_loader = convert_to_dl(X, self.batch_size)
        self.lstmed = self._build_model(1).to(self.device)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss(reduction="sum")

        self.lstmed.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch, _ in train_loader:
                batch = batch.float().to(self.device)
                batch = torch.unsqueeze(torch.squeeze(batch), 2)
                output = self.lstmed(batch)
                loss = loss_func(output, batch)
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
            UtilSys.is_debug_mode() and print(f"epoch: {epoch}, loss: {loss.item()}")

    def score(self, X) -> pd.DataFrame:
        data_loader = convert_to_dl(X, self.batch_size)
        self.lstmed.eval()
        ts = X
        scores, outputs = [], []
        for batch, _ in data_loader:
            # batch.shape=torch.Size([256, 64, 1])
            batch = batch.float().to(self.device)
            batch = torch.unsqueeze(torch.squeeze(batch), 2)

            output = self.lstmed(batch)
            # error.shape=torch.Size([256, 64, 1])
            error = nn.L1Loss(reduction="none")(output, batch)
            score = np.mean(error.view(-1, 1).data.cpu().numpy(), axis=1)
            scores.append(score.reshape(batch.shape[0], self.sequence_length))

        scores = np.concatenate(scores)
        # lattice = np.full((self.sequence_length, X.shape[0]), np.nan)
        # for i, score in enumerate(scores):
        #     lattice[i % self.sequence_length, i: i + self.sequence_length] = score
        # scores = np.nanmean(lattice, axis=0)
        # return scores[-len(X):]
        return scores.mean(axis=1)


class LSTMEDModule(nn.Module):
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
