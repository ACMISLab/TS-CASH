import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVR
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import os
from joblib import dump


class Encoder(nn.Module):
    def __init__(self, window_size, lstm_layers, rnn_hidden_size, latent_size, input_size):
        super(Encoder, self).__init__()

        self.window_size = window_size
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=rnn_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.hidden_to_mu = nn.Linear(rnn_hidden_size, latent_size)
        self.hidden_to_logvar = nn.Linear(rnn_hidden_size, latent_size)

    def forward(self, x):
        batch_size = x.shape[0]

        output, _ = self.lstm(x)
        hidden_state = output.reshape(batch_size * self.window_size, -1)

        mu = self.hidden_to_mu(hidden_state)
        logvar = self.hidden_to_logvar(hidden_state)
        z = self.z_sample(mu, logvar)

        return z.view(batch_size, self.window_size, -1)

    def z_sample(self, mu, logvar):
        epsilon = torch.randn(mu.shape)
        return mu + torch.exp(logvar * 0.5) * epsilon


class Decoder(nn.Module):
    def __init__(self, window_size, lstm_layers, rnn_hidden_size, latent_size):
        super(Decoder, self).__init__()

        self.window_size = window_size

        self.lstm = nn.LSTM(input_size=latent_size,
                            hidden_size=rnn_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.hidden_to_mu = nn.Linear(in_features=rnn_hidden_size, out_features=1)
        self.hidden_to_logvar = nn.Linear(in_features=rnn_hidden_size, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]

        output, _ = self.lstm(x)
        hidden_state = output.reshape(batch_size * self.window_size, -1)

        mu = self.hidden_to_mu(hidden_state)
        logvar = self.hidden_to_logvar(hidden_state)
        F = nn.Tanh()
        logvar = F(logvar)

        return mu.view(batch_size, self.window_size, -1), logvar.view(batch_size, self.window_size, -1)


class LSTM_VAE(nn.Module):
    def __init__(self, window_size, lstm_layers, rnn_hidden_size, latent_size
                 , input_size):
        super(LSTM_VAE, self).__init__()

        self.svr = SVR(max_iter=30000)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.latent_size = latent_size
        self.window_size = window_size

        self.encoder = Encoder(window_size, lstm_layers, rnn_hidden_size, latent_size, input_size)
        self.decoder = Decoder(window_size, lstm_layers, rnn_hidden_size, latent_size)

    def forward(self, x):
        z = self.encoder(x)
        mu, logvar = self.decoder(z)
        return z, mu, logvar

    def fit_svr(self, loader):
        anomaly_scorer = []
        Z = []
        self.eval()
        for idx, x in enumerate(loader):
            x = self.convert_x(x)
            z, mu, logvar = self(x)
            anomaly_score = self.loss_function(x, logvar, mu, logvar, 'none')
            Z = np.append(Z, z.detach().numpy())
            anomaly_scorer = np.append(anomaly_scorer, anomaly_score.detach().numpy())

        Z = Z.reshape((-1, self.latent_size))
        print("SVR Fitting...")
        self.svr.fit(Z, anomaly_scorer)
        print("SVR completed")

    def fit(self, opt, num_epochs, train_loader):

        def early_stopping_callback(i, _l, _e):
            if i:
                self.fit_svr(train_loader)
                # self.save(model_path)

        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for idx, x in enumerate(train_loader):
                opt.zero_grad()
                x = self.convert_x(x)
                _, mu, logvar = self(x)

                loss = self.loss_function(x, logvar, mu, logvar, 'mean')
                total_loss += loss.item()
                loss.backward()
                opt.step()

            print("Epoch No: {}, Average Loss: {}".format(epoch + 1, total_loss / idx))

        self.fit_svr(train_loader)

    def convert_x(self, x):
        _train_x, train_y = x
        _train_x = _train_x[..., None]
        return _train_x

    def score(self, exec_loader):
        Z = []
        for X in exec_loader:
            X = self.convert_x(X)
            z, _, _ = self(X)
            Z = np.append(Z, z.detach().numpy())
        Z = Z.reshape(-1, self.latent_size)
        scores = self.svr.predict(Z)
        scores = scores.reshape(-1, self.window_size)

        agg_scores = []
        len_ts = scores.shape[0] + (self.window_size - 1)

        for x in range(len_ts):
            window_idx, relative_idx = self.window_indices_for(x)
            n_windows = min(scores.shape[0] - window_idx, relative_idx + 1)
            s = np.array([scores[window_idx + i, relative_idx - i] for i in range(n_windows)]).mean()
            agg_scores.append(s)

        return np.array(agg_scores)  # (2893,)

    def loss_function(self, x, x_hat, mean, log_var, reduction_type):
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction=reduction_type)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return KLD + reproduction_loss

    def window_indices_for(self, point_idx: int) -> Tuple[int, int]:
        window_idx = point_idx - (self.window_size - 1)
        relative_idx = self.window_size - 1
        if window_idx < 0:
            relative_idx += window_idx
            window_idx = 0

        return window_idx, relative_idx

    def save(self, path: os.PathLike):
        dump(self, path)
