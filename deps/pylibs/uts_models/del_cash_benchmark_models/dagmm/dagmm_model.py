#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Deep autoencoding Gaussian mixture model for anomaly detection (DAGMM)
"""
from pylibs.utils.util_class import initializer
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_pytorch import convert_to_dl
from pylibs.utils.util_system import UtilSys

log = get_logger()
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

import numpy as np


class DAGMMConfig:
    """
    Configuration class for DAGMM. The normalization is inherited from `NormalizingConfig`.
    The input data will be standardized automatically.
    """

    @initializer
    def __init__(
            self,
            gmm_k: int = 3,
            hidden_size: int = 5,
            sequence_len: int = 1,
            lambda_energy: float = 0.1,
            lambda_cov_diag: float = 0.005,
            lr: float = 1e-3,
            batch_size: int = 256,
            num_epochs: int = 10,
            **kwargs
    ):
        """
        :param gmm_k: The number of Gaussian distributions
        :param hidden_size: The hidden size of the autoencoder module in DAGMM
        :param sequence_len: The input series length, e.g., input = [x(t-sequence_len+1)...,x(t-1),x(t)]
        :param lambda_energy: The regularization weight for the energy term
        :param lambda_cov_diag: The regularization weight for the covariance diagonal entries
        :param lr: The learning rate during training
        :param batch_size: The batch size during training
        :param num_epochs: The number of training epochs
        """
        super().__init__(**kwargs)


class DAGMM:
    """
    Deep autoencoding Gaussian mixture model for anomaly detection (DAGMM).
    DAGMM combines an autoencoder with a Gaussian mixture model to model the distribution
    of the reconstruction errors. DAGMM jointly optimizes the parameters of the deep autoencoder
    and the mixture model simultaneously in an end-to-end fashion.

    - paper: `Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho and Haifeng Chen.
      Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection. 2018.
      <https://openreview.net/forum?id=BJJLHbb0->`_.
    """

    def __init__(self, config: DAGMMConfig):
        self.device = config.device
        self.gmm_k = config.gmm_k
        self.hidden_size = config.hidden_size
        self.sequence_length = config.sequence_len
        self.lambda_energy = config.lambda_energy
        self.lambda_cov_diag = config.lambda_cov_diag

        self.lr = config.lr
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs

        self.data_dim = -1
        self.dagmm, self.optimizer = None, None
        self.train_energy, self._threshold = None, None

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    @property
    def _default_train_config(self):
        return dict()

    def _build_model(self, dim):
        hidden_size = self.hidden_size + int(dim / 20)
        dagmm = DAGMMModule(
            autoencoder=AEModule(n_features=dim, sequence_length=self.sequence_length, hidden_size=hidden_size),
            n_gmm=self.gmm_k,
            latent_dim=hidden_size + 2,
            device=self.device,
        )
        return dagmm

    def _step(self, input_data, max_grad_norm=5):
        enc, dec, z, gamma = self.dagmm(input_data)
        total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_func(
            x=input_data,
            recon_x=dec,
            z=z,
            gamma=gamma,
            lambda_energy=self.lambda_energy,
            lambda_cov_diag=self.lambda_cov_diag,
        )
        self.optimizer.zero_grad()
        total_loss = torch.clamp(total_loss, max=1e7)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), max_grad_norm)
        self.optimizer.step()
        return total_loss, sample_energy, recon_error, cov_diag

    def fit(self, X, y=None):
        # data_loader = RollingWindowDataset(
        #     train_data,
        #     target_seq_index=None,
        #     shuffle=True,
        #     flatten=False,
        #     n_past=self.sequence_length,
        #     n_future=0,
        #     batch_size=self.batch_size,
        # )
        train_loader = convert_to_dl(X, self.batch_size)
        if self.dagmm is None and self.optimizer is None:
            self.dagmm = self._build_model(1).to(self.device)
            self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)
            self.dagmm.train()
        # self.data_dim = train_data.shape[1]
        self.data_dim = 1

        for epoch in range(self.num_epochs):
            total_loss, recon_error = 0, 0
            for input_data, _, in train_loader:
                # input data.shape=(37, 64, 1)
                # input_data = torch.tensor(input_data, dtype=torch.float, device=self.device)
                input_data = torch.unsqueeze(torch.squeeze(input_data, dim=1).float().to(self.device), 2)

                # input_data=input_data..float(). to(self.device)
                loss, _, error, _ = self._step(input_data)
                total_loss += loss
                recon_error += error
            UtilSys.is_debug_mode() and log.info("epoch %s, loss %s" % (epoch, float(total_loss)))

    def get_loss(self, X):
        out_loss = 0
        train_loader = convert_to_dl(X, self.batch_size)
        for input_data, _, in train_loader:
            # input data.shape=(37, 64, 1)
            # input_data = torch.tensor(input_data, dtype=torch.float, device=self.device)
            input_data = torch.unsqueeze(torch.squeeze(input_data, dim=1).float().to(self.device), 2)

            # input_data=input_data..float(). to(self.device)
            loss, _, error, _ = self._step(input_data)
            out_loss += loss
        return out_loss

    def score(self, X):
        self.dagmm.eval()
        # ts = pd.concat((time_series_prev, time_series)) if time_series_prev is None else time_series
        # data_loader = RollingWindowDataset(
        #     ts,
        #     target_seq_index=None,
        #     shuffle=False,
        #     flatten=False,
        #     n_past=self.sequence_length,
        #     n_future=0,
        #     batch_size=1,
        # )

        data_loader = convert_to_dl(X, self.batch_size)
        # test_energy = np.full((self.sequence_length, X.shape[0]), np.nan)
        sample_energys = []
        for i, (sequence, _) in enumerate(data_loader):
            sequence = torch.unsqueeze(torch.squeeze(sequence, dim=1).float().to(self.device), 2)
            # sequence = torch.tensor(sequence, dtype=torch.float, device=self.device)
            enc, dec, z, gamma = self.dagmm(sequence.float())
            sample_energy, _ = self.dagmm.compute_energy(z, size_average=False)
            idx = (i % self.sequence_length, np.arange(i, i + self.sequence_length))
            sample_energys.append(sample_energy.cpu().data.numpy())
        test_energy = np.concatenate(sample_energys, axis=0)
        return test_energy


class AEModule(nn.Module):
    """
    The autoencoder module used in DAGMM.

    :meta private:
    """

    def __init__(self, n_features, sequence_length, hidden_size, activation=nn.Tanh):
        """
        :param n_features: The number of the input features (number of variables)
        :param sequence_length: The length of the input sequence
        :param hidden_size: The latent size
        :param activation: The activation function for the hidden layers
        """
        super().__init__()
        input_length = n_features * sequence_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[nn.Linear(int(a), int(b)), activation()] for a, b in enc_setup.reshape(-1, 2)])
        self.encoder = nn.Sequential(*layers.flatten()[:-1])
        layers = np.array([[nn.Linear(int(a), int(b)), activation()] for a, b in dec_setup.reshape(-1, 2)])
        self.decoder = nn.Sequential(*layers.flatten()[:-1])

    def forward(self, x, return_latent=False):
        enc = self.encoder(x.view(x.shape[0], -1).float())
        dec = self.decoder(enc)
        recon_x = dec.view(x.shape)
        return (recon_x, enc) if return_latent else recon_x


class DAGMMModule(nn.Module):
    """
    The DAGMM module used in the DAGMM detector.

    :meta private:
    """

    def __init__(self, autoencoder, n_gmm, latent_dim, device):
        """
        :param autoencoder: The autoencoder model
        :param n_gmm: The number of Gaussian mixtures
        :param latent_dim: The latent dimension
        :param device: CUDA or CPU
        """
        super(DAGMMModule, self).__init__()
        self.add_module("autoencoder", autoencoder)
        self.device = device

        self.estimation = nn.Sequential(
            *[nn.Linear(latent_dim, 10), nn.Tanh(), nn.Linear(10, n_gmm), nn.Softmax(dim=1)]
        )
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))

    @staticmethod
    def relative_euclidean_distance(a, b, dim=1):
        return (a - b).norm(2, dim=dim) / torch.clamp(a.norm(2, dim=dim), min=1e-10)

    def forward(self, x):
        dec, enc = self.autoencoder(x, return_latent=True)
        a, b = x.view(x.shape[0], -1), dec.view(dec.shape[0], -1)
        cos_distance = F.cosine_similarity(a, b, dim=1).unsqueeze(-1)
        euc_distance = DAGMMModule.relative_euclidean_distance(a, b, dim=1).unsqueeze(-1)
        z = torch.cat([enc, euc_distance, cos_distance], dim=1)
        return enc, dec, z, self.estimation(z)

    def compute_gmms(self, z, gamma):
        # weights
        sum_gamma = torch.sum(gamma, dim=0)
        phi = sum_gamma / gamma.shape[0]
        # means and covariances
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        # store these values for prediction
        self.phi, self.mu, self.cov = phi.data, mu.data, cov.data
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True, eps=1e-6):
        phi = self.phi if phi is None else phi
        mu = self.mu if mu is None else mu
        cov = self.cov if cov is None else cov

        cov_inv, cov_det, cov_diag = [], [], 0
        for i in range(cov.shape[0]):
            cov_k = cov[i] + torch.eye(cov.shape[1], device=self.device) * eps
            inv_k = torch.tensor(np.linalg.pinv(cov_k.cpu().data.numpy()), dtype=torch.float, device=self.device)
            cov_inv.append(inv_k.unsqueeze(0))
            eigenvalues = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            determinant = np.prod(np.clip(eigenvalues, a_min=eps, a_max=None))
            cov_det.append(determinant)
            cov_diag += torch.sum(1.0 / cov_k.diag())

        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_inv = torch.cat(cov_inv, dim=0)
        cov_det = torch.tensor(cov_det, dtype=torch.float, device=self.device)
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inv.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max(exp_term_tmp.clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(cov_det) + eps).unsqueeze(0), dim=1) + eps
        )
        if size_average:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag

    def loss_func(self, x, recon_x, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x.view(*recon_x.shape) - recon_x) ** 2)
        phi, mu, cov = self.compute_gmms(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag
