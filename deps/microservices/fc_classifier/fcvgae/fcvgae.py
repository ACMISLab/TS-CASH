import numpy as np
import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, latent_dim, dropout, num_layers, norm):

        super(GCNEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_feats = hidden_feats
        self.latent_dim = latent_dim
        self.logvar_activation = nn.Tanh()
        self.input_conv = GraphConv(in_feats, hidden_feats, norm=norm)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_feats, hidden_feats, norm=norm))
        if num_layers > 1:
            self.convs.append(GraphConv(hidden_feats, hidden_feats, norm=norm))
        self.mu_layer = nn.Linear(hidden_feats, latent_dim)
        self.logvar_layer = nn.Linear(hidden_feats, latent_dim)
 
    def forward(self, g, features):
        # print(g.device)
        h = F.leaky_relu(self.input_conv(g, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
            h = self.dropout(h)
        mu = self.mu_layer(h)
        logvar = self.logvar_activation(self.logvar_layer(h))
        return mu, logvar

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # print(f"sample\nstd: [{std.min().item(),  std.max().item()}]")
        # print(f"eps: [{eps.min().item(),  eps.max().item()}]")
        if torch.isinf(std.max()):
            print("find inf")
            raise RuntimeError("find inf when sampling z")
        return mu + eps * std

    def transform(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class GCNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_feats, out_feats, dropout, num_layers, norm):
        super(GCNDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_feats = hidden_feats
        self.out_feats=out_feats

        # Shared GraphSAGE layers
        self.input_conv = GraphConv(latent_dim, hidden_feats, norm=norm)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_feats, hidden_feats, norm=norm))
        if num_layers > 1:
            self.convs.append(GraphConv(hidden_feats, out_feats, norm=norm))

    def forward(self, g, z):
        h = F.leaky_relu(self.input_conv(g, z))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
            h = self.dropout(h)
        return h

    def transform(self, g, z):
        h = F.leaky_relu(self.input_conv(g, z))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        return h


class FCVGAE(pytorch_lightning.LightningModule):
    def __init__(self, in_feats, hidden_feats, latent_dim, out_feats, dropout=0.0, mask_rate=0.3, num_layers=2,
                 norm='none', aug_multiple=3):
        super(FCVGAE, self).__init__()
        # n_feats, hidden_feats, latent_dim, dropout, num_layers, norm
        self.mask_rate = mask_rate
        self.aug_multiple = aug_multiple
        self.encoder = GCNEncoder(in_feats, hidden_feats, latent_dim, dropout, num_layers, norm)

        self.decoder = GCNDecoder(latent_dim, hidden_feats, in_feats, dropout, num_layers, norm)

    def forward(self, g, features):
        mu, logvar = self.encoder(g, features)
        z = self.encoder.sample_z(mu, logvar)
        # print(f"z max: {z.max().item()},z min: {z.min().item()}")
        x_hat = self.decoder(g, z)
        # return x_hat, mu, logvar
        return x_hat, mu, logvar

    def transform(self, g, features):
        mu, logvar = self.encoder.transform(g, features)
        z = self.encoder.sample_z(mu, logvar)
        x_hat = self.decoder.transform(g, z)
        return x_hat

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (e.g., MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def data_aug(self, g, features, mask_rate):
        # Randomly mask a certain number of vertices.
        device = features.device  # 获取 features 所在的设备
        mask = torch.rand(g.number_of_nodes()) < mask_rate
        mask = mask.to(device)
        # Randomly mask a certain number of feature dimensions.
        z = features * torch.from_numpy(
            np.random.choice([0, 1], size=features.shape[-1], p=[mask_rate, 1 - mask_rate])).float().to(device)
        zi = torch.zeros_like(z).to(device)
        z = torch.where(mask.unsqueeze(-1), zi, z)
        return g, z

    def training_step(self, batch, batch_index):
        total_loss = 0
        for _ in range(self.aug_multiple):
            _, graphs, features = batch
            aug_gs, aug_inputs = self.data_aug(graphs, features, self.mask_rate)
            # save_pkl([graphs, features, self.mask_rate], "debug_data_auto_torch.pkl")
            # aug_gs_ts, aug_inputs_ts = data_aug_torch(graphs, features, self.mask_rate)
            # aug_gs.to(loader.device())
            # aug_inputs.to(loader.device())
            # aug_gs: no aug data
            x_hat, mu, logvar = self(aug_gs, aug_inputs)
            #
            # Ensure x_hat and aug_inputs have matching feature dimensions
            if x_hat.shape[1] != aug_inputs.shape[1]:
                min_dim = min(x_hat.shape[1], aug_inputs.shape[1])
                x_hat = x_hat[:, :min_dim]
                aug_inputs = aug_inputs[:, :min_dim]
            # Reconstruction loss (e.g., MSE)
            recon_loss = F.mse_loss(x_hat, aug_inputs, reduction='mean')  # 7183839.0
            # KL divergence loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            total_loss += loss
        self.log("train_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

