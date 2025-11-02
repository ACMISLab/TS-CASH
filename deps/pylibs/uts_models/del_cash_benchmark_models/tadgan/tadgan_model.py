import traceback

# import numpy as np
import torch
import torch.optim as optim
from pylibs.utils.util_pytorch import convert_to_dl
from pylibs.utils.util_system import UtilSys
from torch.autograd import Variable
from pylibs.uts_models.benchmark_models.tadgan.TadGAN import model
from pylibs.utils.util_log import get_logger
from scipy import stats

log = get_logger()


class TadGanEdConf:
    def __init__(self):
        self.device = "cpu"
        self.batch_size = 128
        self.lr = 1e-6
        self.signal_shape = 100
        self.latent_space_dim = 20
        self.num_epochs = 50


class TadGanEd:
    def __init__(self, conf: TadGanEdConf):
        self.device = conf.device
        self.conf = conf

        lr = self.conf.lr
        signal_shape = self.conf.signal_shape
        latent_space_dim = self.conf.latent_space_dim
        # lr = 1e-6
        # signal_shape = 100
        # latent_space_dim = 20
        encoder_path = 'models/encoder.pt'
        decoder_path = 'models/decoder.pt'
        critic_x_path = 'models/critic_x.pt'
        critic_z_path = 'models/critic_z.pt'

        self.encoder = model.Encoder(encoder_path, self.conf.signal_shape, device=self.device)
        self.decoder = model.Decoder(decoder_path, self.conf.signal_shape, device=self.device)
        self.critic_x = model.CriticX(critic_x_path, self.conf.signal_shape, device=self.device)
        self.critic_z = model.CriticZ(critic_z_path, device=self.device)

        self.mse_loss = torch.nn.MSELoss()

        self.optim_enc = optim.Adam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optim_dec = optim.Adam(self.decoder.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optim_cx = optim.Adam(self.critic_x.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optim_cz = optim.Adam(self.critic_z.parameters(), lr=lr, betas=(0.5, 0.999))

        # train(n_epochs=1)

        # anomaly_detection.test(test_loader, encoder, decoder, critic_x)

    def fit(self, X, y=None):

        data_loader = convert_to_dl(X, self.conf.batch_size)
        UtilSys.is_debug_mode() and log.info('Starting training')
        cx_epoch_loss = list()
        cz_epoch_loss = list()
        encoder_epoch_loss = list()
        decoder_epoch_loss = list()

        for epoch in range(self.conf.num_epochs):
            UtilSys.is_debug_mode() and log.info('Epoch {}'.format(epoch))
            n_critics = 5

            cx_nc_loss = list()
            cz_nc_loss = list()

            for i in range(n_critics):
                cx_loss = list()
                cz_loss = list()

                for batch, (sample, y) in enumerate(data_loader):
                    sample = sample.squeeze(dim=1).float().to(self.device)
                    loss = self.critic_x_iteration(sample)
                    cx_loss.append(loss)

                    loss = self.critic_z_iteration(sample)
                    cz_loss.append(loss)

                cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
                cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))

            UtilSys.is_debug_mode() and log.info('Critic training mark_as_finished in epoch {}'.format(epoch))
            encoder_loss = list()
            decoder_loss = list()

            for batch, (sample, _) in enumerate(data_loader):
                sample = sample.squeeze(dim=1).float().to(self.device)
                enc_loss = self.encoder_iteration(sample)
                dec_loss = self.decoder_iteration(sample)
                encoder_loss.append(enc_loss)
                decoder_loss.append(dec_loss)

            cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
            cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
            encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
            decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))
            UtilSys.is_debug_mode() and log.info('Encoder decoder training mark_as_finished in epoch {}'.format(epoch))
            UtilSys.is_debug_mode() and log.info(
                'critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(
                    cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

    def critic_x_iteration(self, sample):
        self.optim_cx.zero_grad()

        # x = sample.view(1, self.conf.batch_size, self.conf.signal_shape) #torch.Size([1, 128, 64])

        valid_x = self.critic_x(sample)
        valid_x = torch.squeeze(valid_x).float().to(self.device)
        critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).float().to(self.device) * valid_x).float().to(
            self.device)  # Wasserstein Loss

        # The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
        z = torch.empty(1, sample.shape[0], self.conf.latent_space_dim).uniform_(0, 1).float().to(self.device)
        x_ = self.decoder(z)
        fake_x = self.critic_x(x_)
        fake_x = torch.squeeze(fake_x)

        # print("fake_x.device", fake_x.device)
        critic_score_fake_x = torch.mean(torch.ones(fake_x.shape).float().to(self.device) * fake_x)  # Wasserstein Loss

        alpha = torch.rand(sample.shape).to(self.device)
        # alpha 122*64
        # sample torch.Size([122, 64])
        # alpha torch.Size([122, 64])
        # x_.shape torch.Size([1, 128, 64])
        ix = Variable(alpha * sample + (1 - alpha) * x_).to(self.device)  # Random Weighted Average
        # alpha.shape=torch.Size([128, 64])
        # x_.shape=torch.Size([1, 128, 64])
        # sample.shape=torch.Size([128, 64])
        ix.requires_grad_(True)
        v_ix = self.critic_x(ix)
        v_ix.mean().backward()
        gradients = ix.grad
        # Gradient Penalty Loss
        gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

        # Critic has to maximize Cx(Valid X) - Cx(Fake X).
        # Maximizing the above is same as minimizing the negative.
        wl = critic_score_fake_x - critic_score_valid_x
        loss = wl + gp_loss
        loss.backward()
        self.optim_cx.step()
        return loss

    def critic_z_iteration(self, sample):
        self.optim_cz.zero_grad()

        # x = sample.view(1, self.conf.batch_size, self.conf.signal_shape)
        # print("critic_z_iteration.device", sample.device)
        z = self.encoder(sample)
        valid_z = self.critic_z(z)
        valid_z = torch.squeeze(valid_z)
        critic_score_valid_z = torch.mean(torch.ones(valid_z.shape).to(self.device) * valid_z).to(self.device)

        z_ = torch.empty(1, sample.shape[0], self.conf.latent_space_dim).uniform_(0, 1).to(self.device)
        fake_z = self.critic_z(z_)
        fake_z = torch.squeeze(fake_z)
        critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).to(self.device) * fake_z)  # Wasserstein Loss

        wl = critic_score_fake_z - critic_score_valid_z

        alpha = torch.rand(z.shape).to(self.device)
        iz = Variable(alpha * z + (1 - alpha) * z_).to(self.device)  # Random Weighted Average
        iz.requires_grad_(True)
        v_iz = self.critic_z(iz)
        v_iz.mean().backward()
        gradients = iz.grad
        gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

        loss = wl + gp_loss
        loss.backward()
        self.optim_cz.step()

        return loss

    def encoder_iteration(self, sample):
        self.optim_enc.zero_grad()

        # x = sample.view(1, self.conf.batch_size, self.conf.signal_shape)
        x = sample
        valid_x = self.critic_x(sample)
        valid_x = torch.squeeze(valid_x)
        critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).to(self.device) * valid_x)  # Wasserstein Loss

        z = torch.empty(1, sample.shape[0], self.conf.latent_space_dim).uniform_(0, 1).to(self.device)
        x_ = self.decoder(z)
        fake_x = self.critic_x(x_)
        fake_x = torch.squeeze(fake_x)
        critic_score_fake_x = torch.mean(torch.ones(fake_x.shape).to(self.device) * fake_x)

        enc_z = self.encoder(x)
        gen_x = self.decoder(enc_z)

        mse = self.mse_loss(x.float(), gen_x.float())
        loss_enc = mse + critic_score_valid_x - critic_score_fake_x
        loss_enc.backward(retain_graph=True)
        self.optim_enc.step()

        return loss_enc

    def decoder_iteration(self, sample):
        self.optim_dec.zero_grad()

        # x = sample.view(1, self.conf.batch_size, self.conf.signal_shape)
        x = sample
        z = self.encoder(x)
        valid_z = self.critic_z(z)
        valid_z = torch.squeeze(valid_z)
        critic_score_valid_z = torch.mean(torch.ones(valid_z.shape).to(self.device) * valid_z)

        z_ = torch.empty(1, sample.shape[0], self.conf.latent_space_dim).uniform_(0, 1).to(self.device)
        fake_z = self.critic_z(z_)
        fake_z = torch.squeeze(fake_z)
        critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).to(self.device) * fake_z)

        enc_z = self.encoder(x)
        gen_x = self.decoder(enc_z)

        mse = self.mse_loss(x.float(), gen_x.float())
        loss_dec = mse + critic_score_valid_z - critic_score_fake_z
        loss_dec.backward(retain_graph=True)
        self.optim_dec.step()

        return loss_dec

    def score(self, test_x):
        test_loader = convert_to_dl(test_x, self.conf.batch_size)
        reconstruction_error = list()
        critic_score = list()

        for batch, (sample, _) in enumerate(test_loader):
            sample = sample.squeeze(dim=1).float().to(self.device)
            reconstructed_signal = self.decoder(self.encoder(sample))
            reconstructed_signal = torch.squeeze(reconstructed_signal)

            reconstruction_error.append(torch.abs(reconstructed_signal - sample))
            # for i in range(0, 64):
            #     x_ = reconstructed_signal[i].detach().numpy()
            #     x = sample[i].cpu().numpy()
            # reconstruction_error.append(self.dtw_reconstruction_error(x, x_))
            critic_score.append(torch.squeeze(self.critic_x(sample), dim=1))
            # critic_score.extend(.detach().numpy())
        reconstruction_error = torch.sum(torch.concat(reconstruction_error), dim=1)
        reconstruction_error = stats.zscore(reconstruction_error.cpu().detach().numpy())
        # reconstruction_error = stats.zscore(reconstruction_error.detach().numpy())

        critic_score = torch.concat(critic_score)
        critic_score = stats.zscore(critic_score.cpu().detach().numpy())
        anomaly_score = reconstruction_error * critic_score
        return anomaly_score

    # def dtw_reconstruction_error(self, x, x_):
    #     import
    #     n, m = x.shape[0], x_.shape[0]
    #     dtw_matrix = np.zeros((n + 1, m + 1))
    #     for i in range(n + 1):
    #         for j in range(m + 1):
    #             dtw_matrix[i, j] = np.inf
    #     dtw_matrix[0, 0] = 0
    #
    #     for i in range(1, n + 1):
    #         for j in range(1, m + 1):
    #             cost = abs(x[i - 1] - x_[j - 1])
    #             # take last min from a square box
    #             last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
    #             dtw_matrix[i, j] = cost + last_min
    #     return dtw_matrix[n][m]
