import numpy as np
import torch
import torch.nn.functional as F

from pylibs.utils.util_log import get_logger
from pylibs.utils.util_message import logs
from pylibs.utils.util_metrics import affiliation_metrics
from pylibs.utils.util_numpy import fill_nan_inf
from pylibs.utils.util_pytorch import get_l_out_of_max_pool_1d, convert_to_dl
from pylibs.utils.util_system import UtilSys

log = get_logger()
_PRECISION_INDEX = -3
_RECALL_INDEX = -2
_F1_INDEX = -1


class _COCA(torch.nn.Module):
    def __init__(self, configs, device):
        super(_COCA, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.features_len = configs.features_len
        self.device = device
        self.project_channels = configs.project_channels
        self.hidden_size = configs.hidden_size
        self.window_size = configs.window_size
        self.num_layers = configs.num_layers
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout

        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                            stride=self.stride, bias=False, padding=(self.kernel_size // 2)),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(self.dropout)
        )

        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            torch.nn.BatchNorm1d(self.final_out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv1d(64, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            torch.nn.BatchNorm1d(self.final_out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.encoder = torch.nn.LSTM(
            self.final_out_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout,
        )
        self.decoder = torch.nn.LSTM(
            self.final_out_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout,
        )

        self.output_layer = torch.nn.Linear(self.hidden_size, self.final_out_channels)
        self.project = torch.nn.Linear(self.final_out_channels * self.features_len, self.project_channels, bias=False)

        # L_out of Conv1d of conv_block1
        # l1_out_conv1d = get_l_out_of_max_pool_1d(
        #     self.window_size,
        #     kernel_size=self.kernel_size,
        #     stride=self.stride,
        #     padding=self.kernel_size // 2
        # )
        l1_out_conv1d = get_l_out_of_max_pool_1d(self.window_size, kernel_size=self.kernel_size, stride=self.stride,
                                                 padding=self.kernel_size // 2)
        l1_out_maxpool1d = get_l_out_of_max_pool_1d(l1_out_conv1d, kernel_size=2, stride=2, padding=1)

        l2_out_conv1d = get_l_out_of_max_pool_1d(l1_out_maxpool1d, kernel_size=8, stride=1, padding=4)
        l2_out = get_l_out_of_max_pool_1d(l2_out_conv1d, kernel_size=2, stride=2, padding=1)
        project_input_features = self.final_out_channels * l2_out
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(project_input_features,
                            project_input_features // 2),
            torch.nn.BatchNorm1d(project_input_features // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(project_input_features // 2, self.project_channels),
        )

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x_in):
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        # 1D CNN feature extraction

        x = self.conv_block1(x_in.float())
        x = self.conv_block2(x)

        # Encoder
        hidden = x.permute(0, 2, 1)

        # enc_hidden = self._init_hidden_state(hidden.shape[0])
        _, enc_hidden = self.encoder(hidden)
        # _, enc_hidden = self.encoder(hidden, enc_hidden)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(hidden.shape).to(self.device)
        for i in reversed(range(hidden.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(hidden[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        hidden = hidden.reshape(hidden.size(0), -1)
        output = output.reshape(output.size(0), -1)

        # hidden: torch.Size([6, 435]), feature: 232
        # feature: self.final_out_channels * self.features_len

        project = self.projection_head(hidden)
        rec_project = self.projection_head(output)

        # project: the project of hidden(representation) Z
        # rec_project: the project of the output (reconstruction representation)
        return project, rec_project

    def forward_v1(self, x_in):
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        # 1D CNN feature extraction
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)

        # _s = {
        #     "x_in": torch.sum(x_in),
        #     "x": torch.sum(x)
        # }
        # log.info(f"forward feature========")
        # log.info(pprint.pformat(_s))
        # Encoder
        hidden = x.permute(0, 2, 1)
        # enc_hidden = self._init_hidden_state(hidden.shape[0])
        _, enc_hidden = self.encoder(hidden)
        # _, enc_hidden = self.encoder(hidden, enc_hidden)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(hidden.shape).to(self.device)
        for i in reversed(range(hidden.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(hidden[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        hidden = hidden.reshape(hidden.size(0), -1)
        output = output.reshape(output.size(0), -1)

        # hidden: torch.Size([6, 435]), feature: 232
        # feature: self.final_out_channels * self.features_len
        project = self.projection_head(hidden)
        rec_project = self.projection_head(output)

        # project: the project of hidden(representation) Z
        # rec_project: the project of the output (reconstruction representation)
        return project, rec_project


class COCAModel:
    """A modified version of the COCA: https://github.com/ruiking04/COCA"""

    def __init__(self, config):
        self.device = config.device
        assert self.device is not None
        self.center = None
        self.length = None
        self.config = config
        self.coca = _COCA(self.config, self.device).to(self.device)
        logs("COCA is on device: [{}]".format(self.device))

        self.optimizer = torch.optim.Adam(
            self.coca.parameters(), lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay)

    def score_coca(self, data):
        _, val_score_origin, _, _ = \
            self._model_evaluate(data, self.center, self.length)
        return val_score_origin

    def score(self, test_x):
        data = convert_to_dl(test_x, self.config.batch_size)
        return self.score_coca(data).cpu().numpy()

    def _center_c_v1(self, train_loader):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        self.coca.eval()
        center = torch.zeros(self.config.project_channels, device=self.device)
        n_samples = 0
        c = center
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                data, target, aug1, aug2 = data
                data = data.float().to(self.device)
                aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)
                all_data = torch.cat((data, aug1, aug2), dim=0)
                outputs, dec = self.coca(all_data)
                n_samples += outputs.shape[0]
                all_feature = torch.cat((outputs, dec), dim=0)
                # all_feature = outputs
                c += torch.sum(all_feature, dim=0)
        c /= (2 * n_samples)
        # c /= (n_samples)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        eps = self.config.center_eps
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def _center_c(self, train_loader):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        self.coca.eval()
        center = torch.zeros(self.config.project_channels, device=self.device)
        n_samples = 0
        c = center
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                data, _ = data
                data = data.float().to(self.device)
                outputs, dec = self.coca(data)
                n_samples += outputs.shape[0]
                all_feature = torch.cat((outputs, dec), dim=0)
                # all_feature = outputs
                c += torch.sum(all_feature, dim=0)
        c /= (2 * n_samples)
        # c /= (n_samples)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        eps = self.config.center_eps
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def predict(self, test_dl):
        if self.center is None:
            raise RuntimeError("Must fit the model before predict.")
        val_target, val_score_origin, val_loss, all_projection = self._model_evaluate(test_dl, self.center, self.length)
        return val_score_origin

    def _model_evaluate_v1(self, test_dl, center, length):
        self.coca.eval()
        total_loss, total_f1, total_precision, total_recall = [], [], [], []
        all_target, all_score = [], []
        all_projection = []
        with torch.no_grad():
            for data, target, aug1, aug2 in test_dl:
                data, target = data.float().to(self.device), target.long().to(self.device)
                assert self.coca.training == False
                feature1, feature_dec1 = self.coca(data)
                # feature1, feature_dec1, center, length
                loss, score = self._get_loss(feature1, feature_dec1, center, length)
                total_loss.append(loss.item())
                target = target.reshape(-1)
                all_target.append(target)
                all_score.append(score)
                all_projection.append(feature1)

        total_loss = torch.tensor(total_loss).mean()  # average loss
        all_projection = torch.cat(all_projection, dim=0)

        return all_target, all_score, total_loss, all_projection
        # all_target: 数据标签 label
        # all_score: score, a float value = dist(project of z, center)+ dist(project of output, center)
        # total_loss: a sum of a set of loss in each batch
        # all_projection: project of z

    def _model_evaluate(self, test_dl, center, length):

        self.coca.eval()
        total_loss, total_f1, total_precision, total_recall = [], [], [], []
        all_target, all_score = [], []
        all_projection = []
        with torch.no_grad():
            for data, target in test_dl:
                data, target = data.float().to(self.device), target.long().to(self.device)
                assert self.coca.training == False
                feature1, feature_dec1 = self.coca(data)
                # feature1, feature_dec1, center, length
                loss, score = self._get_loss(feature1, feature_dec1, center, length)
                total_loss.append(loss)
                all_target.append(target)
                all_score.append(score)
                all_projection.append(feature1)

        return torch.concat(all_target, 0), \
            torch.concat(all_score, 0), \
            torch.stack(total_loss).mean(), \
            torch.concat(all_projection, 0)
        # all_target: 数据标签 label
        # all_score: score, a float value = dist(project of z, center)+ dist(project of output, center)
        # total_loss: a sum of a set of loss in each batch
        # all_projection: project of z

    def fit(self, train_x, y=None):
        UtilSys.is_debug_mode() and log.info("Training started ....")
        # save_path = "./best_network/" + self.config.m_dataset
        # os.makedirs(save_path, exist_ok=True)
        train_dl = convert_to_dl(train_x, self.config.batch_size)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        center = self._center_c(train_dl)
        # log.info(self.center)
        length = torch.tensor(0, device=self.device)  # radius R initialized with 0 by default.
        for epoch in range(1, self.config.num_epoch + 1):
            # train_loader, center, length, epoch
            train_target, train_score, train_loss, length = \
                self._model_train(train_dl, center, length, epoch)
            if epoch < self.config.change_center_epoch:
                center = self._center_c(train_dl)
            # log.info(f"Length:{length}")
            scheduler.step(train_loss)
            UtilSys.is_debug_mode() and log.info(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}')
            # Update the loss
            self.train_loss = train_loss
        self.length = length
        self.center = center

    def fit_v1(self, train_dl, valid_dl=None, label=None):
        UtilSys.is_debug_mode() and log.info("Training started ....")
        # save_path = "./best_network/" + self.config.m_dataset
        # os.makedirs(save_path, exist_ok=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        center = self._center_c(train_dl)
        # log.info("center================================================================ ")
        # log.info(self.center)
        length = torch.tensor(0, device=self.device)  # radius R initialized with 0 by default.
        for epoch in range(1, self.config.num_epoch + 1):
            # train_loader, center, length, epoch
            train_target, train_score, train_loss, length = \
                self._model_train(train_dl, center, length, epoch)
            val_target, val_score_origin, val_loss, all_projection = \
                self._model_evaluate(valid_dl, center, length)

            if epoch < self.config.change_center_epoch:
                center = self._center_c(train_dl)
            # log.info(f"Length:{length}")
            scheduler.step(train_loss)
            UtilSys.is_debug_mode() and log.info(
                f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

            # Update the loss
            self.train_loss = train_loss
            self.valid_loss = val_loss
        self.length = length
        self.center = center

    def _get_radius(self, dist: torch.Tensor, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        dist = dist.reshape(-1)
        return torch.quantile(dist, 1 - nu)

    def _get_loss(self, feature1, feature_dec1, center, length):
        # normalize feature vectors,
        # loss for d(Q,Q'), invariance term
        center = center.unsqueeze(0)
        center = F.normalize(center, dim=1)
        feature1 = F.normalize(feature1, dim=1)
        feature_dec1 = F.normalize(feature_dec1, dim=1)

        distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
        distance_dec1 = F.cosine_similarity(feature_dec1, center, eps=1e-6)
        distance1 = 1 - distance1
        distance_dec1 = 1 - distance_dec1

        # Prevent model collapse
        # variance term, avoids "hypersphere collapse", v(Q)
        sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
        sigma_aug2 = torch.sqrt(distance_dec1.var([0]) + 0.0001)
        sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
        sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
        loss_sigam = torch.mean((sigma_loss1 + sigma_loss2) / 2)

        # The Loss function that representations reconstruction
        score = distance1 + distance_dec1
        if self.config.objective == 'soft-boundary':
            diff1 = score - length
            loss_oc = length + (1 / self.config.nu) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
        else:
            loss_oc = torch.mean(score)
        # The loss of invariance and variance term, a weighted average of the
        # invariance and variance.
        loss = self.config.omega1 * loss_oc + self.config.omega2 * loss_sigam
        return loss, score

    def _model_train_v1(self, train_loader, center, length, epoch):
        total_loss, total_f1, total_precision, total_recall = [], [], [], []
        all_target, all_score = [], []
        self.coca.train()
        # torch.autograd.set_detect_anomaly(True)
        for batch_idx, (data, target, aug1, aug2) in enumerate(train_loader):
            # send to device
            data, target = data.float().to(self.device), target.long().to(self.device)
            aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)
            all_data = torch.cat((data, aug1, aug2), dim=0)
            # optimizer
            self.optimizer.zero_grad()

            #  project of z(feature1),
            #  project of output(feature_dec1)
            assert self.coca.training == True
            feature1, feature_dec1 = self.coca(all_data)
            loss, score = self._get_loss(feature1, feature_dec1, center, length)

            # Update hypersphere radius R on mini-batch distances
            if (self.config.objective == 'soft-boundary') and (epoch >= self.config.freeze_length_epoch):
                length = torch.tensor(self._get_radius(score, self.config.nu), device=self.device)
            total_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            target = target.reshape(-1)

            all_target.append(target)
            all_score.append(score)

        total_loss = torch.tensor(total_loss).mean()

        return all_target, all_score, total_loss, length

    def _model_train(self, train_loader, center, length, epoch):
        total_loss, total_f1, total_precision, total_recall = [], [], [], []
        all_target, all_score = [], []
        self.coca.train()
        # torch.autograd.set_detect_anomaly(True)
        for batch_idx, (data, target) in enumerate(train_loader):

            # fix error with batch_size = 1
            if data.shape[0] < 2:
                continue
            # send to device
            data, target = data.float().to(self.device), target.long().to(self.device)
            # optimizer
            self.optimizer.zero_grad()

            #  project of z(feature1),
            #  project of output(feature_dec1)
            # assert self.coca.training == True
            # data.shape=torch.Size([256, 1, 64])
            # error: data.shape=torch.Size([1, 1, 64])

            feature1, feature_dec1 = self.coca(data)
            loss, score = self._get_loss(feature1, feature_dec1, center, length)

            # summ = {
            #     'data': torch.sum(data),
            #     'data_value': data,
            #     'targets': torch.sum(target),
            #     'aug1': torch.sum(aug1),
            #     'aug2': torch.sum(aug1),
            #     "all_data": torch.sum(all_data),
            #     "feature1": torch.sum(feature1),
            #     "feature_dec1": torch.sum(feature_dec1),
            #     "center": torch.sum(center),
            #     "length": length,
            # }
            # log.info("batch features====\n" + pprint.pformat(summ))

            # Update hypersphere radius R on mini-batch distances
            if (self.config.objective == 'soft-boundary') and (epoch >= self.config.freeze_length_epoch):
                # length = torch.tensor(self._get_radius(score, self.config.nu).clone().detach(), device=self.device)
                length = self._get_radius(score, self.config.nu).clone().detach()
            total_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            target = target.reshape(-1)
            all_target.append(target)
            all_score.append(score)

        total_loss = torch.tensor(total_loss).mean()
        return all_target, all_score, total_loss, length

    def ad_predict_v3_coca(self, target, scores, log_best=True):

        """
        Return best_affiliation, predict_label.

        This function only return affiliation metrics

        Parameters
        ----------
        log_best : bool
            Whether to log the best metrics
        target : np.ndarray
        scores : np.ndarray

        Returns
        -------
        """
        _aff_all = []
        # _percentiles = np.arange(0.0, 10.5, 0.5)
        _percentiles = np.arange(1, 301) / 1e3
        # nu_list is in [0.001,0.30]
        for _percentile in _percentiles:
            _threshold = np.percentile(scores, 100 - _percentile)
            predict = np.where(scores > _threshold, 1, 0)

            # Affiliation metrics
            aff_precision, aff_recall, aff_f1 = affiliation_metrics(target, predict)
            _aff_all.append([_percentile, _threshold, aff_precision, aff_recall, aff_f1])

        # Find the maximum affiliation F1 score
        best_affiliation = self._find_max_metric(_aff_all)

        # best_affiliation is 5 dimension array:
        # 0: (0-based): _percentile(the best percentile with best f1)
        # 1: _threshold(the threshold(float) corresponding to the best _percentile) ,
        # 2: precision(metric),
        # 3: recall(metric),
        # 4: f1(metric)
        return best_affiliation

    def _gather_metrics_v3(self, valid_dl, test_dl, app_dt: DateTimeHelper):
        UtilSys.is_debug_mode() and log.info("Calculating model metrics...")
        app_dt.evaluate_start()
        val_target, val_score_origin, val_loss, all_projection = \
            self._model_evaluate(valid_dl, self.center, self.length)
        test_target, test_score_origin, test_loss, all_projection = \
            self._model_evaluate(test_dl, self.center, self.length)
        self.valid_loss = val_loss
        self.test_loss = test_loss

        val_target = val_target.cpu().numpy()
        val_score_origin = val_score_origin.cpu().numpy()
        test_target = test_target.cpu().numpy()
        test_score_origin = test_score_origin.cpu().numpy()

        val_best_affiliation = self.ad_predict_v3(val_target, val_score_origin)
        test_best_affiliation = self.ad_predict_v3(test_target, test_score_origin)
        app_dt.evaluate_end()
        return {
            'valid_affiliation_f1': val_best_affiliation[_F1_INDEX],
            'valid_affiliation_precision': val_best_affiliation[_PRECISION_INDEX],
            'valid_affiliation_recall': val_best_affiliation[_RECALL_INDEX],
            'test_affiliation_f1': test_best_affiliation[_F1_INDEX],
            'test_affiliation_precision': test_best_affiliation[_PRECISION_INDEX],
            'test_affiliation_recall': test_best_affiliation[_RECALL_INDEX],
        }

    @staticmethod
    def _find_max_metric(_aff_all):
        _aff_all = np.asarray(_aff_all)
        _index_aff_all_max = np.argmax(fill_nan_inf(_aff_all[:, -1], -9999))
        best_affiliation = _aff_all[_index_aff_all_max]
        return best_affiliation

    def ad_predict_v3(self, target: np.ndarray, scores: np.ndarray, log_best=True):

        """
        Return best_affiliation, predict_label.

        This function only return affiliation metrics

        Parameters
        ----------
        log_best : bool
            Whether to log the best metrics
        target : np.ndarray
        scores : np.ndarray

        Returns
        -------
        """
        _aff_all = []
        _percentiles = np.arange(0.0, 10.5, 0.5)
        # _percentiles = np.arange(1, 301) / 1e3
        # nu_list is in [0.001,0.30]
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        for _percentile in _percentiles:
            _threshold = np.percentile(scores, 100 - _percentile)
            predict = np.where(scores > _threshold, 1, 0)

            # Affiliation metrics
            aff_precision, aff_recall, aff_f1 = affiliation_metrics(target, predict)
            _aff_all.append([_percentile, _threshold, aff_precision, aff_recall, aff_f1])

        # Find the maximum affiliation F1 score
        best_affiliation = self._find_max_metric(_aff_all)

        # best_affiliation is 5 dimension array:
        # 0: (0-based): _percentile(the best percentile with best f1)
        # 1: _threshold(the threshold(float) corresponding to the best _percentile) ,
        # 2: precision(metric),
        # 3: recall(metric),
        # 4: f1(metric)
        return best_affiliation

    def ad_predict_v3_coca(self, target, scores, log_best=True):

        """
        Return best_affiliation, predict_label.

        This function only return affiliation metrics

        Parameters
        ----------
        log_best : bool
            Whether to log the best metrics
        target : np.ndarray
        scores : np.ndarray

        Returns
        -------
        """
        _aff_all = []
        # _percentiles = np.arange(0.0, 10.5, 0.5)
        _percentiles = np.arange(1, 301) / 1e3
        # nu_list is in [0.001,0.30]
        for _percentile in _percentiles:
            _threshold = np.percentile(scores, 100 - _percentile)
            predict = np.where(scores > _threshold, 1, 0)

            # Affiliation metrics
            aff_precision, aff_recall, aff_f1 = affiliation_metrics(target, predict)
            _aff_all.append([_percentile, _threshold, aff_precision, aff_recall, aff_f1])

        # Find the maximum affiliation F1 score
        best_affiliation = self._find_max_metric(_aff_all)

        # best_affiliation is 5 dimension array:
        # 0: (0-based): _percentile(the best percentile with best f1)
        # 1: _threshold(the threshold(float) corresponding to the best _percentile) ,
        # 2: precision(metric),
        # 3: recall(metric),
        # 4: f1(metric)
        return best_affiliation

    def _gather_metrics_v3_coca(self, valid_dl, test_dl, app_dt: DateTimeHelper):
        UtilSys.is_debug_mode() and log.info("Calculating model metrics...")
        app_dt.evaluate_start()
        val_target, val_score_origin, val_loss, all_projection = \
            self._model_evaluate(valid_dl, self.center, self.length)
        test_target, test_score_origin, test_loss, all_projection = \
            self._model_evaluate(test_dl, self.center, self.length)
        self.valid_loss = (val_loss)
        self.test_loss = (test_loss)

        val_best_affiliation = self.ad_predict_v3_coca(val_target, val_score_origin)
        test_best_affiliation = self.ad_predict_v3_coca(test_target, test_score_origin)
        app_dt.evaluate_end()
        return {
            'valid_affiliation_f1': val_best_affiliation[_F1_INDEX],
            'valid_affiliation_precision': val_best_affiliation[_PRECISION_INDEX],
            'valid_affiliation_recall': val_best_affiliation[_RECALL_INDEX],
            'test_affiliation_f1': test_best_affiliation[_F1_INDEX],
            'test_affiliation_precision': test_best_affiliation[_PRECISION_INDEX],
            'test_affiliation_recall': test_best_affiliation[_RECALL_INDEX],
        }

    def report_metrics(self, valid_dl, test_dl, dt: DateTimeHelper = DateTimeHelper()):
        metrics = self._gather_metrics_v3(valid_dl, test_dl, dt)
        metrics.update(dt.collect_metrics())
        metrics['default'] = metrics.get("test_affiliation_f1")
        # log.info(pprint.pformat(metrics))
        return metrics

    def report_metrics_coca(self, valid_dl, test_dl, dt: DateTimeHelper = DateTimeHelper()):
        metrics = self._gather_metrics_v3_coca(valid_dl, test_dl, dt)
        metrics.update(dt.collect_metrics())
        metrics['default'] = metrics.get("test_affiliation_f1")
        # log.info(pprint.pformat(metrics))
        return metrics
