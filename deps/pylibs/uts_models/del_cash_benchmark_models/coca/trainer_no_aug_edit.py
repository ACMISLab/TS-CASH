import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from models import ad_predict
from models.COCA.coca_trainer.early_stopping import EarlyStopping

sys.path.append("../../COCA")


def TrainerEdit(model, model_optimizer, train_dl, val_dl, test_dl, device, logger, config, experiment_log_dir, idx):
    # Start training
    save_path = "./best_network/" + config.dataset
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, idx)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    all_epoch_train_loss, all_epoch_test_loss = [], []
    center = torch.zeros(config.project_channels, device=device)
    center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
    length = torch.tensor(0, device=device)  # radius R initialized with 0 by default.

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_target, train_score, train_loss, length = model_train(model, model_optimizer, train_dl, center,
                                                                    length, config, device, epoch)
        # val_target, val_score_origin, val_loss, all_projection = model_evaluate(model, val_dl, center, length, config,
        #                                                                         device, epoch)
        # test_target, test_score_origin, test_loss, all_projection = model_evaluate(model, test_dl, center, length,
        #                                                                            config, device, epoch)

        if epoch < config.change_center_epoch:
            center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
        scheduler.step(train_loss)
        logger.debug(
            f'Epoch : {epoch},Train Loss: {train_loss:.4f}'
        )
        # logger.debug(
        #     f'Epoch : {epoch},Train Loss: {train_loss:.4f} Valid Loss: {val_loss:.4f}Test Loss: {test_loss:.4f}'
        # )
        all_epoch_train_loss.append(train_loss.item())
        # all_epoch_test_loss.append(val_loss.item())

    # Isolate the validation process
    val_target, val_score_origin, val_loss, all_projection = model_evaluate(model, val_dl, center, length, config,
                                                                            device, epoch)
    test_target, test_score_origin, test_loss, all_projection = model_evaluate(model, test_dl, center, length,
                                                                               config, device, epoch)
    # according to scores to create predicting labels
    # floating
    val_affiliation, val_score, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                               config.detect_nu)
    test_affiliation, test_score, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                       config.detect_nu)

    logger.debug("\n################## Training is Done! #########################")
    logger.debug(
        f'Valid affiliation precision: {val_affiliation["precision"]},Valid affiliation recall: {val_affiliation["recall"]}\n')

    logger.debug(
        f'Test affiliation precision: {test_affiliation["precision"]}, Test affiliation recall: {test_affiliation["recall"]}\n')


def model_train(model, model_optimizer, train_loader, center, length, config, device, epoch):
    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []

    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        # send to device
        data = data.float().to(device)
        # optimizer
        model_optimizer.zero_grad()
        feature1, feature_dec1 = model(data)
        loss, score = train(feature1, feature_dec1, center, length, epoch, config, device)
        # Update hypersphere radius R on mini-batch distances
        if (config.objective == 'soft-boundary') and (epoch >= config.freeze_length_epoch):
            length = torch.tensor(get_radius(score, config.nu), device=device)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

        target = target.reshape(-1)

        predict = score.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        all_target.extend(target)
        all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()

    return all_target, all_predict, total_loss, length


def model_evaluate(model, test_dl, center, length, config, device, epoch):
    model.eval()
    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []
    all_projection = []
    with torch.no_grad():
        for data, target in test_dl:
            data, target = data.float().to(device), target.long().to(device)
            feature1, feature_dec1 = model(data)
            loss, score = train(feature1, feature_dec1, center, length, epoch, config, device)
            total_loss.append(loss.item())
            predict = score.detach().cpu().numpy()
            target = target.reshape(-1)
            all_target.extend(target.detach().cpu().numpy())
            all_predict.extend(predict)
            all_projection.append(feature1)

    total_loss = torch.tensor(total_loss).mean()  # average loss
    all_projection = torch.cat(all_projection, dim=0)
    all_target = np.array(all_target)

    return all_target, all_predict, total_loss, all_projection


def train(feature1, feature_dec1, center, length, epoch, config, device):
    # normalize feature vectors
    center = center.unsqueeze(0)
    center = F.normalize(center, dim=1)
    feature1 = F.normalize(feature1, dim=1)
    feature_dec1 = F.normalize(feature_dec1, dim=1)

    distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
    distance_dec1 = F.cosine_similarity(feature_dec1, center, eps=1e-6)
    distance1 = 1 - distance1
    distance_dec1 = 1 - distance_dec1

    # Prevent model collapse
    sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
    sigma_aug2 = torch.sqrt(distance_dec1.var([0]) + 0.0001)
    sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
    sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
    loss_sigam = torch.mean((sigma_loss1 + sigma_loss2) / 2)

    # The Loss function that representations reconstruction
    score = distance1 + distance_dec1
    if config.objective == 'soft-boundary':
        diff1 = score - length
        loss_oc = length + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
    else:
        loss_oc = torch.mean(score)
    loss = config.omega1 * loss_oc + config.omega2 * loss_sigam
    return loss, score


def center_c(train_loader, model, device, center, config, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = center
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            data, target = data
            data = data.float().to(device)
            outputs, dec = model(data)
            n_samples += outputs.shape[0]
            all_feature = torch.cat((outputs, dec), dim=0)
            c += torch.sum(all_feature, dim=0)

    c /= (2 * n_samples)
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    # return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)
