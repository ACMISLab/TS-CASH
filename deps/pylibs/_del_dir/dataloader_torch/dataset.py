import os
import pprint

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from merlion.transform.normalize import MeanVarNormalize, MinMaxNormalize
from merlion.utils import TimeSeries
from pylibs.common import SampleType
from pylibs.dataloader_torch.ConfigBase import ConfigBase
from pylibs.dataloader_torch.augmentations import DataTransform
from pylibs.utils.util_args import parse_str_to_float
from pylibs.utils.util_log import get_logger
from pysampling.sample import sample

log = get_logger()


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # if X_train.shape.exp_index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if hasattr(config, 'augmentation'):
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if hasattr(self, 'aug1'):
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_from_timeseries(train_data: TimeSeries, test_data: TimeSeries, train_labels: TimeSeries,
                                   test_labels: TimeSeries, configs: ConfigBase):
    """
    Return the dataloader of pytorch: train_loader, val_loader, test_loader, test_anomaly_window_num

    Parameters
    ----------
    train_data :
    test_data :
    train_labels :
    test_labels :
    configs :

    Returns
    -------

    """
    train_time_series_ts = train_data
    test_time_series_ts = test_data
    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts + test_time_series_ts)
    bias, scale = mvn.bias, mvn.scale
    train_time_series = train_time_series_ts.to_pd().to_numpy()
    train_time_series = (train_time_series - bias) / scale
    test_time_series = test_time_series_ts.to_pd().to_numpy()
    test_time_series = (test_time_series - bias) / scale

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_time_series, configs.window_size, configs.time_step)
    test_x = subsequences(test_time_series, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y_window, test_size=0.2, shuffle=False)
    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y_window

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)
    # _s = {
    #     "train_dataset feature": np.sum(train_dataset.aug1) + np.sum(train_dataset.aug2.cpu().numpy()) + np.sum(
    #         train_dataset.x_data.cpu().numpy()) + np.sum(train_dataset.y_data.cpu().numpy()),
    #     "val_dataset feature": np.sum(val_dataset.aug1) + np.sum(val_dataset.aug2.cpu().numpy()) + np.sum(
    #         val_dataset.x_data.cpu().numpy()) + np.sum(val_dataset.y_data.cpu().numpy()),
    #     "test_dataset feature": np.sum(test_dataset.aug1) + np.sum(test_dataset.aug2.cpu().numpy()) + np.sum(
    #         test_dataset.x_data.cpu().numpy()) + np.sum(test_dataset.y_data.cpu().numpy()),
    #     "configs": configs
    # }
    # log.info("data================================")
    # log.info(pprint.pformat(_s))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                             shuffle=False, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num


def data_generator_from_timeseries_with_sample(train_data: TimeSeries, test_data: TimeSeries, train_labels: TimeSeries,
                                               test_labels: TimeSeries, configs: ConfigBase, data_sample_method,
                                               data_sample_rate, ret_type="torch_loader", valid_rate=0.2,
                                               processing="normalize", anomaly_window_type="time_step"):
    """
    Return the dataloader of pytorch: train_loader, val_loader, test_loader, test_anomaly_window_num

    Parameters
    ----------
    anomaly_window_type : str
        time_step: The window is anomaly when the number of time_step data points in the front of the window contain
            at least anomaly data point. The strategy for COCA:
            R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly Detection.” arXiv, Oct. 08, 2022. Accessed:
            Dec. 09, 2022. [Online]. Available: http://arxiv.org/abs/2207.01472

        any:The window is anomaly if the window contain at least one anomaly. The strategy for DONUT: H. Xu et al.,
        “Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications,”
        in Proceedings of the 2018 world wide web conference, 2018, pp. 187–196.


    processing : str
        The data processing method to use.
        Normalize if  processing="standardize": scale to [0,1]
        Standardize if processing="normalize": scale by mean and std
    ret_type : str
        One of torch_loader, dict
    data_sample_method :
    data_sample_rate :
    train_data :
    test_data :
    train_labels :
    test_labels :
    configs :

    Returns
    -------

    """
    train_time_series_ts = train_data
    test_time_series_ts = test_data
    if processing == "normalize":
        mvn = MeanVarNormalize()
        mvn.train(train_time_series_ts + test_time_series_ts)
        bias, scale = mvn.bias, mvn.scale
        train_time_series = train_time_series_ts.to_pd().to_numpy()
        train_time_series = (train_time_series - bias) / scale
        test_time_series = test_time_series_ts.to_pd().to_numpy()
        test_time_series = (test_time_series - bias) / scale
    elif processing == "standardize":
        stand = MinMaxNormalize()
        stand.train(train_time_series_ts + test_time_series_ts)
        bias, scale = stand.bias, stand.scale
        train_time_series = train_time_series_ts.to_pd().to_numpy()
        train_time_series = (train_time_series - bias) / scale
        test_time_series = test_time_series_ts.to_pd().to_numpy()
        test_time_series = (test_time_series - bias) / scale
    else:
        raise RuntimeError(f"Unsupported processing method = {processing}, expect normalize or standardize")

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_time_series, configs.window_size, configs.time_step)
    test_x = subsequences(test_time_series, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    if anomaly_window_type == "time_step":
        for i, item in enumerate(train_y[:]):
            if sum(item[:configs.time_step]) >= 1:
                train_anomaly_window_num += 1
                train_y_window[i] = 1
            else:
                train_y_window[i] = 0
        for i, item in enumerate(test_y[:]):
            if sum(item[:configs.time_step]) >= 1:
                test_y_window[i] = 1
            else:
                test_y_window[i] = 0
    elif anomaly_window_type == "any":
        for i, item in enumerate(train_y[:]):
            if sum(item) >= 1:
                train_anomaly_window_num += 1
                train_y_window[i] = 1
            else:
                train_y_window[i] = 0
        for i, item in enumerate(test_y[:]):
            if sum(item) >= 1:
                test_y_window[i] = 1
            else:
                test_y_window[i] = 0
    else:
        raise RuntimeError(f"Unsupported anomaly_window_type = {anomaly_window_type}, expect any or time_step")
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y_window, test_size=valid_rate, shuffle=False)
    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    float_date_sample_rate = parse_str_to_float(data_sample_rate)
    train_dat_dict = dict()
    # sampling a subset samples to train the model,
    if data_sample_method == SampleType.RANDOM:
        if float_date_sample_rate < 1:
            n_points = int(train_x.shape[0] * float_date_sample_rate)
            sample_index = np.asarray(sample("random", n_points, 1) * train_x.shape[0], dtype=int).reshape(-1)
            train_dat_dict["samples"] = train_x[sample_index]
            train_dat_dict["labels"] = train_y[sample_index]
        else:
            train_dat_dict["samples"] = train_x
            train_dat_dict["labels"] = train_y

    elif data_sample_method is None:
        train_dat_dict["samples"] = train_x
        train_dat_dict["labels"] = train_y
    else:
        raise ValueError(f"data_sample_method is not supported. received {data_sample_method}")

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y_window

    if ret_type == "dict":
        return train_dat_dict, val_dat_dict, test_dat_dict, test_anomaly_window_num
    elif ret_type == 'torch_loader':
        train_dataset = Load_Dataset(train_dat_dict, configs)
        val_dataset = Load_Dataset(val_dat_dict, configs)
        test_dataset = Load_Dataset(test_dat_dict, configs)
        # _s = {
        #     "train_dataset feature": np.sum(train_dataset.aug1) + np.sum(train_dataset.aug2.cpu().numpy()) + np.sum(
        #         train_dataset.x_data.cpu().numpy()) + np.sum(train_dataset.y_data.cpu().numpy()),
        #     "val_dataset feature": np.sum(val_dataset.aug1) + np.sum(val_dataset.aug2.cpu().numpy()) + np.sum(
        #         val_dataset.x_data.cpu().numpy()) + np.sum(val_dataset.y_data.cpu().numpy()),
        #     "test_dataset feature": np.sum(test_dataset.aug1) + np.sum(test_dataset.aug2.cpu().numpy()) + np.sum(
        #         test_dataset.x_data.cpu().numpy()) + np.sum(test_dataset.y_data.cpu().numpy()),
        #     "configs": configs
        # }
        # log.info("data================================")
        # log.info(pprint.pformat(_s))
        UtilSys.is_debug_mode() and log.info(f"Data describe:\n"
                                             f"data points:\n"
                                             f"train: {train_x.shape}\n"
                                             f"valid: {val_x.shape}\n"
                                             f"test: {test_x.shape}\n\n"

                                             f"rolling windows:\n"
                                             f"train (all): {train_x.shape},\n"
                                             f"train (sampling): {train_dat_dict['samples'].shape},\n"
                                             f"valid: {val_x.shape},\n"
                                             f"test: {test_x.shape}\n"
                 f"config: {pprint.pformat(configs.__dict__)}\n"
                 )
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                   shuffle=True, drop_last=configs.drop_last,
                                                   num_workers=0)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                                 shuffle=False, drop_last=False,
                                                 num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                                  shuffle=False, drop_last=False,
                                                  num_workers=0)
        return train_loader, val_loader, test_loader, test_anomaly_window_num
    else:
        raise RuntimeError(f"Unsupported datatype for type={ret_type}")


def data_generator5(train_data, test_data, train_labels, test_labels, configs):
    train_time_series_ts = train_data
    test_time_series_ts = test_data

    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts + test_time_series_ts)
    bias, scale = mvn.bias, mvn.scale
    train_time_series = train_time_series_ts.to_pd().to_numpy()
    train_time_series = (train_time_series - bias) / scale
    test_time_series = test_time_series_ts.to_pd().to_numpy()
    test_time_series = (test_time_series - bias) / scale

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_time_series, configs.window_size, configs.time_step)
    test_x = subsequences(test_time_series, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, shuffle=False)
    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                             shuffle=False, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num


# according to window size and time step to create subsequence
def subsequences(sequence, window_size, time_step):
    # An array of non-contiguous memory is converted to an array of contiguous memory
    sq = np.ascontiguousarray(sequence)
    a = (sq.shape[0] - window_size + time_step) % time_step
    # label array
    if sq.ndim == 1:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size)
        stride = sq.itemsize * np.array([time_step * 1, 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a]
    # data array
    elif sq.ndim == 2:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size, sq.shape[1])
        stride = sq.itemsize * np.array([time_step * sq.shape[1], sq.shape[1], 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a, :]
    else:
        print('Array dimension error')
        os.exit()
    sq = np.lib.stride_tricks.as_strided(sq, shape=shape, strides=stride)
    return sq
