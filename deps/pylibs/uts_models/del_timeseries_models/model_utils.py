from typing import Union
import torch.utils.data
from merlion.utils import TimeSeries
from pylibs.dataloader_torch.dataset import data_generator_from_timeseries, \
    data_generator_from_timeseries_with_sample
from pylibs.ts_datasets.anomaly import NAB
from pylibs.util_dataset import DatasetType
from pylibs.utils.util_log import get_logger
from ts_datasets.anomaly import IOpsCompetition

log = get_logger()


class ModelType:
    VAE = "VAE"
    DEMO = "DEMO"
    COCA = "COCA"
    PCI = "PCI"
    OCSVM = "OCSVM"
    IsolationForest = "IsolationForest"

    @staticmethod
    def is_pci(args):
        if args.model.upper() == ModelType.PCI.upper():
            return True
        else:
            return False


def parse_model(model: str, hpconfig_file: Union[str, dict, BaseConfig], device: str):
    """
    Parse the model.

    Parameters
    ----------
    model : str
        The model name,e.g., VAE, COCA
    hpconfig_file : str
        The configuration file for the model, usually a json file.
    device : str
        The device to train, getting the device by `select_automatically_one_gpu_for_nni(gpus)`

    Returns
    -------


    """
    assert isinstance(model, str), f"model must be a string, got {type(model)}"
    upper_model = model.upper()
    # parse the model configuration
    # if upper_model == ModelType.OCSVM.upper():
    #     return OCSVMFactory(hpconfig_file, device).get_model()
    # if upper_model == ModelType.IsolationForest.upper():
    #     return IsolationForestFactory(hpconfig_file, device).get_model()
    # elif upper_model == ModelType.COCA.upper():
    #     return COCAFactory(hpconfig_file, device).get_model()
    # elif upper_model == ModelType.VAE.upper():
    #     return VAEFactory(hpconfig_file, device).get_model()
    # elif upper_model == ModelType.PCI.upper():
    #     return PCIFactory(hpconfig_file, device).get_model()
    # else:
    #     raise RuntimeError(f"Unsupported model {model}")


def parse_dataset(dataset, data_id, configs):
    """
    Return the m_dataset load of pytorch: train_loader, val_loader, test_loader, test_anomaly_window_num

    Parameters
    ----------
    data_sample_rate :
    data_sample_method :
    data_id : str
        The data id of the m_dataset
    dataset : str
        The KPI ID
    configs :

    Returns
    -------

    """
    available_dataset = ["IOpsCompetition", "NAB"]
    assert dataset in available_dataset, f"m_dataset is not supported, excepts {available_dataset}"
    if dataset == DatasetType.IOpsCompetition:
        dt = IOpsCompetition()
        time_series, meta_data = dt[dt.kpi_ids.exp_index(data_id)]
        train_data = TimeSeries.from_pd(time_series[meta_data.trainval])
        test_data = TimeSeries.from_pd(time_series[~meta_data.trainval])
        train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
        test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
        train_loader, val_loader, test_loader, test_anomaly_window_num = \
            data_generator_from_timeseries(train_data,
                                           test_data,
                                           train_labels,
                                           test_labels,
                                           configs)
    elif dataset == DatasetType.NAB:
        dt = NAB()
        time_series, meta_data = dt[0]
        train_data = TimeSeries.from_pd(time_series[meta_data.trainval])
        test_data = TimeSeries.from_pd(time_series[~meta_data.trainval])
        train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
        test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
        train_loader, val_loader, test_loader, test_anomaly_window_num = \
            data_generator_from_timeseries(train_data,
                                           test_data,
                                           train_labels,
                                           test_labels,
                                           configs)
    else:
        raise RuntimeError("Not supported m_dataset. ")
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    UtilSys.is_debug_mode() and log.info(f"train: {train_loader.dataset.len},"
                                         f"valid: {val_loader.dataset.len},"
                                         f"test: {test_loader.dataset.len}")
    return train_loader, val_loader, test_loader, test_anomaly_window_num


def parse_dataset_with_sample(dataset, data_id, configs, data_sample_method, data_sample_rate, ret_type=None,
                              valid_rate=0.5, processing="normalize", anomaly_window_type='time_step'):
    """
    Return the m_dataset load of pytorch: train_loader, val_loader, test_loader, test_anomaly_window_num

    Parameters
    ----------
    anomaly_window_type :  str
        time_step: The window is anomaly when the number of time_step data points in the front of the window contain
            at least anomaly data point. The strategy for COCA:
            R. Wang et al., “Deep Contrastive One-Class Time Series Anomaly Detection.” arXiv, Oct. 08, 2022. Accessed:
            Dec. 09, 2022. [Online]. Available: http://arxiv.org/abs/2207.01472

        any:The window is anomaly if the window contain at least one anomaly. The strategy for DONUT: H. Xu et al.,
        “Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications,”
        in Proceedings of the 2018 world wide web conference, 2018, pp. 187–196.

    valid_rate : float
        The percentage for validation data set
    processing : str
        The data processing method to use.
        Normalize if  processing="normalize": scale to [0,1]
        Standardize if processing="standardize": scale by mean and std

    ret_type : str
        One of torch_loader, dict. If ret_type is torch_loader, it returns the dataloader of pytorch, otherwise returns
        the dictionary of the m_dataset.
    data_sample_rate :
    data_sample_method :
    data_id : str
        The data id of the m_dataset
    dataset : str
        The KPI ID, One of IOpsCompetition or NAB
    configs :

    Returns
    -------

    """
    if ret_type is None:
        ret_type = "torch_loader"
    available_dataset = ["IOpsCompetition", "NAB"]
    assert dataset in available_dataset, f"m_dataset is not supported, excepts {available_dataset}"
    if dataset == DatasetType.IOpsCompetition:
        dt = IOpsCompetition()
        time_series, meta_data = dt[dt.kpi_ids.exp_index(data_id)]
        train_data = TimeSeries.from_pd(time_series[meta_data.trainval])
        test_data = TimeSeries.from_pd(time_series[~meta_data.trainval])
        train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
        test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
        train_loader, val_loader, test_loader, test_anomaly_window_num = \
            data_generator_from_timeseries_with_sample(train_data, test_data, train_labels, test_labels, configs,
                                                       data_sample_method, data_sample_rate, ret_type,
                                                       processing=processing, valid_rate=valid_rate,
                                                       anomaly_window_type=anomaly_window_type)
    elif dataset == DatasetType.NAB:
        dt = NAB()
        time_series, meta_data = dt[0]
        train_data = TimeSeries.from_pd(time_series[meta_data.trainval])
        test_data = TimeSeries.from_pd(time_series[~meta_data.trainval])
        train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
        test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
        train_loader, val_loader, test_loader, test_anomaly_window_num = \
            data_generator_from_timeseries_with_sample(train_data, test_data, train_labels, test_labels, configs,
                                                       data_sample_method, data_sample_rate, ret_type,
                                                       processing=processing, valid_rate=valid_rate,
                                                       anomaly_window_type=anomaly_window_type)
    else:
        raise RuntimeError("Not supported m_dataset. ")
    # assert isinstance(train_loader, torch.utils.data.DataLoader)
    # assert isinstance(val_loader, torch.utils.data.DataLoader)
    # assert isinstance(test_loader, torch.utils.data.DataLoader)
    return train_loader, val_loader, test_loader, test_anomaly_window_num
