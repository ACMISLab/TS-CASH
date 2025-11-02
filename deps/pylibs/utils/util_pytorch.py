import os
import random
import numpy
import numpy as np
from torch.utils.data import DataLoader

from pylibs._del_dir.dataset.AIOpsDataset import AIOpsDataset
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_message import log_setting_msg
import torch

from pylibs.utils.util_system import UtilSys

log = get_logger()


def enable_pytorch_reproduciable(seed=0):
    """
    For reproducible results.

    Parameters
    ----------
    seed :

    Returns
    -------

    """
    log_setting_msg(f"Enable reproducible results for seed {seed}")
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def get_device_cpu():
    return 'cpu'
    # return "cpu"


def get_specify_gpus_from_str(gpus: str = None) -> list:
    """
    Limit the GPUs to work.

    If there have 5 available GPUs, '0,1' means it will not use the 2-4 GPUs, which only work on
    the 0 and 1 GPUs devices.

    Parameters
    ----------
    gpus : str
        The number of GPUs to limit to. e.g., '0,1' or '1,2' or '2,3'.
        None means it will use the all GPUs.

    Parameters
    ----------
    gpus :

    Returns
    -------

    """
    gpus = gpus.strip().strip(",")
    gpus_index = [int(i) for i in gpus.split(',')]
    return gpus_index


def get_l_out_of_max_pool_1d(l_in, kernel_size, padding=0, dilation=1, stride=1):
    """
    Get the L_out of MAXPOOL1D  https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    or CONV1D https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html


    Parameters
    ----------
    l_in :
    padding :
    dilation :
    kernel_size :
    stride :

    Returns
    -------

    """
    return int(1 + ((l_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride))


def get_l_out_of_conv1d(l_in, kernel_size, padding=0, dilation=1, stride=1):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

    Parameters
    ----------
    l_in :
    kernel_size :
    padding :
    dilation :
    stride :

    Returns
    -------

    """
    return get_l_out_of_max_pool_1d(l_in, kernel_size, padding, dilation, stride)


def get_select_one_gpu_index(exp_sequence, gpus: str) -> int:
    """

    Parameters
    ----------
    exp_sequence : int
        the sequence of experiment, exp_sequence is in [0,1,....,N], where N is the number of experiments.

    gpus : str
         The exp_index of the GPU to use for training. It will automate selecting a specified gpu to train the experiments.
         Multi-GPUs can be used by comma separated.
         e.g.,
         "gpus=0,1" means to use the first and second GPU to train the model.
         "gpus=-1" means to use CPUs to train the model


    Returns
    -------
    int
        The specified GPU to train this experiment.

    """
    UtilSys.is_debug_mode() and log.info("Specified gpus: %s" % gpus)
    if gpus == -1 or gpus == '-1':
        selected_gpu = 1
    else:
        gpus = get_specify_gpus_from_str(gpus)  # specify the gpus
        if len(gpus) > 0:
            target_gpus_index = exp_sequence % len(gpus)

            if target_gpus_index < 0:
                selected_gpu = 1
            else:
                selected_gpu = gpus[target_gpus_index]
        else:
            UtilSys.is_debug_mode() and log.info("Not found any GPU. ")
            selected_gpu = None

    UtilSys.is_debug_mode() and log.info(f"Target gpus exp_index: {selected_gpu}")
    return selected_gpu


def summary_dataset(dl: torch.utils.data.DataLoader):
    if hasattr(dl.dataset, "x_data"):
        data = dl.dataset.x_data
    else:
        data = dl.dataset.data
    res = "x.sum:{0},x.max:{1},x.min:{2},x.count:{3}" \
        .format(data.sum(), data.max(),
                data.min(), str(data.shape[0]))
    print(res)


def convert_to_dl(train_x, batch_size):
    return DataLoader(AIOpsDataset(np.expand_dims(train_x, 1), train_x), batch_size=batch_size)
