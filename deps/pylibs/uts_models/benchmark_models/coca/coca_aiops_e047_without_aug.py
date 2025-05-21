import argparse
import os
import random
from datetime import datetime

import numpy
import numpy as np
import torch
from merlion.utils import TimeSeries

from dataloader import data_generator1
from models.COCA.coca_network.model_no_aug import base_Model
from models.TS_TCC.TS_utils import _logger
from trainer_no_aug_edit import TrainerEdit
from ts_datasets.ts_datasets.anomaly import IOpsCompetition


def summary_dataset(dl: torch.utils.data.DataLoader):
    if hasattr(dl.dataset, "x_data"):
        data = dl.dataset.x_data
    else:
        data = dl.dataset.data
    res = "x.sum:{0},x.max:{1},x.min:{2},x.count:{3}" \
        .format(data.sum(), data.max(),
                data.min(), str(data.shape[0]))
    print(res)


# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--visualization', default=False, type=bool,
                    help='Visualize')
parser.add_argument('--seed', default=2, type=int,
                    help='seed value')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay (L2 penalty) hyperparameter for COCA objective')
parser.add_argument('--selected_dataset', default='UCR', type=str,
                    help='Dataset of choice: NAB, IOpsCompetition, SMAP, UCR')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cpu', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'COCA_no_aug'
run_description = args.run_description
selected_dataset = args.selected_dataset
weight_decay = args.weight_decay
visualization = args.visualization

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


class ConfigEdit(object):
    def __init__(self):
        # datasets
        self.dataset = 'IOpsCompetition'
        # model configs
        self.input_channels = 1
        self.kernel_size = 4
        self.stride = 1
        self.final_out_channels = 32
        self.hidden_size = 64
        self.num_layers = 3
        self.project_channels = 20

        self.dropout = 0.45
        self.features_len = 6
        self.window_size = 16
        self.time_step = 2

        # training configs
        self.num_epoch = 5
        self.freeze_length_epoch = 2
        self.change_center_epoch = 1

        self.center_eps = 0.1
        self.omega1 = 1
        self.omega2 = 0.1

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 512

        # Anomaly Detection parameters
        self.nu = 0.001
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0015
        # Methods for determining thresholds ("fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
        # Specify COCA objective ("one-class" or "soft-boundary")
        self.objective = 'soft-boundary'
        # Specify loss objective ("arc1","arc2","mix","no_reconstruction", or "distance")
        self.loss_type = 'distance'


configs = ConfigEdit()
# ##### fix random seeds for reproducibility ########
SEED = args.seed

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug("=" * 45)

log_dir = experiment_log_dir
# Load datasets
dt = IOpsCompetition()

# Get the lead & lag time for the dataset
early, delay = dt.max_lead_sec, dt.max_lag_sec
# Aggregate statistics from full dataset
all_anomaly_num, all_test_score, all_test_scores_reasonable = [], [], []
all_test_aff_score, all_test_aff_precision, all_test_aff_recall = [], [], []
detect_list = np.zeros(len(dt))
idx = -3


def enable_numpy_reproduce(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)


def enable_pytorch_reproduciable(seed=0):
    """
    For reproducible results.

    Parameters
    ----------
    seed :

    Returns
    -------

    """
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


enable_numpy_reproduce(1)
enable_pytorch_reproduciable(1)

logger.debug(str(idx) + "time series")
experiment_log_dir = os.path.join(log_dir, selected_dataset, '_' + str(idx))
dt = IOpsCompetition()
time_series, meta_data = dt[idx]
train_data = TimeSeries.from_pd(time_series[meta_data.trainval])
test_data = TimeSeries.from_pd(time_series[~meta_data.trainval])
train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])

print('>' * 32, len(train_data))
print('>' * 32, len(test_data))

# Load Model
model = base_Model(configs, device).to(device)
logger.debug("Data loaded ...")
train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator1(train_data, test_data, train_labels, test_labels,
                                                                     configs)
summary_dataset(train_dl)
summary_dataset(val_dl)
summary_dataset(test_dl)
model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                   weight_decay=weight_decay)

# Trainer
TrainerEdit(model, model_optimizer, train_dl,
            val_dl, test_dl, device, logger,
            configs, experiment_log_dir, idx)
