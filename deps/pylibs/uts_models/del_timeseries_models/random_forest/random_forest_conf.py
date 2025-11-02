import numpy as np

from pylibs.utils.util_log import get_logger
from timeseries_models.base_conf import BaseModelConfig

log = get_logger()


# [1] P. Probst, M. Wright, A.-L. Boulesteix, Hyperparameters and Tuning Strategies for Random Forest,
# Wires Data Min. Knowl. Discovery. 9 (2019). https://doi.org/10/gf3sz2.

# The attributes from paper to sklearn
# mtry Number of drawn candidate variables in each split √p, p/3 for regression
# ✅max_samples: sample size Number of observations that are drawn for each tree n
# ❌ not found: replacement Draw observations with or without replacement TRUE (with replacement)
# ✅min_samples_leaf: node size Minimum number of observations in a terminal node 1 for classification, 5 for regression
# ✅n_estimators: number of trees Number of trees in the forest 500, 1000
# ✅criterion: splitting rule Splitting criteria in the nodes Gini impurity, p-value, random
#

class RandomForestConf(BaseModelConfig):
    def __init__(self, n_estimators=100, verbose=0):
        super().__init__()
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.criterion = "gini"
        self.max_features = "sqrt"  # sqrt, log2,
        self.max_samples = None

    def update_parameters(self, parameters: dict):
        # window_size
        # time_step
        # batch_size
        key_of_int_parameter = ['n_estimators', 'window_size', 'time_step', 'batch_size']
        count = 0
        for key, val in parameters.items():
            if hasattr(self, key):
                if key in key_of_int_parameter:
                    val = int(np.round(val))
                    assert isinstance(val, int)
                UtilSys.is_debug_mode() and log.info(f"Updating parameters {key:20} = {val}")
                setattr(self, key, val)
            else:
                raise KeyError(f"{__class__} has not the hyper-parameter key named {key} ")
            count += 1
        UtilSys.is_debug_mode() and log.info(f"The number of parameters updated: {count:10}")
