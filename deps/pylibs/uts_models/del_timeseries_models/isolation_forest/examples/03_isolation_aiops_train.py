from pylibs.utils.sklearn.util_sklearn import save_sk_model
from pylibs.utils.util_numpy import enable_numpy_reproduce
from timeseries_models.isolation_forest.examples.isolation_aiops_lib import get_best_conf, get_worst_conf, \
    BEST_MODEL_NAME, get_worst_kpi_data, get_best_kpi_data, WORST_MODEL_NAME
from timeseries_models.isolation_forest.isolation_forest_model import IsolationForestModel

enable_numpy_reproduce(1)

# 431a8542 D7
# 6d1114ae D13
train_x, train_y, valid_x, valid_y, test_x, test_y = get_best_kpi_data()
conf = get_best_conf()
best_if = IsolationForestModel(conf)
best_if.fit(train_x)
save_sk_model(best_if, BEST_MODEL_NAME)
# the score has been negative.
train_x, train_y, valid_x, valid_y, test_x, test_y = get_worst_kpi_data()
conf = get_worst_conf()
best_if = IsolationForestModel(conf)
best_if.fit(train_x)
save_sk_model(best_if, WORST_MODEL_NAME)
