# todo: this code may be error.

from numpy.testing import assert_almost_equal

from pylibs.uts_models.benchmark_models.cnn.cnn import CNNModel
from pylibs.uts_models.benchmark_models.vae.vae import VAEModel
from pylibs.uts_models.benchmark_models.vae.vae_conf import VAEConfig
from pylibs.utils.util_common import UtilComm
from pylibs.uts_dataset.dataset_loader import DatasetLoader

window_size = 99
max_length = 1000
dataset, data_id = "ECG", "MBA_ECG805_data.out"
# 获取训练数据，只有异常
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
logging.getLogger('tensorflow').disabled = True
dl_train = DatasetLoader(dataset, data_id, window_size=window_size,
                         is_include_anomaly_window=False,
                         max_length=max_length)
train_x, train_y = dl_train.get_sliding_windows()

# 获取测试数据，包含异常和正常
dl_test = DatasetLoader(dataset, data_id, window_size=window_size,
                        is_include_anomaly_window=True,
                        max_length=max_length)
test_x, test_y = dl_test.get_sliding_windows()
conf = VAEConfig()
conf.window_size = window_size
conf.epochs = 5
clf = CNNModel(slidingwindow=conf.window_size, epochs=conf.epochs, verbose=-1)
clf.fit(train_x)
score = clf.score(test_x)
path = clf.save_model(UtilComm.get_file_name("cnn"))

# load
new_vae = CNNModel(slidingwindow=conf.window_size, epochs=conf.epochs).load_model(path)
new_score = new_vae.score(test_x)
print(score.sum(), new_score.sum())
assert_almost_equal(score, new_score)
print("✅ Test passed")
