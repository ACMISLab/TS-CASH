# todo: this code may be error.

from pylibs.uts_models.benchmark_models.lstm.lstm import LSTMModel
from pylibs.uts_dataset.dataset_loader import KFoldDatasetLoader

window_size = 99
dataset, data_id = "IOPS", "KPI-54350a12-7a9d-3ca8-b81f-f886b9d156fd.test.out"
dl = KFoldDatasetLoader(dataset, data_id,
                        window_size=window_size,
                        is_include_anomaly_window=False,
                        max_length=10000,
                        sample_rate=1)
train_x, train_y, origin_train_x, origin_train_y = dl.get_kfold_sliding_windows_train_and_test_by_fold_index()

clf = LSTMModel(slidingwindow=window_size, epochs=1)
clf.fit(train_x, train_x)

score = clf.score(origin_train_x)

print(score.sum(), score.std())
