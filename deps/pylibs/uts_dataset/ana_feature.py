from pylibs.utils.util_gnuplot import UTSViewGnuplot
from pylibs.utils.util_numpy import dist_eu, feature_
from pylibs.uts_dataset.dataset_loader import KFoldDatasetLoader

dataset = "SMD"
TOP1 = "machine-3-11.test.csv@6.out"
# dataids = ["machine-2-1.test.csv@21.out",
#            "machine-2-3.test.csv@11.out",
#            "machine-2-3.test.csv@6.out",
#            "machine-2-4.test.csv@20.out",
#            "machine-2-7.test.csv@24.out",
#            "machine-2-8.test.csv@16.out",
#            "machine-2-8.test.csv@30.out",
#            "machine-2-9.test.csv@9.out",
#            "machine-3-10.test.csv@14.out",
#            "machine-3-10.test.csv@6.out",
#            "machine-3-11.test.csv@3.out",
#            "machine-3-11.test.csv@31.out",
#            "machine-3-11.test.csv@6.out",
#            "machine-3-6.test.csv@32.out",
#            "machine-3-8.test.csv@32.out", ]
from pylibs.utils.util_joblib import cache_


@cache_
def _load_dataset(_dataset_name, _data_id, _fold_index=-1, time_step=1, window_size=64):
    dl = KFoldDatasetLoader(dataset_name=_dataset_name, data_id=_data_id, window_size=window_size, time_step=time_step)
    return dl.get_kfold_sliding_windows_train_and_test_by_fold_index(_fold_index)


if __name__ == '__main__':

    train_x, train_y, test_x, test_y = _load_dataset(dataset, TOP1, 0)
    dist = feature_(train_x)
    UTSViewGnuplot().plot_a_sliding_window(train_x[:, -1], 0, save_file_name="smd_all.png")
    for _i, (_xi, _yi, _dist) in enumerate(zip(train_x, train_y, dist)):
        uv1 = UTSViewGnuplot()
        uv1.plot_a_sliding_window(_xi, _yi, save_file_name=f"{dataset}_{TOP1}_{round(_dist, 2)}_window_{_i}.png")
        # uv.plot_a_sliding_window(_xi, _yi)
