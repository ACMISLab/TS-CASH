# Daphnet&S09R01E4.test.csv@3.out
# Daphnet&S09R01E4.test.csv@2.out
# SMD&machine-2-8.test.csv@16.out
# SMD&machine-3-11.test.csv@6.out
import numpy as np

from pylibs.utils.util_gnuplot import UTSViewGnuplot
from pylibs.utils.util_numpy import feature_
from pylibs.utils.util_sampling import latin_hypercube_sampling
from pylibs.uts_dataset.ana_feature import _load_dataset

# dataset, id = "Daphnet", "S09R01E4.test.csv@3.out"
dataset, id = "SMD", "machine-3-11.test.csv@6.out"
train_x, train_y, test_x, test_y = _load_dataset(dataset, id, time_step=1)
train_x = np.concatenate([train_x, test_x])
dist = feature_(train_x)
UTSViewGnuplot().plot_a_sliding_window_and_dist(train_x[:, -1], dist, save_file_name=f"{dataset}_{id}_all.png")
_index_small = np.where(dist <= 0.33)[0]
_index_middle = np.where((dist > 0.33) & (dist < 0.67))[0]
_index_large = np.where(dist >= 0.67)[0]

_index_sampled_small = _index_small[latin_hypercube_sampling(_index_small.shape[0], 3)]
_index_sampled_mid = _index_middle[latin_hypercube_sampling(_index_middle.shape[0], 100)]
_index_sampled_large = _index_large[latin_hypercube_sampling(_index_large.shape[0], 3)]

_all_sampled = np.concatenate([_index_sampled_small, _index_sampled_mid, _index_sampled_large])
small = train_x[_index_small]
middle = train_x[_index_middle]
large = train_x[_index_large]
assert small.shape[0] + large.shape[0] + middle.shape[0] == train_x.shape[0]
print(dist)
