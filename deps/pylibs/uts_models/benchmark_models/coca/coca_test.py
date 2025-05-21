import numpy as np
from numpy.testing import assert_almost_equal

from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_pytorch import enable_pytorch_reproduciable
from timeseries_models.coca.coca_config import COCAConf
from timeseries_models.coca.coca_model import COCAModel

enable_numpy_reproduce(1)
enable_pytorch_reproduciable(1)
conf = COCAConf()

vae = COCAModel(conf)
vae.fit(train_dataloader)
score = vae.score(test_dataloader)
_sum = np.sum(score)
assert_almost_equal(_sum, 1155.228271484375, decimal=4)
metric = vae.report_metrics(valid_dataloader, test_dataloader)
assert_almost_equal(metric['valid_affiliation_f1'], 0.7551478349841094)
assert_almost_equal(metric['test_affiliation_f1'], 0.6403283206383559)
