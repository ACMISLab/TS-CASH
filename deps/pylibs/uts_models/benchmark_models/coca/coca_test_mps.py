"""
Run on macOS mps
"""
import os

import numpy as np
from numpy.testing import assert_almost_equal

from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.utils.util_pytorch import enable_pytorch_reproduciable
from timeseries_models.coca.coca_config import COCAConf
from timeseries_models.coca.coca_model import COCAModel

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
enable_numpy_reproduce(1)
enable_pytorch_reproduciable(1)
conf = COCAConf()
da = DatasetAIOps2018(kpi_id=AIOpsKPIID.TEST_LINE_VAGUE,
                      windows_size=conf.window_size,
                      is_include_anomaly_windows=False,
                      sampling_rate=1,
                      valid_rate="0.2"
                      )
train_dataloader, valid_dataloader, test_dataloader = da.get_pydl_windows_3splits_with_origin_label_coca(
    conf.batch_size)
vae = COCAModel(conf, device="mps")
vae.fit(train_dataloader)
score = vae.score(test_dataloader)
_sum = np.mean(score)
assert_almost_equal(_sum, 1.1340135, decimal=4)
metric = vae.report_metrics(valid_dataloader, test_dataloader)
assert_almost_equal(metric['valid_affiliation_f1'], 0.7551478349841094, decimal=2)
assert_almost_equal(metric['test_affiliation_f1'], 0.7180521547689381, decimal=2)
