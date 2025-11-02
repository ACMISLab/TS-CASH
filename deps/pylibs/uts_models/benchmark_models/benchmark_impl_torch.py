import time

from pylibs.experiments.exp_config import ExpConf
from pylibs.experiments.exp_helper import JobConfV1, train_models_kfold, train_models_kfold_v2

models = JobConfV1.MODEL_TORCH
# models = ['COCA']
for _sr in [10, 100, 1000]:
    for model_name in models:
        conf = ExpConf(
            model_name=model_name,
            dataset_name="SVDB",
            data_id="801.test.csv@1.out",
            exp_name=f"test_{time.time()}",
            epoch=51,
            data_sample_rate=_sr, fold_index=1,
            batch_size=1
        )

        train_models_kfold_v2(conf)
