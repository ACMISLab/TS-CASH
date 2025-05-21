import time

from pylibs.experiments.exp_config import ExpConf
from pylibs.experiments.exp_helper import JobConfV1, train_models_kfold, train_models_kfold_v2, load_model

_sr = 10

from pylibs.utils.util_joblib import cache_


@cache_
def train(model_name):
    conf = ExpConf(
        model_name=model_name,
        dataset_name="SVDB",
        data_id="801.test.csv@1.out",
        exp_name=f"test_{time.time()}",
        epoch=1,
        data_sample_rate=3,
        fold_index=1,
        batch_size=2
    )
    train_x, train_y, test_x, test_y = conf.load_csv()
    test_x = test_x[:10]
    test_y = test_y[:10]

    model = load_model(conf)
    model.fit(train_x)
    score = model.score(test_x)
    print(score)


models = JobConfV1.MODEL_TORCH
# models = ['COCA']
for model_name in models:
    train(model_name)
