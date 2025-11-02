from pylibs.experiments.exp_config import ExpConf

conf = ExpConf()
train_x, train_y, test_x, test_y = conf.load_dataset_at_fold_k()
print(train_x)
