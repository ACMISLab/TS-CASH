"""
exp_index	sr	VUS_ROC
0	8	0.616815804
1	16	0.578078863
2	32	0.60975608
3	64	0.725382617
4	128	0.738366176
5	256	0.70718911
6	512	0.814988886
7	1024	0.812873488


"""
from pylibs.uts_dataset.dataset_loader import KFoldDatasetLoader
from pylibs.uts_models.benchmark_models.pyod.models.auto_encoder import AutoEncoder
model = AutoEncoder(batch_size=128, epochs=50)
kf=KFoldDatasetLoader()
train_x, train_y, test_x, test_y=kf.load_train_and_test()
model.fit(train_x)