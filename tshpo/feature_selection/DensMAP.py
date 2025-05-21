"""
pip install umap-learn
TSNEkhorn
"""
import umap

"""
问题：torchdr 在很多数据集上会出现: ValueError: [TorchDR] ERROR Affinity: NaN at iter 92, consider decreasing the learning rate.
解决： 换用umap
"""
import numpy as np
from tshpo.feature_selection.api import FS
from tshpo.lib_class import ExpConf


class DensMAP(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        # err: ValueError: [TorchDR] ERROR Affinity: NaN at iter 92, consider decreasing the learning rate.

        train_len = X_train.shape[0]
        k = self.check_n_feature(X_train.shape[1] * econf.feature_selec_rate)
        tran_data: np.ndarray = np.concatenate([X_train, X_test])
        tran_data = np.nan_to_num(tran_data, nan=0)

        # model
        clf = umap.UMAP(n_components=k, densmap=True)
        # clf = TN(n_components=k)
        X_train_trans = clf.fit_transform(tran_data)
        X_train_new = X_train_trans[:train_len]
        X_test_new = X_train_trans[train_len:]
        return X_train_new, y_train, X_test_new, y_test
