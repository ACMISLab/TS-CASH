"""
pip install umap-learn annoy
# pip install torchdr
TSNEkhorn
"""
import umap
import numpy as np
from torchdr import UMAP as TN

from tshpo.feature_selection.api import FS
from tshpo.feature_selection.parampacmap import ParamPaCMAP
from tshpo.lib_class import ExpConf


class ParamRepulsor(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        # err: ValueError: [TorchDR] ERROR Affinity: NaN at iter 92, consider decreasing the learning rate.

        train_len = X_train.shape[0]
        k = self.check_n_feature(X_train.shape[1] * econf.feature_selec_rate)
        tran_data: np.ndarray = np.concatenate([X_train, X_test])
        tran_data = np.nan_to_num(tran_data, nan=0)

        # model
        clf = ParamPaCMAP(n_components=k)
        # X_low = reducer.fit_transform(tran_data)
        # clf = TN(n_components=k)
        X_train_trans = clf.fit_transform(tran_data)
        X_train_new = X_train_trans[:train_len]
        X_test_new = X_train_trans[train_len:]
        return X_train_new, y_train, X_test_new, y_test
