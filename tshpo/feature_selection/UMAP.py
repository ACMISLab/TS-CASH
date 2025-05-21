"""
pip install umap-learn
# pip install torchdr
TSNEkhorn
"""
import umap

"""
问题：在很多数据集上会出现: ValueError: [TorchDR] ERROR Affinity: NaN at iter 92, consider decreasing the learning rate.
============
dresses-sales:UMAP
UMAP failed on dataset dresses-sales 
============
climate-model-simulation-crashes:UMAP
(486, 18) (54, 18)
(486, 3) (54, 3)
============
cylinder-bands:UMAP
UMAP failed on dataset cylinder-bands 
============
ilpd:UMAP
UMAP failed on dataset ilpd 
============
credit-approval:UMAP
(621, 15) (69, 15)
(621, 3) (69, 3)
============
breast-w:UMAP
UMAP failed on dataset breast-w 
============
diabetes:UMAP
(691, 8) (77, 8)
(691, 3) (77, 3)
============
tic-tac-toe:UMAP
UMAP failed on dataset tic-tac-toe 
============
credit-g:UMAP
(900, 20) (100, 20)
(900, 3) (100, 3)
============
qsar-biodeg:UMAP
UMAP failed on dataset qsar-biodeg 
============
pc1:UMAP
UMAP failed on dataset pc1 
============
pc4:UMAP
UMAP failed on dataset pc4 
============
pc3:UMAP
UMAP failed on dataset pc3 
============
kc1:UMAP
UMAP failed on dataset kc1 
============
ozone-level-8hr:UMAP
UMAP failed on dataset ozone-level-8hr 
============
madelon:UMAP
UMAP failed on dataset madelon 
============
kr-vs-kp:UMAP
UMAP failed on dataset kr-vs-kp 
============
Bioresponse:UMAP
UMAP failed on dataset Bioresponse 
============
sick:UMAP
UMAP failed on dataset sick 
============
spambase:UMAP
UMAP failed on dataset spambase 
============
wilt:UMAP
UMAP failed on dataset wilt 
============
churn:UMAP
(4500, 20) (500, 20)
(4500, 3) (500, 3)
============
phoneme:UMAP
UMAP failed on dataset phoneme 
============
jm1:UMAP
UMAP failed on dataset jm1 
============
PhishingWebsites:UMAP
UMAP failed on dataset PhishingWebsites 
============
nomao:UMAP
UMAP failed on dataset nomao 
============
bank-marketing:UMAP
(40689, 16) (4522, 16)
(40689, 3) (4522, 3)
============
electricity:UMAP
(40780, 8) (4532, 8)
(40780, 3) (4532, 3)
"""
import numpy as np
from torchdr import UMAP as TN

from tshpo.feature_selection.api import FS
from tshpo.lib_class import ExpConf


class UMAPBak(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        # err: ValueError: [TorchDR] ERROR Affinity: NaN at iter 92, consider decreasing the learning rate.

        train_len = X_train.shape[0]
        k = self.check_n_feature(X_train.shape[1] * econf.feature_selec_rate)
        tran_data: np.ndarray = np.concatenate([X_train, X_test])
        tran_data = np.nan_to_num(tran_data, nan=0)

        # model
        clf = TN(n_components=k)
        X_train_trans = clf.fit_transform(tran_data)
        X_train_new = X_train_trans[:train_len]
        X_test_new = X_train_trans[train_len:]
        return X_train_new, y_train, X_test_new, y_test


class UMAP(FS):
    def transform(self, X_train, y_train, X_test, y_test, econf: ExpConf):
        # err: ValueError: [TorchDR] ERROR Affinity: NaN at iter 92, consider decreasing the learning rate.

        train_len = X_train.shape[0]
        k = self.check_n_feature(X_train.shape[1] * econf.feature_selec_rate)
        tran_data: np.ndarray = np.concatenate([X_train, X_test])
        tran_data = np.nan_to_num(tran_data, nan=0)

        # model
        clf = umap.UMAP(n_components=k)
        # clf = TN(n_components=k)
        X_train_trans = clf.fit_transform(tran_data)
        X_train_new = X_train_trans[:train_len]
        X_test_new = X_train_trans[train_len:]
        return X_train_new, y_train, X_test_new, y_test
