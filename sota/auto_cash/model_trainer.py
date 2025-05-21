import dataclasses
import os.path
import time

import pandas as pd
from autosklearn.pipeline.components.classification import ClassifierChoice
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from metrics import log_loss
from sota.auto_cash.alg_performance_db import KVDB
from sota.auto_cash.auto_cash_helper import is_number, get_model_args_from_dict, get_model_args_from_dict_by_model_name, \
    ModelTrainConf, ModelTrainValue
from tshpo.automl_libs import load_dataset_at_fold


class ModelTrainer:
    def __init__(self, conf: ModelTrainConf):
        self.conf = conf
        self.kvdb_home = "/Users/sunwu/SW-Research/AutoML-Benchmark/sota/"
        self.kvdb = KVDB(db_name="model_perf_db.dump", db_home=self.kvdb_home)

    def train(self) -> ModelTrainValue:
        conf = self.conf
        perf = self.kvdb.query_by_dict(self.conf.get_kvdb_dict())
        if perf is None:
            _model_hpys = conf.get_model_confs()
            print(f"🔍evaluating new config for {conf.model}...\n{_model_hpys}")
            X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=conf.dataset_name,
                                                                    n_fold=conf.n_fold,
                                                                    fold_index=conf.fold_index,
                                                                    seed=conf.seed)
            start = time.time()

            clf = ClassifierChoice.get_components()[conf.model](**_model_hpys)

            clf.fit(X_train, y_train)

            y_predict = clf.predict(X_test)
            y_predict_proba = clf.predict_proba(X_test)[:, 1]

            _f1 = f1_score(y_test, y_predict)
            _precision = precision_score(y_test, y_predict)
            _recall = recall_score(y_test, y_predict)

            _roc_auc = roc_auc_score(y_test, y_predict_proba)

            # _log_loss = log_loss(y_test, y_predict)
            _log_loss = -1

            _accuracy = accuracy_score(y_test, y_predict)

            elapse = time.time() - start
            res = ModelTrainValue(
                elapsed_seconds=elapse,
                f1=round(_f1, 4),
                precision=round(_precision, 4),
                recall=round(_recall, 4),
                roc_auc=round(_roc_auc, 4),
                log_loss=round(_log_loss, 4),
                accuracy=round(_accuracy, 4),
            )

            # self.to_cache(_model_conf, res)
            self.kvdb.add_by_dict(model_hpys=_model_hpys, model_name=conf.model, dataset=conf.dataset_name,
                                  fold_index=conf.fold_index, value=res.__dict__)
            return res


        else:
            print("✅loading from cache...")
            return perf
            # return ModelTrainValue(
            #     elapsed_seconds=perf['elapsed_seconds'],
            #     f1=perf['f1'],
            #     precision=perf['precision'],
            #     recall=perf['recall'],
            #     roc_auc=perf['roc_auc'],
            #     log_loss=-1,
            #     accuracy=perf['accuracy'],
            #     error_msg=perf['error_msg']
            # )

    def to_cache(self, model_conf: dict, v: ModelTrainValue):
        # item = get_model_args_from_dict_by_model_name(self.conf.get_model_confs(), self.conf.model, self.conf.seed)
        # item = self.conf.get_model_xargs()
        model_conf.update({
            "model": self.conf.model,
            "dataset": self.conf.dataset_name,
            "fold_index": self.conf.fold_index,
            "elapsed_seconds": v.elapsed_seconds,
            "f1": v.f1,
            "precision": v.precision,
            "recall": v.recall,
            "roc_auc": v.roc_auc,
            "log_loss": -1,
            "accuracy": v.accuracy,
            "error_msg": v.error_msg

        })

        tmp_db = os.path.join(self.kvdb_home, f"{self.conf.model}_tmp.csv")
        new_data = pd.DataFrame([model_conf])
        if os.path.exists(tmp_db):
            new_df = pd.concat([pd.read_csv(tmp_db, index_col=0), new_data]).drop_duplicates()
            new_df.to_csv(tmp_db)
        else:

            new_data.to_csv(tmp_db)
