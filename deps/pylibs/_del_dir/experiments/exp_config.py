import base64
import copy
import json
import os
import pprint
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pylibs.exp_ana_common.ExcelOutKeys import EK
from pylibs.utils.datatime_helper import MetricsCollector, DTH
from pylibs.utils.util_base64 import UtilBase64
from pylibs.utils.util_common import UtilComm, UC
from pylibs.utils.util_directory import make_dirs
from pylibs.utils.util_feishu import FSUtil
from pylibs.utils.util_hash import get_str_hash
from pylibs.utils.util_joblib import JLUtil
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_sys import get_gpu_device
from pylibs.utils.util_system import UtilSys, US
from pylibs.uts_dataset.dataset_loader import KFoldDatasetLoader, DataProcessingType

log = get_logger()

from pylibs.utils.util_joblib import cache_

"""
The value of _SAVE_TYPE is either redis or file. 
File means save metrics and score to file; and redis means save score to file.
"""
# _SAVE_TYPE = "redis"
_SAVE_TYPE = "file"


@cache_
def _get_data_processing_time(is_include_anomaly_window,
                              dataset_name,
                              data_id,
                              data_sample_method,
                              data_sample_rate,
                              window_size,
                              processing,
                              anomaly_window_type,
                              test_rate,
                              kfold):
    dl = KFoldDatasetLoader(
        dataset_name,
        data_id,
        sample_rate=data_sample_rate,
        window_size=window_size,
        is_include_anomaly_window=is_include_anomaly_window,
        processing=processing,
        anomaly_window_type=anomaly_window_type,
        data_sampling_method=data_sample_method,
        test_rate=test_rate,
        fill_nan=True,
    )
    _data_processing_time = []
    for fold_index in range(kfold):
        dl.get_kfold_sliding_windows_train_and_test_by_fold_index(fold_index)
        _data_processing_time.append(dl.get_data_processing_time())
    # 五折交叉验证所需要的时间
    return np.sum(_data_processing_time)


def _load_data_set_v2(**kwargs):
    dl = KFoldDatasetLoader(
        **kwargs
    )
    return dl.load_train_and_test()


def _process_metrics(metrics: dict):
    if metrics is not None:
        if metrics.get("score"):
            del metrics["score"]
        if metrics.get("config"):
            del metrics['config']
        if metrics.get("fastuts_sample_rate"):
            del metrics['fastuts_sample_rate']

    return metrics


class ExpConf:
    CACHE_KEYS = [
        "data_id",
        "debug",
        "model_name",
        "dataset_name",
        "data_sample_method",
        "data_sample_rate",
        "anomaly_window_type",
        "fold_index",
        "window_size",
        "seed",
        "kfold",
        "epoch",
        "test_rate",
        "batch_size",
        "data_scale_beta",
    ]

    def __init__(self,
                 model_name: str = 'iforest',
                 exp_name: str = "exp_name",
                 dataset_name: str = 'IOPS',
                 data_id: str = "KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out",
                 data_sample_method: str = "random",
                 data_sample_rate=-1,
                 metrics_save_home: str = UtilComm.get_runtime_directory(),
                 test_rate: float = 0.2,
                 exp_index: int = -1,
                 exp_total: int = -1,
                 anomaly_window_type: str = "coca",
                 window_size: int = 64,
                 seed=None,
                 job_id=None,
                 is_send_message_to_feishu: bool = True,
                 epoch=100,
                 gpu_memory_limit=1024,
                 batch_size=64,
                 kfold=5,
                 fold_index=-1,
                 gpu_index=None,
                 verbose=0,
                 fastuts_sample_rate=None,
                 stop_alpha=0.001,
                 optimal_target="VUS_ROC",
                 data_scale_beta=None,
                 debug=US.is_debug_mode(),
                 **kwargs):
        """
        A configuration for an experiment

        Parameters
        ----------
        model_name:
        dataset_name :
        data_id :
        exp_index :
            The exp_index of this experiment
        exp_total :
            The number of experiments
        """
        self.debug = debug
        self.data_processing_time = None
        self.window_size = int(window_size)
        self.model_name = str(model_name).lower().strip()
        self.data_sample_method = data_sample_method
        self.data_sample_rate = float(data_sample_rate)
        self.dataset_name = dataset_name
        self.data_id = data_id
        self.exp_total = exp_total
        self.exp_index = exp_index
        self.exp_name = exp_name.lower()
        self.anomaly_window_type = anomaly_window_type
        self.metrics_save_home = metrics_save_home
        self.test_rate = test_rate
        self.seed = seed
        self.job_id = job_id
        self.kfold = kfold
        self.is_send_message_to_feishu = is_send_message_to_feishu
        self.gpu_index = gpu_index
        self.fastuts_sample_rate = fastuts_sample_rate
        # tell the runners which fold to run
        self.fold_index = fold_index
        self.stop_alpha = stop_alpha
        # Epoch for deep method
        self.epoch = epoch

        # GPU memory limit for deep model
        self.gpu_memory_limit = gpu_memory_limit

        self.batch_size = batch_size

        self.data_scale_beta = data_scale_beta

        self.verbose = verbose

        self.optimal_target = optimal_target

    @staticmethod
    def get_series_values_id(_data: pd.Series):
        return json.dumps(_data[ExpConf.CACHE_KEYS].values.tolist())

    def get_redis_key(self):
        _data = pd.Series(self.get_dict())
        return ExpConf.get_series_values_id(_data)

    def is_lstm_model(self):
        return self.model_name in ['lstm-ad', 'cnn', 'ae', 'vae']

    def key_dicts(self):
        return self._get_key_dicts()

    def get_plot_label(self):
        labels = []
        for _index, (key, val) in enumerate(self.key_dicts().items()):
            labels.append(f"{key}={val}")
            if _index > 0 and _index % 5 == 0:
                labels.append("\\n")
        return "".join(labels)

    def get_small_identifier(self):
        return f"{self.model_name}_{self.dataset_name}_{self.data_id}"

    def update_fold_index(self, fold_index):
        self.fold_index = fold_index

    def get_exp_id(self):
        exp_info_arr = [
            self.model_name,
            self.dataset_name,
            self.data_id,
            self.data_sample_method,
            self.data_sample_rate,
            self.test_rate,
            self.anomaly_window_type,
            self.window_size,
            self.seed,
            UtilSys.is_debug_mode(),
            self.epoch,
            self.batch_size,
            self.kfold,
            self.fold_index,
            self.test_rate,
            self.is_semi_supervised_model()
        ]

        if self.data_sample_rate != -1:
            exp_info_arr.append(self.data_scale_beta)
        return get_str_hash(str(exp_info_arr))

    def get_dict(self) -> dict:
        ret_dict = self.__dict__
        if self.debug is None:
            ret_dict["debug"] = US.is_debug_mode()
        return ret_dict

    def print_parameters(self):
        UtilSys.is_debug_mode() and log.info(f" 👍 Exp config: \n{self.get_pretty_parameters()}")

    def get_pretty_parameters(self):
        return pprint.pformat(self.get_parameters())

    def get_parameters(self):
        return self.__str__()

    def save_config_to_file(self):
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_configs", self.get_exp_id() + ".txt")
        make_dirs(os.path.dirname(file))
        with open(file, "w") as f:
            f.write(self.get_pretty_parameters())

    def get_metrics_file_name(self, ext=".bz2"):
        make_dirs(self.get_metrics_save_home())
        return os.path.abspath(os.path.join(self.get_metrics_save_home(),
                                            f"{self.get_exp_id()}_metrics{ext}"
                                            ))

    def get_score_file_name(self):
        return os.path.abspath(os.path.join(self.get_metrics_save_home(),
                                            f"{self.get_exp_id()}_score.npz"
                                            ))

    def get_redis_id(self, _type="score"):
        """
        Parameters
        ----------
        _type :  类型

        Returns
        -------

        """
        return f"{self.exp_name}_{self.get_exp_id()}_{_type}"

        # return self.get_redis().get(self.get_exp_id())

    def get_redis_score_id(self):
        return self.get_redis_id(_type="score")

    def get_redis_metrics_id(self):
        return self.get_redis_id(_type="metrics")

    def update_metrics_save_home(self, home=None):
        if home is None:
            self.metrics_save_home = UC.get_runtime_directory()
        else:
            self.metrics_save_home = home

    @DeprecationWarning
    def get_dump_file_path(self):
        _home = os.path.join(UtilComm.get_system_runtime(), "job_params")
        make_dirs(_home)
        return os.path.abspath(os.path.join(_home,
                                            f"{self.get_exp_id()}.joblib"
                                            ))

    def is_metrics_file_exists(self):
        return self.is_compute_metrics()

    def update_exp_total(self, n_experiments):
        self.exp_total = n_experiments

    def get_home_directory(self):
        return os.path.join(self.metrics_save_home, self.exp_name, self.model_name, self.dataset_name)

    def get_image_save_home(self):
        return os.path.join(self.metrics_save_home, self.exp_name, self.model_name, self.dataset_name, "images")

    def get_metrics_save_home(self):
        home = os.path.join(self.metrics_save_home, self.exp_name, self.model_name, self.dataset_name, "metrics")
        home = os.path.abspath(home)
        make_dirs(home)
        return home

    def get_model_save_home(self):
        return os.path.join(self.metrics_save_home, self.exp_name, self.model_name, self.dataset_name,
                            "models_checkpoints")

    def get_model_save_path(self):
        return os.path.join(self.get_model_save_home(), self.get_exp_id())

    @DeprecationWarning
    def is_finished(self):
        return self.is_computed_score()
        # if os.path.exists(self.get_metrics_file_name()) and os.path.getsize(self.get_metrics_file_name()) > 0:
        #     return True
        # else:
        #     return False

    @DeprecationWarning
    def is_computed_score(self):
        # replace by self.is_compute_metrics()

        if self.is_save_to_redis():
            # save to redis
            rd = self.get_redis()
            return rd.exist(self.get_redis_score_id())
        else:
            # save to file
            return os.path.exists(self.get_score_file_name())

    def load_metrics_as_dict(self):
        _t = pd.read_pickle(self.get_metrics_file_name())
        if _t.shape[1] == 0:
            return None
        else:
            return pd.Series(data=_t.iloc[:, 1].values, index=_t.iloc[:, 0]).to_dict()

    def delete_metrics_file(self):
        try:
            os.remove(self.get_metrics_file_name())
        except:
            traceback.print_exc()

    def save_metrics_to_file(self, metrics=None):
        metrics = _process_metrics(metrics)
        if self.is_save_to_redis():
            # save to redis
            if metrics is None:
                metrics = {}
            else:
                red = self.get_redis()
                red.set(self.get_redis_metrics_id(), json.dumps(metrics))

        else:
            # save to file
            if metrics is None:
                df = pd.DataFrame()
            elif isinstance(metrics, dict):
                df = pd.DataFrame({
                    "metric": metrics.keys(),
                    "val": metrics.values()
                })
            else:
                UtilSys.is_debug_mode() and log.error(f"❌❌❌ Unsupported metrics type: {type(metrics)}")
                return False

            self._save_metrics_to_bz2(df)

    def save_fastuts_metrics_to_file(self, df):
        if df is None:
            UtilSys.is_debug_mode() and log.error(
                f"Df is none!")
            return None

        path = os.path.join(self.get_metrics_save_home(), "../", "fast_uts_metrics", f"{self.get_exp_id()}_fastuts.csv")
        make_dirs(os.path.dirname(path))
        df.to_csv(path,
                  index=False)
        UtilSys.is_debug_mode() and log.info(
            f"✅✅✅ Df has saved to file: {os.path.abspath(path)}")

    def report_progress_to_feishu(self):
        if self.exp_index % (self.exp_total // 10) == 0 or self.exp_index == self.exp_total - 1:  # 每完成10%进度
            progress = self.exp_index / (self.exp_total // 10) * 10
            FSUtil.send_msg_to_feishu(
                f"Exp [{self.exp_name}] {progress}%  [{self.model_name}, {self.dataset_name}]")

    def is_semi_supervised_model(self):
        """
        如果是半监督，只能包含正常数据。

        监督和非监督按正常流程处理。监督包含label，非监督虽然包含label，但是没有用label。

        Returns
        -------

        """
        # fully unsupervised: IForest, IForest1, LOF, MP, NormA, PCA, HBOS, and POLY.
        # semi-supervised: OCSVM, AE, LSTM-AD, and CNN.
        #
        # ref: Theseus: Navigating the Labyrinth of Time-Series Anomaly Detection
        # We select 12 different AD methods, summarized in Table 1. Out of these, 8 are fully
        # unsupervised (i.e., they require no prior information on the anomalies to be detected): IForest, IForest1,
        # LOF, MP, NormA, PCA, HBOS, and POLY. The remaining 4 methods are semi-supervised (i.e., they require some
        # information related to the normal behavior): OCSVM, AE, LSTM-AD, and CNN.
        #
        return self.model_name in ['ocsvm', 'ae', 'vae', 'lstm-ad', 'cnn']

    def get_training_len(self):
        """
        获取训练窗口的梳理
        Returns
        -------

        """
        train_x, train_y, test_x, test_y = self.load_dataset_with_fold_k(0)
        return train_x.shape[0]

    @DeprecationWarning
    def dump(self):
        """
        Use load_exp_conf() to load.
        Returns
        -------

        """
        assert self.fold_index is not None, "Fold exp_index cannot be None"
        joblib.dump(self, self.get_dump_file_path())
        assert os.path.exists(self.get_dump_file_path())
        return self.get_dump_file_path()

    def is_debug_mode(self):
        return UtilSys.is_debug_mode()

    def print_data_info(self):
        pass
        # if UtilSys.is_debug_mode():
        #     log.info("Data info: ")
        #     train_x, train_y, test_x, test_y = self.load_csv()
        #     data_disc = [
        #         get_data_describe(train_x, data_type=f"train_x", model_name=self.model_name),
        #         get_data_describe(train_y, data_type="train_y", model_name=self.model_name),
        #         get_data_describe(test_x, data_type="test_x", model_name=self.model_name),
        #         get_data_describe(test_y, data_type="test_y", model_name=self.model_name),
        #     ]
        #     df = pd.DataFrame(data_disc, columns=['data_type', 'model_name', 'mean', 'std', 'min', 'max', 'count'])
        #     PDUtil.print_pretty_table_with_header(df)

    def get_is_include_anomaly(self):
        is_include_anomaly_window = True
        if self.is_semi_supervised_model():
            is_include_anomaly_window = False
        return is_include_anomaly_window

    def get_data_processing_time(self):
        return self.data_processing_time
        # return _get_data_processing_time(
        #     is_include_anomaly_window=self.get_is_include_anomaly(),
        #     dataset_name=self.dataset_name,
        #     data_id=self.data_id,
        #     data_sample_method=self.data_sample_method,
        #     data_sample_rate=self.data_sample_rate,
        #     window_size=self.window_size,
        #     processing=DataProcessingType.STANDARDIZATION,
        #     anomaly_window_type=self.anomaly_window_type,
        #     test_rate=self.test_rate,
        #     kfold=self.kfold
        # )

    def load_dataset_with_fold_k(self, fold_index):
        self._get_key_dicts()
        dl = KFoldDatasetLoader(
            dataset_name=self.dataset_name,
            data_id=self.data_id,
            sample_rate=self.data_sample_rate,
            window_size=self.window_size,
            is_include_anomaly_window=self.get_is_include_anomaly(),
            processing=DataProcessingType.STANDARDIZATION,
            anomaly_window_type=self.anomaly_window_type,
            data_sampling_method=self.data_sample_method,
            test_rate=self.test_rate,
            data_scale_beta=self.data_scale_beta
        )
        data = dl.get_kfold_sliding_windows_train_and_test_by_fold_index(fold_index)
        self.data_processing_time = dl.get_data_processing_time()
        return data

    def load_data_for_lstm_and_cnn(self):
        """
         dirty_train_sample_x, dirty_train_sample_y, clean_train_sample_x, clean_train_sample_y, test_x, test_y = dl.load_data_for_lstm_and_cnn()

        Returns
        -------

        """
        dl = KFoldDatasetLoader(
            dataset_name=self.dataset_name,
            data_id=self.data_id,
            sample_rate=self.data_sample_rate,
            window_size=self.window_size,
            is_include_anomaly_window=self.get_is_include_anomaly(),
            processing=DataProcessingType.STANDARDIZATION,
            anomaly_window_type=self.anomaly_window_type,
            data_sampling_method=self.data_sample_method,
            test_rate=self.test_rate,
            fill_nan=True,
            data_scale_beta=self.data_scale_beta,
            fold_index=self.fold_index
        )
        self.data_processing_time = dl.get_data_processing_time()
        return dl.get_kfold_data_for_lstm_cnn()

    def load_data(self):
        """
        train_x, train_y, test_x, test_y=c.load_csv()
        Returns
        -------

        """
        # return self.load_dataset_at_fold_k()
        return self._load_data_v2()

    def _load_data_v2(self):
        train_x, train_y, test_x, test_y, data_processing_time = _load_data_set_v2(
            dataset_name=self.dataset_name,
            data_id=self.data_id,
            sample_rate=self.data_sample_rate,
            window_size=self.window_size,
            is_include_anomaly_window=self.get_is_include_anomaly(),
            processing=DataProcessingType.STANDARDIZATION,
            anomaly_window_type=self.anomaly_window_type,
            data_sampling_method=self.data_sample_method,
            test_rate=self.test_rate,
            data_scale_beta=self.data_scale_beta,
            fold_index=self.fold_index
        )
        self.data_processing_time = data_processing_time
        return train_x, train_y, test_x, test_y

    def load_dataset_at_fold_k(self):
        """
        自动获取当前config 所对应的训练数据

        train_x, train_y, test_x, test_y=conf.load_dataset_at_fold_k()
        Returns
        -------

        """
        return self.load_dataset_with_fold_k(self.fold_index)

    def report_message_to_feishu(self, step=50):
        if self.exp_index % step == 0:
            FSUtil.send_msg_to_feishu(f"Exp {self.exp_name}: ({self.exp_index}/{self.exp_total})")

    def encode_configs_bs64(self):
        return base64.b64encode(json.dumps(self.__dict__).encode('utf-8')).decode("utf-8")

    def encode_key_params_bs64(self):
        _conf = _reprocess_config(self)
        return base64.b64encode(json.dumps(_conf._get_key_dicts()).encode('utf-8')).decode("utf-8")

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def generate_key_params_encode_for_joblib_key(self):
        return UtilBase64.encode_json({'window_size': self.window_size,
                                       'model_name': self.model_name,
                                       'data_sample_method': self.data_sample_method,
                                       'data_sample_rate': self.data_sample_rate,
                                       'dataset_name': self.dataset_name,
                                       'data_id': self.data_id,
                                       'anomaly_window_type': self.anomaly_window_type,
                                       'test_rate': self.test_rate,
                                       'seed': self.seed,
                                       'kfold': self.kfold,
                                       'fold_index': self.fold_index,
                                       'epoch': self.epoch,
                                       'batch_size': self.batch_size,
                                       'data_scale_beta': self.data_scale_beta,
                                       'debug': UtilSys.is_debug_mode(),
                                       "is_semi_supervised": self.is_semi_supervised_model()
                                       })

    @staticmethod
    def decode_configs_bs64(b64string):
        conf = ExpConf()
        _str = base64.b64decode(b64string).decode("utf-8")
        _config = json.loads(_str)
        for key, value in _config.items():
            if hasattr(conf, key):
                setattr(conf, key, value)
        return conf

    def is_available_data(self):
        pass

    def _get_key_dicts(self):
        return {'window_size': self.window_size,
                'model_name': self.model_name,
                'data_sample_method': self.data_sample_method,
                'data_sample_rate': self.data_sample_rate,
                'dataset_name': self.dataset_name,
                'data_id': self.data_id,
                'anomaly_window_type': self.anomaly_window_type,
                'test_rate': self.test_rate,
                'seed': self.seed,
                'kfold': self.kfold,
                'fastuts_sample_rate': self.fastuts_sample_rate,
                'fold_index': self.fold_index,
                'stop_alpha': self.stop_alpha,
                'epoch': self.epoch,
                'batch_size': self.batch_size,
                'data_scale_beta': self.data_scale_beta,
                'debug': UtilSys.is_debug_mode(),
                "is_semi_supervised": self.is_semi_supervised_model()
                }

    @DeprecationWarning
    def save_score_and_exp_conf(self, score):
        if score is None:
            score = []
        score = np.round(score, 3)
        UtilSys.is_debug_mode() and log.info(f"Saving score to {os.path.abspath(self.get_score_file_name())}")
        np.savez_compressed(self.get_score_file_name(), score=score, conf=[self.encode_key_params_bs64()])
        return self.get_score_file_name()

    def save_score_dth_exp_conf_to_npz(self, score: np.ndarray, dt: MetricsCollector):
        if dt is None:
            dt = MetricsCollector()
        if score is None:
            score = np.asarray([-1])

        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        # score = (score - score.min()) / (score.max() - score.min())
        score = np.round(score, 3)
        dt.end()
        UtilSys.is_debug_mode() and log.info(f"Saving score to {os.path.abspath(self.get_score_file_name())}")

        np.savez_compressed(self.get_score_file_name(),
                            score=score,
                            conf=[self.get_dict()],
                            time=[dt.collect_metrics()])
        return self.get_score_file_name()

    def save_score_dth_exp_conf(self, score: np.ndarray, dt: MetricsCollector):
        if self.is_save_to_redis():
            return self.save_score_dth_exp_conf_to_redis(score, dt)
        else:
            return self.save_score_dth_exp_conf_to_npz(score, dt)

    def save_score_dth_exp_conf_to_redis(self, score, dt: MetricsCollector):
        if score is None:
            score = []
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        score = np.round(score, 3)
        dt.end()

        saved_metrics = self.get_dict()
        saved_metrics.update({"id": self.get_exp_id()})
        saved_metrics.update({"score": score.tolist()})
        saved_metrics.update(dt.collect_metrics())
        red = self.get_redis()
        red.set(self.get_redis_score_id(), json.dumps(saved_metrics))
        return saved_metrics

    @staticmethod
    def load_score_and_exp_conf(npz_file):
        """
        Load score and conf from a .npz file

        score, conf=
        Parameters
        ----------
        npz_file :

        Returns
        -------

        """
        load = np.load(npz_file, allow_pickle=True)
        exp_conf = load['conf'][0]
        score = load['score']
        _time = load['time'][0]
        # return score, ExpConf.decode_configs_bs64(exp_conf), _time
        return score, ExpConf(**exp_conf), _time

    def get_metrics(self):
        if self.is_save_to_redis():
            return self.get_metrics_from_redis()
        else:
            return ExpConf.get_metrics_from_npz(self.get_score_file_name())

    @staticmethod
    def get_metrics_from_npz(npz_file):
        try:
            return _cal_metrics_from_file(npz_file)
        except:
            traceback.print_exc()
            return None

    def get_redis(self):
        return RedisUtil()

    def get_metrics_from_redis(self):
        return _cal_metrics_from_id(self.get_redis_score_id())

    def is_save_to_redis(self):
        return _SAVE_TYPE == "redis"

    def is_compute_metrics(self):
        if self.is_save_to_redis():
            # save to redis
            rd = self.get_redis()
            return rd.exist(self.get_redis_metrics_id())
        else:
            # save to file
            return os.path.exists(self.get_metrics_file_name())

    def load_model(self):
        return ModelUtils.load_model(self)

    def calculate_metrcs_by_conf_score_dth(self, conf, score, dth=DTH()):
        train_x, train_y, test_x, test_y = conf.load_csv()
        if np.max(test_y) == 0:
            UtilSys.is_debug_mode() and log.warn("Pass for all labels = 0")
            return None

        all_metrics = UTSMetricHelper.get_metrics_all(test_y, score, conf.window_size, metric_type="all")

        if all_metrics is None:
            return None
        # VUS_ROC, VUS_PR = UTSMetricHelper.vus_accuracy(score, test_y, _conf.window_size)
        else:
            _train_len = train_x.shape[0]
            _test_len = test_x.shape[0]
            metrics = conf.get_dict()
            metrics.update(dth.collect_metrics())
            metrics.update({
                EK.TRAIN_LEN: _train_len,
                EK.TEST_LEN: _test_len,
                EK.DATA_PROCESSING_TIME: conf.get_data_processing_time()
            })
            metrics.update(all_metrics)
            return metrics

    def _save_metrics_to_bz2(self, df):
        if UtilSys.is_debug_mode() and os.path.exists(self.get_metrics_file_name()):
            log.info(
                f"✅✅✅ OptMetricsType has saved to file: {os.path.abspath(self.get_metrics_file_name())}\n")
        else:
            log.info(f"✅✅✅ OptMetricsType has saved to file: {os.path.abspath(self.get_metrics_file_name())}\n")
        df.to_pickle(self.get_metrics_file_name(),
                     compression="bz2",
                     )

    def get_anomaly_rate_of_windows(self):
        train_x, train_y, test_x, test_y = self.load_data()
        return np.sum(train_y) / train_y.shape[0]


from pylibs.utils.util_joblib import cache_


@cache_
def _cal_metrics_from_file(npz_file):
    score, exp_conf, exp_time = ExpConf.load_score_and_exp_conf(npz_file)

    # When len(score) is 1, which means no scores according to conf.save_score_dth_exp_conf()
    if len(score) == 1:
        return None

    # _conf = ExpConf(**exp_conf)
    train_x, train_y, test_x, test_y = exp_conf.load_data()

    # test
    all_metrics = UTSMetricHelper.get_metrics_all(test_y, score, exp_conf.window_size, metric_type="all")

    if all_metrics is None:
        return None
    # VUS_ROC, VUS_PR = UTSMetricHelper.vus_accuracy(score, test_y, _conf.window_size)
    else:
        _train_len = train_x.shape[0]
        _test_len = test_x.shape[0]
        metrics = exp_conf.get_dict()
        metrics.update(exp_time)
        metrics.update({
            EK.TRAIN_LEN: _train_len,
            EK.TEST_LEN: _test_len,
            EK.DATA_PROCESSING_TIME: exp_conf.get_data_processing_time()
        })
        metrics.update(all_metrics)
        return metrics


@cache_
def _cal_metrics_from_id(id):
    from pylibs.utils.util_redis import RedisUtil
    rd = RedisUtil()
    data = rd.get(id)
    json_data = json.loads(data)
    score = json_data['score']
    score = np.asarray(score)
    del json_data['score']
    # score, exp_conf, exp_time = ExpConf.load_score_and_exp_conf(id)
    # _conf = ExpConf(**exp_conf)
    exp_conf = ExpConf(**json_data)
    train_x, train_y, test_x, test_y = exp_conf.load_data()

    # test
    all_metrics = UTSMetricHelper.get_metrics_all(test_y, score, exp_conf.window_size, metric_type="all")

    if all_metrics is None:
        return None
    # VUS_ROC, VUS_PR = UTSMetricHelper.vus_accuracy(score, test_y, _conf.window_size)
    else:
        _train_len = train_x.shape[0]
        _test_len = test_x.shape[0]
        metrics = exp_conf.get_dict()
        metrics.update({
            EK.TRAIN_LEN: _train_len,
            EK.TEST_LEN: _test_len,
            EK.DATA_PROCESSING_TIME: exp_conf.get_data_processing_time(),
            EK.ELAPSED_TRAIN: json_data[EK.ELAPSED_TRAIN]
        })

        metrics.update(all_metrics)
        return metrics


mem = JLUtil.get_memory()

from pylibs.utils.util_joblib import cache_


@cache_
def _load_data_set(is_include_anomaly_window,
                   dataset_name,
                   data_id,
                   data_sample_method,
                   data_sample_rate,
                   window_size,
                   processing,
                   anomaly_window_type,
                   test_rate,
                   fold_index,
                   data_scale_beta):
    dl = KFoldDatasetLoader(
        dataset_name,
        data_id,
        sample_rate=data_sample_rate,
        window_size=window_size,
        is_include_anomaly_window=is_include_anomaly_window,
        processing=processing,
        anomaly_window_type=anomaly_window_type,
        data_sampling_method=data_sample_method,
        test_rate=test_rate,
        fill_nan=True,
        data_scale_beta=data_scale_beta
    )

    return dl.get_kfold_sliding_windows_train_and_test_by_fold_index(fold_index)


# @mem.cache


@mem.cache
def _load_data_set_info(dataset_name, data_id, processing=DataProcessingType.STANDARDIZATION):
    kdl = KFoldDatasetLoader(dataset_name, data_id, processing=processing)
    return kdl.get_kfold_sliding_windows_train_and_test_by_fold_index(0)


@DeprecationWarning
def load_exp_conf(file_path) -> ExpConf:
    return joblib.load(file_path)


def _reprocess_config(conf: ExpConf) -> ExpConf:
    _conf = copy.deepcopy(conf)
    # 移除与模型性能不相关得参数(控制FastUTS的参数),保证已训练的模型的可重用性

    # 这3个参数与模型的性能无关. 他是用来决定FastUTS的停止条件的, 设置成什么, 对模型的性能都无所谓无所谓
    # 这里固定的目的是: 使用joblib 缓存同一个模型即可
    _conf.stop_alpha = 0.001
    _conf.fastuts_sample_rate = None
    _conf.optimal_target = "VUS_ROC"
    return _conf


class ModelUtils:
    MODELS = ['lstm-ad', 'cnn', 'lof', 'hbos', 'ocsvm', 'iforest', 'ae', 'vae', 'dagmm', ]

    def __init__(self):
        pass

    @staticmethod
    def load_model(conf):
        """

        Parameters
        ----------
        conf : ExpConf

        Returns
        -------

        """
        return load_model(conf)


def load_model(conf: ExpConf):
    if conf.model_name == "hbos":
        assert conf.is_semi_supervised_model() == False
        from pylibs.uts_models.benchmark_models.tsbuad.models.hbos import HBOS
        clf = HBOS()
    elif conf.model_name == "iforest":
        assert conf.is_semi_supervised_model() == False
        from pylibs.uts_models.benchmark_models.iforest.iforest import IForest
        clf = IForest()
    elif conf.model_name == "lof":
        assert conf.is_semi_supervised_model() == False
        from pylibs.uts_models.benchmark_models.lof.lof import LOF
        clf = LOF()
    elif conf.model_name == "ocsvm":
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.ocsvm.ocsvm import OCSVM
        clf = OCSVM()
    elif conf.model_name == "ae":
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.pyod.models.auto_encoder import AutoEncoder
        clf = AutoEncoder(batch_size=conf.batch_size, epochs=conf.epoch)
    elif conf.model_name == "vae":
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.pyod.models.vae import VAE
        clf = VAE(epochs=conf.epoch,
                  batch_size=conf.batch_size,
                  verbose=conf.verbose)
    elif conf.model_name == "lstm-ad":
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.tsbuad.models.lstm import lstm
        clf = lstm(slidingwindow=conf.window_size,
                   epochs=conf.window_size,
                   verbose=conf.verbose,
                   batch_size=conf.batch_size)
    elif conf.model_name == 'cnn':
        assert conf.is_semi_supervised_model() == True
        from pylibs.uts_models.benchmark_models.tsbuad.models.cnn import cnn
        clf = cnn(slidingwindow=conf.window_size, epochs=conf.epoch,
                  batch_size=conf.batch_size)

    elif conf.model_name == "dagmm":
        from pylibs.uts_models.benchmark_models.dagmm.dagmm_model import DAGMMConfig, DAGMM
        cf = DAGMMConfig()
        cf.sequence_len = conf.window_size
        cf.batch_size = conf.batch_size
        cf.num_epochs = conf.epoch
        cf.device = get_gpu_device()
        clf = DAGMM(cf)
    elif conf.model_name == "coca":
        from pylibs.uts_models.benchmark_models.coca.coca_config import COCAConf
        from pylibs.uts_models.benchmark_models.coca.coca_model import COCAModel
        cf = COCAConf()
        cf.window_size = conf.window_size
        cf.batch_size = conf.batch_size
        cf.num_epoch = conf.epoch
        cf.device = get_gpu_device()
        clf = COCAModel(cf)
    elif conf.model_name == "tadgan":
        from pylibs.uts_models.benchmark_models.tadgan.tadgan_model import TadGanEd, TadGanEdConf
        cf = TadGanEdConf()
        cf.signal_shape = conf.window_size
        cf.batch_size = conf.batch_size
        cf.num_epochs = conf.epoch
        cf.device = get_gpu_device()
        clf = TadGanEd(cf)

    else:
        raise RuntimeError(f"Unknown model {conf.model_name}")
    return clf


if __name__ == '__main__':
    #  model_name: str,
    #                  dataset_name: str,
    #                  data_id: str,
    #                  data_sample_method: str,
    #                  data_sample_rate: float,
    #                  exp_name: str,
    #                  metrics_save_home: str,

    # for i in range(10):
    #     a = ExpConf(model_name="as", dataset_name="IPO",
    #                 data_id="3", data_sample_method="random", data_sample_rate=0.2,
    #                 exp_index=i, exp_total=10,
    #                 exp_name="djls", metrics_save_home=UtilComm.get_system_runtime())
    #     a.report_progress_to_feishu()
    ori_conf = ExpConf(model_name="as", dataset_name="IOPS",
                       data_id="KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd.train.out",
                       data_sample_method="random",
                       data_sample_rate=time.time(),
                       exp_index=1, exp_total=10,
                       exp_name="djls",
                       metrics_save_home=UtilComm.get_system_runtime(),
                       test_rate=time.time_ns()
                       )
    # path = a.dump()
    # load_conf = load_exp_conf(path)
    # print(load_conf)
    # assert list(load_conf.__dict__.values()) == list(a.__dict__.values())
    # print(a.get_training_len())
    #
    # print(get_data_describe(np.asarray([1, 23]), a=1, b=2))
    b64 = ori_conf.encode_configs_bs64()
    print(b64)
    decode_conf = ori_conf.decode_configs_bs64(b64)

    assert decode_conf.get_exp_id() == ori_conf.get_exp_id()
    assert decode_conf.__dict__ == ori_conf.__dict__

    decode_conf = ori_conf.encode_key_params_bs64()

    conf = ExpConf()

    for k, v in conf._get_key_dicts().items():
        if hasattr(conf, k):
            setattr(conf, k, k + str(time.time()))
    encode_conf = conf.encode_key_params_bs64()
    decode_conf = conf.decode_configs_bs64(encode_conf)

    ori_conf.is_available_data()

    conf.save_metrics_to_file(None)
    dth = MetricsCollector()
    file = conf.save_score_dth_exp_conf(score=np.asarray([1, 234]), dt=dth)
    conf.save_score_dth_exp_conf_to_redis(score=np.asarray([1, 234]), dt=dth)
