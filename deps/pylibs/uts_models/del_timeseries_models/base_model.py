import abc
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable

from merlion.evaluate.anomaly import accumulate_tsad_score, ScoreType, TSADScoreAccumulator
from merlion.utils import TimeSeries
from typeguard import typechecked

import nni
from pylibs.affiliation.generics import convert_vector_to_events
from pylibs.affiliation.metrics import pr_from_events
from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_message import log_metric_msg
from pylibs.utils.util_metrics import affiliation_metrics
from pylibs.utils.util_numpy import fill_nan_inf
from timeseries_models.base_conf import BaseModelConfig

log = get_logger()

_PRECISION_INDEX = -3
_RECALL_INDEX = -2
_F1_INDEX = -1


class ModelInterface:

    @abc.abstractmethod
    def predict(self, data):
        """
        Predict the data.

        Parameters
        ----------
        data :

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def fit(self, **kwargs):
        """
        Training the model.

        Parameters
        ----------

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def score(self, data):
        """
        Get anomaly score for each windows.

        Parameters
        ----------
        data :

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def report_metrics(self, **kwargs):
        """
        Report the metrics for the model.

        Parameters
        ----------

        Returns
        -------

        """
        pass

    @staticmethod
    def _find_max_metric(_aff_all):
        _aff_all = np.asarray(_aff_all)
        _index_aff_all_max = np.argmax(fill_nan_inf(_aff_all[:, -1], -9999))
        best_affiliation = _aff_all[_index_aff_all_max]
        return best_affiliation

    def ad_predict_v3(self, target: np.ndarray, scores: np.ndarray, log_best=True):

        """
        Return best_affiliation, predict_label.

        This function only return affiliation metrics

        Parameters
        ----------
        log_best : bool
            Whether to log the best metrics
        target : np.ndarray
        scores : np.ndarray

        Returns
        -------
        """
        _aff_all = []
        _percentiles = np.arange(0.0, 10.5, 0.5)
        # _percentiles = np.arange(1, 301) / 1e3
        # nu_list is in [0.001,0.30]
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        for _percentile in _percentiles:
            _threshold = np.percentile(scores, 100 - _percentile)
            predict = np.where(scores > _threshold, 1, 0)

            # Affiliation metrics
            aff_precision, aff_recall, aff_f1 = affiliation_metrics(target, predict)
            _aff_all.append([_percentile, _threshold, aff_precision, aff_recall, aff_f1])

        # Find the maximum affiliation F1 score
        best_affiliation = self._find_max_metric(_aff_all)

        if log_best:
            x = PrettyTable()
            x.field_names = ["type", "percentile", "threshold", "precision", "recall", "f1"]
            x.add_rows(
                [
                    np.concatenate([['affiliation'], best_affiliation]),
                ]
            )

            log_metric_msg(f"Found best metric:\n{x}")

        # best_affiliation is 5 dimension array:
        # 0: (0-based): _percentile(the best percentile with best f1)
        # 1: _threshold(the threshold(float) corresponding to the best _percentile) ,
        # 2: precision(metric),
        # 3: recall(metric),
        # 4: f1(metric)
        return best_affiliation

    @typechecked
    def ad_predict_v3(self, target: np.ndarray, scores: np.ndarray, log_best=True):

        """
        Return best_affiliation, predict_label.

        This function only return affiliation metrics

        Parameters
        ----------
        log_best : bool
            Whether to log the best metrics
        target : np.ndarray
        scores : np.ndarray

        Returns
        -------
        """
        _aff_all = []
        _percentiles = np.arange(0.0, 10.5, 0.5)
        # _percentiles = np.arange(1, 301) / 1e3
        # nu_list is in [0.001,0.30]
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        for _percentile in _percentiles:
            _threshold = np.percentile(scores, 100 - _percentile)
            predict = np.where(scores > _threshold, 1, 0)

            # Affiliation metrics
            aff_precision, aff_recall, aff_f1 = affiliation_metrics(target, predict)
            _aff_all.append([_percentile, _threshold, aff_precision, aff_recall, aff_f1])

        # Find the maximum affiliation F1 score
        best_affiliation = self._find_max_metric(_aff_all)

        if log_best:
            x = PrettyTable()
            x.field_names = ["type", "percentile", "threshold", "precision", "recall", "f1"]
            x.add_rows(
                [
                    np.concatenate([['affiliation'], best_affiliation]),
                ]
            )

            log_metric_msg(f"Found best metric:\n{x}")

        # best_affiliation is 5 dimension array:
        # 0: (0-based): _percentile(the best percentile with best f1)
        # 1: _threshold(the threshold(float) corresponding to the best _percentile) ,
        # 2: precision(metric),
        # 3: recall(metric),
        # 4: f1(metric)
        return best_affiliation

    def ad_predict_v3_coca(self, target, scores, log_best=True):

        """
        Return best_affiliation, predict_label.

        This function only return affiliation metrics

        Parameters
        ----------
        log_best : bool
            Whether to log the best metrics
        target : np.ndarray
        scores : np.ndarray

        Returns
        -------
        """
        _aff_all = []
        # _percentiles = np.arange(0.0, 10.5, 0.5)
        _percentiles = np.arange(1, 301) / 1e3
        # nu_list is in [0.001,0.30]
        for _percentile in _percentiles:
            _threshold = np.percentile(scores, 100 - _percentile)
            predict = np.where(scores > _threshold, 1, 0)

            # Affiliation metrics
            aff_precision, aff_recall, aff_f1 = affiliation_metrics(target, predict)
            _aff_all.append([_percentile, _threshold, aff_precision, aff_recall, aff_f1])

        # Find the maximum affiliation F1 score
        best_affiliation = self._find_max_metric(_aff_all)

        if log_best:
            x = PrettyTable()
            x.field_names = ["type", "percentile", "threshold", "precision", "recall", "f1"]
            x.add_rows(
                [
                    np.concatenate([['affiliation'], best_affiliation]),
                ]
            )

            log_metric_msg(f"Found best metric:\n{x}")

        # best_affiliation is 5 dimension array:
        # 0: (0-based): _percentile(the best percentile with best f1)
        # 1: _threshold(the threshold(float) corresponding to the best _percentile) ,
        # 2: precision(metric),
        # 3: recall(metric),
        # 4: f1(metric)
        return best_affiliation


class BaseModel(ModelInterface, metaclass=abc.ABCMeta):
    def __init__(self, config):
        """
        Parameters
        ----------
        config : BaseModelConfig
            A model config
        """
        self.config = config

        # The loss of the model, including training, validation, testing.
        self.train_loss = None
        self.valid_loss = None
        self.test_loss = None

    @abc.abstractmethod
    def report_metrics(self, valid_dl, test_dl, app_dt: DateTimeHelper):
        """
        Report performance metrics to NNI by `report_nni_final_metric(**time_elapse, **test_losses,
        model_save_home=model_save_home)`

        Parameters
        ----------
        app_dt : MetricsCollector
        valid_dl : a dataloader of pytorch if pytorch, np.ndarray otherwise.
        test_dl : a dataloader of pytorch if pytorch, np.ndarray otherwise.

        Returns
        -------

        """
        pass

    def get_config(self) -> BaseModelConfig:
        """
        Get   model configuration.

        Parameters
        ----------

        Returns
        -------

        """
        assert self.config is not None
        return self.config

    def ad_predict(self, target, scores):
        """
        Return best_affiliation, best_accumulator_score, predict_label

        Parameters
        ----------
        target :
        scores :

        Returns
        -------
        """
        if_aff = np.count_nonzero(target)
        if if_aff != 0:
            events_gt = convert_vector_to_events(target)
        target = TimeSeries.from_pd(pd.DataFrame(target))
        scores = np.array(scores)
        # standardization
        mean = np.mean(scores)
        std = np.std(scores)
        if std != 0:
            scores = (scores - mean) / std

        # Set the anomaly  percentile to [0.001, 0.3] (100 - np.max(nu_list)=99.7 )

        # To reduce the quantity to calculate
        # nu_list = np.arange(1, 301) / 1e3
        nu_list = np.linspace(1, 301, 50) / 1e3  # nu_list is in [0.001,0.30]
        f1_revised_point_adjusted_list, accumulate_tsad_score_list, f1_affiliation_list, affiliation_metric_list = [], [], [], []
        for detect_nu in nu_list:
            threshold = np.percentile(scores, 100 - detect_nu)
            predict = np.where(scores > threshold, 1, 0)
            dic = dict()
            if if_aff != 0:
                events_pred = convert_vector_to_events(predict)
                Trange = (0, len(predict))
                affiliation_metric = pr_from_events(events_pred, events_gt, Trange)
                dic["precision"] = affiliation_metric['precision']
                dic["recall"] = affiliation_metric['recall']
                affiliation_f1 = 2 * (dic["precision"] * dic["recall"]) / (dic["precision"] + dic["recall"])
                dic['f1'] = affiliation_f1
                f1_affiliation_list.append(affiliation_f1)
            else:

                dic["precision"] = 0
                dic["recall"] = 0
                dic['f1'] = 0
                f1_affiliation_list.append(0)
            affiliation_metric_list.append(dic)
            predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))

            # 这个对象可以求很多的score, such as PW, PA, RPA
            # metric = {
            #     "Pointwise Precision": ats.precision(score_type=ScoreType.Pointwise),
            #     "Pointwise Recall": ats.recall(score_type=ScoreType.Pointwise),
            #     "Pointwise F1": ats.f1(score_type=ScoreType.Pointwise),
            #     "PointAdjusted Precision": ats.precision(score_type=ScoreType.PointAdjusted),
            #     "PointAdjusted Recall": ats.recall(score_type=ScoreType.PointAdjusted),
            #     "PointAdjusted F1": ats.f1(score_type=ScoreType.PointAdjusted),
            #     "RevisedPointAdjusted Precision": ats.precision(score_type=ScoreType.RevisedPointAdjusted),
            #     "RevisedPointAdjusted Recall": ats.recall(score_type=ScoreType.RevisedPointAdjusted),
            #     "RevisedPointAdjusted F1": ats.f1(score_type=ScoreType.RevisedPointAdjusted),
            #
            # }
            ats = accumulate_tsad_score(ground_truth=target, predict=predict_ts)

            f1_revised_point_adjusted = ats.f1(ScoreType.RevisedPointAdjusted)
            f1_revised_point_adjusted_list.append(f1_revised_point_adjusted)
            accumulate_tsad_score_list.append(ats)

        # Find the maximum affiliation F1 score
        index_max_f1_affiliation = np.argmax(f1_affiliation_list, axis=0)
        best_affiliation = affiliation_metric_list[index_max_f1_affiliation]

        # Find the maximum affiliation percentile(such as  2.26%)
        nu_max1 = nu_list[index_max_f1_affiliation]

        # log(f"Best affiliation quantile: {nu_max1}\n"
        #     f"Best affiliation  metric: {best_affiliation}")

        # Find the maximum RevisedPointAdjusted f1
        index_max_f1_RPA = np.argmax(f1_revised_point_adjusted_list, axis=0)
        best_accumulator_score = accumulate_tsad_score_list[index_max_f1_RPA]
        # nu_max2 = nu_list[index_max_f1_RPA]
        # revised_point_adjust = {
        #     "f1": best_accumulator_score.f1(score_type=ScoreType.RevisedPointAdjusted),
        #     "precision": best_accumulator_score.precision(score_type=ScoreType.RevisedPointAdjusted),
        #     "recall": best_accumulator_score.recall(score_type=ScoreType.RevisedPointAdjusted)
        # }
        # UtilSys.is_debug_mode()  and log.info(f"Best revised point adjusted anomaly quantile: {nu_max2},\n"
        #          f"Best revised point adjusted: {revised_point_adjust}")

        threshold = np.percentile(scores, 100 - nu_max1)
        predict_label = np.where(scores > threshold, 1, 0)
        # best_affiliation={'precision': 0, 'recall': 0},
        # best_accumulator: <merlion.evaluate.anomaly.TSADScoreAccumulator>
        # predict_label.shape=59, a value either 0 or 1
        #   the best predict label where the affiliation f1 is maximal. Predict is
        #   not a score, it is a label which value is one of 0 or 1.
        #   predict = np.where(scores > threshold, 1, 0)

        # best_accumulator_score can obtain three different metrics. PW, PA, RPA
        # {
        #     "Pointwise Precision": best_accumulator_score.precision(score_type=ScoreType.Pointwise),
        #     "Pointwise Recall": best_accumulator_score.recall(score_type=ScoreType.Pointwise),
        #     "Pointwise F1": best_accumulator_score.f1(score_type=ScoreType.Pointwise),
        #     "PointAdjusted Precision": best_accumulator_score.precision(score_type=ScoreType.PointAdjusted),
        #     "PointAdjusted Recall": best_accumulator_score.recall(score_type=ScoreType.PointAdjusted),
        #     "PointAdjusted F1": best_accumulator_score.f1(score_type=ScoreType.PointAdjusted),
        #     "RevisedPointAdjusted Precision":
        #       best_accumulator_score.precision(score_type=ScoreType.RevisedPointAdjusted),
        #     "RevisedPointAdjusted Recall": best_accumulator_score.recall(score_type=ScoreType.RevisedPointAdjusted),
        #     "RevisedPointAdjusted F1": best_accumulator_score.f1(score_type=ScoreType.RevisedPointAdjusted),
        #
        # }

        # best_affiliation contains for metrics, the affiliation precision, affiliation recall,
        # affiliation F1 by:
        # best_affiliation["precision"]
        # best_affiliation["recall"]
        # best_affiliation['f1']
        return best_affiliation, best_accumulator_score, predict_label

    def round(self, number: torch.Tensor, decimals=4):
        """
        Round  a tensor to numpy. seeing
        https://pytorch.org/docs/stable/generated/torch.round.html?highlight=round#torch.round

        Parameters
        ----------
        number :

        Returns
        -------

        """
        assert isinstance(number, torch.Tensor), "Round number must be a torch.Tensor"
        return np.round(number.item(), decimals)


class PyTorchBase(ModelInterface, metaclass=abc.ABCMeta):
    def __init__(self, config):
        """
        Parameters
        ----------
        config :
            A model config
        """
        self.config = config

        # The loss of the model, including training, validation, testing.
        self.train_loss = None
        self.valid_loss = None
        self.test_loss = None

    @abc.abstractmethod
    def report_metrics(self, valid_dl, test_dl, app_dt: DateTimeHelper):
        """
        Report performance metrics to NNI by `report_nni_final_metric(**time_elapse, **test_losses,
        model_save_home=model_save_home)`

        Parameters
        ----------
        app_dt : MetricsCollector
        valid_dl : a dataloader of pytorch if pytorch, np.ndarray otherwise.
        test_dl : a dataloader of pytorch if pytorch, np.ndarray otherwise.

        Returns
        -------

        """
        pass

    def get_config(self) -> BaseModelConfig:
        """
        Get   model configuration.

        Parameters
        ----------

        Returns
        -------

        """
        assert self.config is not None
        return self.config

    def ad_predict(self, target, scores):
        """
        Return best_affiliation, best_accumulator_score, predict_label

        Parameters
        ----------
        target :
        scores :

        Returns
        -------
        """
        if_aff = np.count_nonzero(target)
        if if_aff != 0:
            events_gt = convert_vector_to_events(target)
        target = TimeSeries.from_pd(pd.DataFrame(target))
        scores = np.array(scores)
        # standardization
        mean = np.mean(scores)
        std = np.std(scores)
        if std != 0:
            scores = (scores - mean) / std

        # Set the anomaly  percentile to [0.001, 0.3] (100 - np.max(nu_list)=99.7 )

        # To reduce the quantity to calculate
        # nu_list = np.arange(1, 301) / 1e3
        nu_list = np.linspace(1, 301, 50) / 1e3  # nu_list is in [0.001,0.30]
        f1_revised_point_adjusted_list, accumulate_tsad_score_list, f1_affiliation_list, affiliation_metric_list = [], [], [], []
        for detect_nu in nu_list:
            threshold = np.percentile(scores, 100 - detect_nu)
            predict = np.where(scores > threshold, 1, 0)
            dic = dict()
            if if_aff != 0:
                events_pred = convert_vector_to_events(predict)
                Trange = (0, len(predict))
                affiliation_metric = pr_from_events(events_pred, events_gt, Trange)
                dic["precision"] = affiliation_metric['precision']
                dic["recall"] = affiliation_metric['recall']
                affiliation_f1 = 2 * (dic["precision"] * dic["recall"]) / (dic["precision"] + dic["recall"])
                dic['f1'] = affiliation_f1
                f1_affiliation_list.append(affiliation_f1)
            else:

                dic["precision"] = 0
                dic["recall"] = 0
                dic['f1'] = 0
                f1_affiliation_list.append(0)
            affiliation_metric_list.append(dic)
            predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))

            # 这个对象可以求很多的score, such as PW, PA, RPA
            # metric = {
            #     "Pointwise Precision": ats.precision(score_type=ScoreType.Pointwise),
            #     "Pointwise Recall": ats.recall(score_type=ScoreType.Pointwise),
            #     "Pointwise F1": ats.f1(score_type=ScoreType.Pointwise),
            #     "PointAdjusted Precision": ats.precision(score_type=ScoreType.PointAdjusted),
            #     "PointAdjusted Recall": ats.recall(score_type=ScoreType.PointAdjusted),
            #     "PointAdjusted F1": ats.f1(score_type=ScoreType.PointAdjusted),
            #     "RevisedPointAdjusted Precision": ats.precision(score_type=ScoreType.RevisedPointAdjusted),
            #     "RevisedPointAdjusted Recall": ats.recall(score_type=ScoreType.RevisedPointAdjusted),
            #     "RevisedPointAdjusted F1": ats.f1(score_type=ScoreType.RevisedPointAdjusted),
            #
            # }
            ats = accumulate_tsad_score(ground_truth=target, predict=predict_ts)

            f1_revised_point_adjusted = ats.f1(ScoreType.RevisedPointAdjusted)
            f1_revised_point_adjusted_list.append(f1_revised_point_adjusted)
            accumulate_tsad_score_list.append(ats)

        # Find the maximum affiliation F1 score
        index_max_f1_affiliation = np.argmax(f1_affiliation_list, axis=0)
        best_affiliation = affiliation_metric_list[index_max_f1_affiliation]

        # Find the maximum affiliation percentile(such as  2.26%)
        nu_max1 = nu_list[index_max_f1_affiliation]

        # log(f"Best affiliation quantile: {nu_max1}\n"
        #     f"Best affiliation  metric: {best_affiliation}")

        # Find the maximum RevisedPointAdjusted f1
        index_max_f1_RPA = np.argmax(f1_revised_point_adjusted_list, axis=0)
        best_accumulator_score = accumulate_tsad_score_list[index_max_f1_RPA]
        # nu_max2 = nu_list[index_max_f1_RPA]
        # revised_point_adjust = {
        #     "f1": best_accumulator_score.f1(score_type=ScoreType.RevisedPointAdjusted),
        #     "precision": best_accumulator_score.precision(score_type=ScoreType.RevisedPointAdjusted),
        #     "recall": best_accumulator_score.recall(score_type=ScoreType.RevisedPointAdjusted)
        # }
        # UtilSys.is_debug_mode()  and log.info(f"Best revised point adjusted anomaly quantile: {nu_max2},\n"
        #          f"Best revised point adjusted: {revised_point_adjust}")

        threshold = np.percentile(scores, 100 - nu_max1)
        predict_label = np.where(scores > threshold, 1, 0)
        # best_affiliation={'precision': 0, 'recall': 0},
        # best_accumulator: <merlion.evaluate.anomaly.TSADScoreAccumulator>
        # predict_label.shape=59, a value either 0 or 1
        #   the best predict label where the affiliation f1 is maximal. Predict is
        #   not a score, it is a label which value is one of 0 or 1.
        #   predict = np.where(scores > threshold, 1, 0)

        # best_accumulator_score can obtain three different metrics. PW, PA, RPA
        # {
        #     "Pointwise Precision": best_accumulator_score.precision(score_type=ScoreType.Pointwise),
        #     "Pointwise Recall": best_accumulator_score.recall(score_type=ScoreType.Pointwise),
        #     "Pointwise F1": best_accumulator_score.f1(score_type=ScoreType.Pointwise),
        #     "PointAdjusted Precision": best_accumulator_score.precision(score_type=ScoreType.PointAdjusted),
        #     "PointAdjusted Recall": best_accumulator_score.recall(score_type=ScoreType.PointAdjusted),
        #     "PointAdjusted F1": best_accumulator_score.f1(score_type=ScoreType.PointAdjusted),
        #     "RevisedPointAdjusted Precision":
        #       best_accumulator_score.precision(score_type=ScoreType.RevisedPointAdjusted),
        #     "RevisedPointAdjusted Recall": best_accumulator_score.recall(score_type=ScoreType.RevisedPointAdjusted),
        #     "RevisedPointAdjusted F1": best_accumulator_score.f1(score_type=ScoreType.RevisedPointAdjusted),
        #
        # }

        # best_affiliation contains for metrics, the affiliation precision, affiliation recall,
        # affiliation F1 by:
        # best_affiliation["precision"]
        # best_affiliation["recall"]
        # best_affiliation['f1']
        return best_affiliation, best_accumulator_score, predict_label

    def round(self, number: torch.Tensor, decimals=4):
        """
        Round  a tensor to numpy. seeing
        https://pytorch.org/docs/stable/generated/torch.round.html?highlight=round#torch.round

        Parameters
        ----------
        number :

        Returns
        -------

        """
        # return self.round_numpy(number,decimals)
        return self.round_torch(number, decimals)

    def round_numpy(self, number, decimals):
        assert isinstance(number, torch.Tensor), "Round number must be a torch.Tensor"
        return np.round(number.item(), decimals)

    def round_torch(self, number, decimals):
        return torch.round(number.item(), decimals)


class SKLearnBase(ModelInterface, metaclass=abc.ABCMeta):

    def __init__(self, config):
        """
        Parameters
        ----------
        config :
            A model config
        """
        self.config = config

        # The loss of the model, including training, validation, testing.
        self.train_loss = None
        self.valid_loss = None
        self.test_loss = None

    @abc.abstractmethod
    def _gather_metrics(self, valid_x, valid_y, test_x, test_y, app_dt: DateTimeHelper):
        pass

    def report_metrics(self, valid_x, valid_y, test_x, test_y, app_dt: DateTimeHelper = DateTimeHelper()):
        """
        Report performance metrics to NNI by `report_nni_final_metric(**time_elapse, **test_losses,
        model_save_home=model_save_home)`

        Parameters
        ----------
        test_y :
        test_x :
        valid_y :
        valid_x :
        app_dt : MetricsCollector

        Returns
        -------

        """
        # Fix: assertion Error('nni.get_next_parameter() needs to be called before report_final_result')
        metrics = self._gather_metrics(valid_x, valid_y, test_x, test_y, app_dt)
        assert metrics is not None, f"You must override the _gather_metrics when you  call report_metrics. " \
                                    f"received {metrics}"
        metrics.update(app_dt.collect_metrics())
        metrics['default'] = metrics.get("test_affiliation_f1")
        self.report_nni_final_metric(**metrics)

    def get_config(self):
        """
        Get  model configuration.

        Parameters
        ----------

        Returns
        -------

        """
        assert self.config is not None
        return self.config

    def ad_predict_v2(self, target, scores, log_best=False):
        """
        Return  best_affiliation, best_pw, best_pa, best_rpa

        Parameters
        ----------
        log_best : bool
            Whether to log the best metrics
        target : np.ndarray
        scores : np.ndarray

        Returns
        -------
        """

        # Set the anomaly  percentile to [0.001, 0.3] (100 - np.max(nu_list)=99.7 )
        # To reduce the quantity to calculate
        # nu_list = np.arange(1, 301) / 1e3
        _aff_all, _all_pw, _all_pa, _all_rpa = [], [], [], []
        _percentiles = np.arange(0.25, 15.5, 0.25)  # nu_list is in [0.001,0.30]
        for _percentile in _percentiles:
            _threshold = np.percentile(scores, 100 - _percentile)
            predict = np.where(scores > _threshold, 1, 0)

            #
            # Affiliation metrics
            aff_precision, aff_recall, aff_f1 = affiliation_metrics(target, predict)
            _aff_all.append([_percentile, _threshold, aff_precision, aff_recall, aff_f1])

            predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
            ground_truth_ts = TimeSeries.from_pd(pd.DataFrame(target))

            ats: TSADScoreAccumulator = accumulate_tsad_score(ground_truth=ground_truth_ts, predict=predict_ts)
            _all_pw.append(
                [_percentile, _threshold,
                 ats.precision(ScoreType.Pointwise),
                 ats.recall(ScoreType.Pointwise),
                 ats.f1(ScoreType.Pointwise)])
            _all_pa.append(
                [_percentile, _threshold,
                 ats.precision(ScoreType.PointAdjusted),
                 ats.recall(ScoreType.PointAdjusted),
                 ats.f1(ScoreType.PointAdjusted)])
            _all_rpa.append(
                [_percentile, _threshold,
                 ats.precision(ScoreType.RevisedPointAdjusted),
                 ats.recall(ScoreType.RevisedPointAdjusted),
                 ats.f1(ScoreType.RevisedPointAdjusted)])

        # Find the maximum affiliation F1 score
        best_affiliation = self._find_max_metric(_aff_all)
        best_pw = self._find_max_metric(_all_pw)
        best_pa = self._find_max_metric(_all_pa)
        best_rpa = self._find_max_metric(_all_rpa)

        if log_best:
            x = PrettyTable()
            x.field_names = ["type", "percentile", "threshold", "precision", "recall", "f1"]
            x.add_rows(
                [
                    np.concatenate([['affiliation'], best_affiliation]),
                    np.concatenate([['Pointwise'], best_pw]),
                    np.concatenate([['PointAdjusted'], best_pa]),
                    np.concatenate([['RevisedPointAdjusted'], best_rpa]),
                ]
            )

            log_metric_msg(f"Found best metric:\n{x}")

        # best_affiliation, best_pw, best_pa, best_rpw is 5 dimension array:
        # 0: (0-based): _percentile(the best percentile with best f1)
        # 1: _threshold(the value according to the best _percentile) ,
        # 2: precision(metric),
        # 3: recall(metric),
        # 4: f1(metric)
        return best_affiliation, best_pw, best_pa, best_rpa

    def ad_predict(self, target, scores):
        """
        Return best_affiliation, best_accumulator_score, predict_label

        Parameters
        ----------
        target : np.ndarray
        scores : np.ndarray

        Returns
        -------
        """
        if_aff = np.count_nonzero(target)
        if if_aff != 0:
            events_gt = convert_vector_to_events(target)
        target = TimeSeries.from_pd(pd.DataFrame(target))
        scores = np.array(scores)
        # standardization
        mean = np.mean(scores)
        std = np.std(scores)
        if std != 0:
            scores = (scores - mean) / std

        # Set the anomaly  percentile to [0.001, 0.3] (100 - np.max(nu_list)=99.7 )

        # To reduce the quantity to calculate
        # nu_list = np.arange(1, 301) / 1e3
        nu_list = np.linspace(0, 20, 200)  # nu_list is in [0.001,0.30]
        f1_revised_point_adjusted_list, accumulate_tsad_score_list, f1_affiliation_list, affiliation_metric_list = [], [], [], []
        for detect_nu in nu_list:
            threshold = np.percentile(scores, 100 - detect_nu)
            predict = np.where(scores > threshold, 1, 0)
            dic = dict()
            if if_aff != 0:
                events_pred = convert_vector_to_events(predict)
                Trange = (0, len(predict))
                affiliation_metric = pr_from_events(events_pred, events_gt, Trange)
                dic["precision"] = affiliation_metric['precision']
                dic["recall"] = affiliation_metric['recall']
                affiliation_f1 = 2 * (dic["precision"] * dic["recall"]) / (dic["precision"] + dic["recall"])
                dic['f1'] = affiliation_f1
                f1_affiliation_list.append(affiliation_f1)
            else:

                dic["precision"] = 0
                dic["recall"] = 0
                dic['f1'] = 0
                f1_affiliation_list.append(0)
            affiliation_metric_list.append(dic)
            predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))

            # 这个对象可以求很多的score, such as PW, PA, RPA
            # metric = {
            #     "Pointwise Precision": ats.precision(score_type=ScoreType.Pointwise),
            #     "Pointwise Recall": ats.recall(score_type=ScoreType.Pointwise),
            #     "Pointwise F1": ats.f1(score_type=ScoreType.Pointwise),
            #     "PointAdjusted Precision": ats.precision(score_type=ScoreType.PointAdjusted),
            #     "PointAdjusted Recall": ats.recall(score_type=ScoreType.PointAdjusted),
            #     "PointAdjusted F1": ats.f1(score_type=ScoreType.PointAdjusted),
            #     "RevisedPointAdjusted Precision": ats.precision(score_type=ScoreType.RevisedPointAdjusted),
            #     "RevisedPointAdjusted Recall": ats.recall(score_type=ScoreType.RevisedPointAdjusted),
            #     "RevisedPointAdjusted F1": ats.f1(score_type=ScoreType.RevisedPointAdjusted),
            #
            # }
            ats = accumulate_tsad_score(ground_truth=target, predict=predict_ts)

            f1_revised_point_adjusted = ats.f1(ScoreType.RevisedPointAdjusted)
            f1_revised_point_adjusted_list.append(f1_revised_point_adjusted)
            accumulate_tsad_score_list.append(ats)

        # Find the maximum affiliation F1 score
        index_max_f1_affiliation = np.argmax(f1_affiliation_list, axis=0)
        best_affiliation = affiliation_metric_list[index_max_f1_affiliation]

        # Find the maximum affiliation percentile(such as  2.26%)
        nu_max1 = nu_list[index_max_f1_affiliation]

        # log(f"Best affiliation quantile: {nu_max1}\n"
        #     f"Best affiliation  metric: {best_affiliation}")

        # Find the maximum RevisedPointAdjusted f1
        index_max_f1_RPA = np.argmax(f1_revised_point_adjusted_list, axis=0)
        best_accumulator_score = accumulate_tsad_score_list[index_max_f1_RPA]

        threshold = np.percentile(scores, 100 - nu_max1)
        predict_label = np.where(scores > threshold, 1, 0)
        return best_affiliation, best_accumulator_score, predict_label

    def round(self, number: torch.Tensor, decimals=4):
        """
        Round  a tensor to numpy. seeing
        https://pytorch.org/docs/stable/generated/torch.round.html?highlight=round#torch.round

        Parameters
        ----------
        number :

        Returns
        -------

        """
        assert isinstance(number, torch.Tensor), "Round number must be a torch.Tensor"
        return np.round(number.item(), decimals)

    @staticmethod
    def report_nni_final_metric(**kwargs):
        """
        Report  metric to nni.

        Examples
        --------
        .. code-block::

            res = report_nni_final_metric(auc1=3, auc2=4)
            assert res == {'auc1': 3, 'auc2': 4}

            res = report_nni_final_metric(auc1=3, auc2=4, **{'key1': 3})
            assert res == {'auc1': 3, 'auc2': 4, 'key1': 3}

        """

        # The result
        if kwargs.get("default") is None:
            raise ValueError(
                "Report metric must contain a key named default according to the NNI. More details see "
                "https://nni.readthedocs.io/en/stable/reference/hpo.html#nni.report_final_result")
        # log.info(f"Metric report: \n{pprint.pformat(kwargs)}")
        nni.report_final_result(kwargs)
        return kwargs


class KerasBase(ModelInterface, metaclass=abc.ABCMeta):

    def __init__(self, config):
        """
        Parameters
        ----------
        config : BaseModelConfig
            A model config
        """
        self.config = config

        # The loss of the model, including training, validation, testing.
        self.train_loss = None
        self.valid_loss = None
        self.test_loss = None

    @abc.abstractmethod
    def _gather_metrics(self, valid_x, valid_y, test_x, test_y, app_dt: DateTimeHelper):
        pass

    def report_metrics(self, valid_x, valid_y, test_x, test_y, app_dt: DateTimeHelper):
        """
        Report performance metrics to NNI by `report_nni_final_metric(**time_elapse, **test_losses,
        model_save_home=model_save_home)`

        Parameters
        ----------
        test_y :
        test_x :
        valid_y :
        valid_x :
        app_dt : MetricsCollector

        Returns
        -------

        """
        metrics = self._gather_metrics(valid_x, valid_y, test_x, test_y, app_dt)
        metrics.update(app_dt.collect_metrics())
        metrics['default'] = metrics.get("test_affiliation_f1")
        report_nni_final_metric(**metrics)

    def get_config(self) -> BaseModelConfig:
        """
        Get  model configuration.

        Parameters
        ----------

        Returns
        -------

        """
        assert self.config is not None
        return self.config

    def ad_predict(self, target, scores):
        """
        Return best_affiliation, best_accumulator_score, predict_label

        Parameters
        ----------
        target : np.ndarray
        scores : np.ndarray

        Returns
        -------
        """
        if_aff = np.count_nonzero(target)
        if if_aff != 0:
            events_gt = convert_vector_to_events(target)
        target = TimeSeries.from_pd(pd.DataFrame(target))
        scores = np.array(scores)
        # standardization
        mean = np.mean(scores)
        std = np.std(scores)
        if std != 0:
            scores = (scores - mean) / std

        # Set the anomaly  percentile to [0.001, 0.3] (100 - np.max(nu_list)=99.7 )

        # To reduce the quantity to calculate
        # nu_list = np.arange(1, 301) / 1e3
        nu_list = np.linspace(1, 201, 50) / 1e3  # nu_list is in [0.001,0.30]
        f1_revised_point_adjusted_list, accumulate_tsad_score_list, f1_affiliation_list, affiliation_metric_list = [], [], [], []
        for detect_nu in nu_list:
            threshold = np.percentile(scores, 100 - detect_nu)
            predict = np.where(scores > threshold, 1, 0)
            dic = dict()
            if if_aff != 0:
                events_pred = convert_vector_to_events(predict)
                Trange = (0, len(predict))
                affiliation_metric = pr_from_events(events_pred, events_gt, Trange)
                dic["precision"] = affiliation_metric['precision']
                dic["recall"] = affiliation_metric['recall']
                affiliation_f1 = 2 * (dic["precision"] * dic["recall"]) / (dic["precision"] + dic["recall"])
                dic['f1'] = affiliation_f1
                f1_affiliation_list.append(affiliation_f1)
            else:

                dic["precision"] = 0
                dic["recall"] = 0
                dic['f1'] = 0
                f1_affiliation_list.append(0)
            affiliation_metric_list.append(dic)
            predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))

            # 这个对象可以求很多的score, such as PW, PA, RPA
            # metric = {
            #     "Pointwise Precision": ats.precision(score_type=ScoreType.Pointwise),
            #     "Pointwise Recall": ats.recall(score_type=ScoreType.Pointwise),
            #     "Pointwise F1": ats.f1(score_type=ScoreType.Pointwise),
            #     "PointAdjusted Precision": ats.precision(score_type=ScoreType.PointAdjusted),
            #     "PointAdjusted Recall": ats.recall(score_type=ScoreType.PointAdjusted),
            #     "PointAdjusted F1": ats.f1(score_type=ScoreType.PointAdjusted),
            #     "RevisedPointAdjusted Precision": ats.precision(score_type=ScoreType.RevisedPointAdjusted),
            #     "RevisedPointAdjusted Recall": ats.recall(score_type=ScoreType.RevisedPointAdjusted),
            #     "RevisedPointAdjusted F1": ats.f1(score_type=ScoreType.RevisedPointAdjusted),
            #
            # }
            ats = accumulate_tsad_score(ground_truth=target, predict=predict_ts)

            f1_revised_point_adjusted = ats.f1(ScoreType.RevisedPointAdjusted)
            f1_revised_point_adjusted_list.append(f1_revised_point_adjusted)
            accumulate_tsad_score_list.append(ats)

        # Find the maximum affiliation F1 score
        index_max_f1_affiliation = np.argmax(f1_affiliation_list, axis=0)
        best_affiliation = affiliation_metric_list[index_max_f1_affiliation]

        # Find the maximum affiliation percentile(such as  2.26%)
        nu_max1 = nu_list[index_max_f1_affiliation]

        # log(f"Best affiliation quantile: {nu_max1}\n"
        #     f"Best affiliation  metric: {best_affiliation}")

        # Find the maximum RevisedPointAdjusted f1
        index_max_f1_RPA = np.argmax(f1_revised_point_adjusted_list, axis=0)
        best_accumulator_score = accumulate_tsad_score_list[index_max_f1_RPA]

        threshold = np.percentile(scores, 100 - nu_max1)
        predict_label = np.where(scores > threshold, 1, 0)
        return best_affiliation, best_accumulator_score, predict_label

    def round(self, number: torch.Tensor, decimals=4):
        """
        Round  a tensor to numpy. seeing
        https://pytorch.org/docs/stable/generated/torch.round.html?highlight=round#torch.round

        Parameters
        ----------
        number :

        Returns
        -------

        """
        assert isinstance(number, torch.Tensor), "Round number must be a torch.Tensor"
        return np.round(number.item(), decimals)
