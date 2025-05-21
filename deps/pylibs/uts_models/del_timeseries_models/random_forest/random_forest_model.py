import numpy as np

from sklearn.ensemble import RandomForestClassifier
from typeguard import typechecked

from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.utils.util_log import get_logger
from timeseries_models.base_model import SKLearnBase
from timeseries_models.random_forest.random_forest_conf import RandomForestConf

log = get_logger()

_PRECISION_INDEX = -3
_RECALL_INDEX = -2
_F1_INDEX = -1


class RandomForestModel(SKLearnBase):
    """
    The OC-SVM-based time series anomaly detector.
    """

    def score(self, data):
        """
        In the isolation forest: The lower score, the more abnormal.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------

        """
        return -self.model.predict_proba(data)[:, 0]

    @typechecked
    def __init__(self, config: RandomForestConf, device: str = "cpu"):
        super(RandomForestModel, self).__init__(config)
        self.center = None
        self.length = None
        self.device = device
        assert self.device is not None
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            criterion=config.criterion,
            max_features=config.max_features,
            max_samples=config.max_samples,
            verbose=config.verbose,

        )

    def predict(self, data):
        return self.model.predict(data)

    @typechecked
    def fit(self, x: np.ndarray, y: np.ndarray):
        return self.model.fit(x, y)

    @DeprecationWarning
    def _gather_metrics_all(self, valid_x, valid_y, test_x, test_y, app_dt: DateTimeHelper):
        """
        A function that gather all performance, including  affiliation, pw, pa,  rpa
        Parameters
        ----------
        valid_x :
        valid_y :
        test_x :
        test_y :
        app_dt :

        Returns
        -------

        """
        UtilSys.is_debug_mode() and log.info("Calculating model metrics...")
        valid_score = self.score(valid_x)
        test_score = self.score(test_x)
        UtilSys.is_debug_mode() and log.info("Calculating model metrics on validation data set...")
        val_best_affiliation, val_best_pw, val_best_pa, val_best_rpa = \
            self.ad_predict_v2(valid_y, valid_score, log_best=True)

        UtilSys.is_debug_mode() and log.info("Calculating model metrics on testing data set...")
        test_best_affiliation, test_best_pw, test_best_pa, test_best_rpa = \
            self.ad_predict_v2(test_y, test_score, log_best=True)

        return {
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'test_loss': self.test_loss,
            'valid_affiliation_f1': val_best_affiliation[_F1_INDEX],
            'valid_affiliation_precision': val_best_affiliation[_PRECISION_INDEX],
            'valid_affiliation_recall': val_best_affiliation[_RECALL_INDEX],
            "valid_point_wise_f1": val_best_pw[_F1_INDEX],
            "valid_point_wise_precision": val_best_pw[_PRECISION_INDEX],
            "valid_point_wise_recall": val_best_pw[_RECALL_INDEX],
            "valid_point_adjusted_f1": val_best_pa[_F1_INDEX],
            "valid_point_adjusted_precision": val_best_pa[_PRECISION_INDEX],
            "valid_point_adjusted_recall": val_best_pa[_RECALL_INDEX],
            "valid_revised_point_adjusted_precision": val_best_rpa[_PRECISION_INDEX],
            "valid_revised_point_adjusted_recall": val_best_rpa[_RECALL_INDEX],
            "valid_revised_point_adjusted_f1": val_best_rpa[_F1_INDEX],
            'test_affiliation_f1': test_best_affiliation[_F1_INDEX],
            'test_affiliation_precision': test_best_affiliation[_PRECISION_INDEX],
            'test_affiliation_recall': test_best_affiliation[_RECALL_INDEX],
            "test_point_wise_precision": test_best_pw[_PRECISION_INDEX],
            "test_point_wise_recall": test_best_pw[_RECALL_INDEX],
            "test_point_wise_f1": test_best_pw[_F1_INDEX],
            "test_point_adjusted_precision": test_best_pa[_PRECISION_INDEX],
            "test_point_adjusted_recall": test_best_pa[_RECALL_INDEX],
            "test_point_adjusted_f1": test_best_pa[_F1_INDEX],
            "test_revised_point_adjusted_precision": test_best_rpa[_PRECISION_INDEX],
            "test_revised_point_adjusted_recall": test_best_rpa[_RECALL_INDEX],
            "test_revised_point_adjusted_f1": test_best_rpa[_F1_INDEX],

        }

    def _gather_metrics(self, valid_x, valid_y, test_x, test_y, app_dt: DateTimeHelper):
        return self._gather_metrics_affiliation(valid_x, valid_y, test_x, test_y, app_dt)
        # return self._gather_metrics_all(valid_x, valid_y, test_x, test_y, app_dt)

    def _gather_metrics_affiliation(self, valid_x, valid_y, test_x, test_y, app_dt: DateTimeHelper):
        """

        Parameters
        ----------
        valid_x :
        valid_y :
        test_x :
        test_y :
        app_dt :

        Returns
        -------

        """
        app_dt.evaluate_start()
        UtilSys.is_debug_mode() and log.info("Calculating model metrics...")
        valid_score = self.score(valid_x)
        test_score = self.score(test_x)
        UtilSys.is_debug_mode() and log.info("Calculating model metrics on validation data set...")
        val_best_affiliation = self.ad_predict_v3(valid_y, valid_score, log_best=True)

        UtilSys.is_debug_mode() and log.info("Calculating model metrics on testing data set...")
        test_best_affiliation = self.ad_predict_v3(test_y, test_score, log_best=True)
        app_dt.evaluate_end()
        return {
            'valid_affiliation_f1': val_best_affiliation[_F1_INDEX],
            'valid_affiliation_precision': val_best_affiliation[_PRECISION_INDEX],
            'valid_affiliation_recall': val_best_affiliation[_RECALL_INDEX],
            'test_affiliation_f1': test_best_affiliation[_F1_INDEX],
            'test_affiliation_precision': test_best_affiliation[_PRECISION_INDEX],
            'test_affiliation_recall': test_best_affiliation[_RECALL_INDEX],
        }


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)

    rf = RandomForestConf()
    clf = RandomForestModel(rf)
    clf.fit(X, y)
    print(clf.predict([[0, 0, 0, 0]]))
