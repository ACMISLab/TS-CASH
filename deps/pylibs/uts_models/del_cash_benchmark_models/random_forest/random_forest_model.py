import numpy as np

from sklearn.ensemble import RandomForestClassifier
from typeguard import typechecked

from pylibs.uts_models.benchmark_models.random_forest.random_forest_conf import RandomForestConf
from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.utils.util_log import get_logger

log = get_logger()

_PRECISION_INDEX = -3
_RECALL_INDEX = -2
_F1_INDEX = -1


class RandomForestModel:
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


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)

    rf = RandomForestConf()
    clf = RandomForestModel(rf)
    clf.fit(X, y)
    print(clf.predict([[0, 0, 0, 0]]))
