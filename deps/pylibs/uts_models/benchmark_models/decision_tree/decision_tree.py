from __future__ import division
from __future__ import print_function

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array

from ..utils.detectorB import DetectorB
from ..utils.utility import invert_order
from ..utils.utility import _get_sklearn_version


class DecisionTree(DetectorB):

    def __init__(self):
        self.is_deep = False
        pass

    # noinspection PyIncorrectDocstring
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """

        # validate inputs X and y (optional)

        try:
            X = check_array(X)
        except:
            X = X.reshape(-1, 1)

        X = check_array(X)

        self.detector_ = DecisionTreeClassifier()
        self.detector_.fit(X=X, y=y)

        # Invert decision_scores_. Outliers comes with higher outlier scores
        # self.decision_scores_ = -self.detector_.negative_outlier_factor_
        # self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        # Invert outlier scores. Outliers comes with higher outlier scores
        # noinspection PyProtectedMember
        if _get_sklearn_version() > 19:
            return invert_order(self.detector_._score_samples(X))
        else:
            return invert_order(self.detector_._decision_function(X))

    def score(self, x):
        return -self.detector_.predict_proba(x)[:, 0]

    @property
    def n_neighbors_(self):
        """The actual number of neighbors used for kneighbors queries.
        Decorator for scikit-learn LOF attributes.
        """
        return self.detector_.n_neighbors_
