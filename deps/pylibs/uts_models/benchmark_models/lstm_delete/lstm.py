from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import numpy as np
import math
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler
from pylibs.uts_models.benchmark_models.lstm.distance import Fourier

"""
When training, one need to remove the anomaly data according to the original paper:
a network is trained on non-anomalous data and used as a predictor over a number of time steps.


@inproceedings{malhotraLongShortTerm2015a,
  title = {Long {{Short Term Memory Networks}} for {{Anomaly Detection}} in {{Time Series}}.},
  booktitle = {Esann},
  author = {Malhotra, Pankaj and Vig, Lovekesh and Shroff, Gautam and Agarwal, Puneet},
  date = {2015},
  volume = {2015},
  pages = {89}
}

"""

from pylibs.utils.util_joblib import cache_


def _create_dataset(X, slidingwindow, predict_time_steps):
    Xs, ys = [], []
    for i in range(len(X) - slidingwindow - predict_time_steps + 1):
        tmp = X[i: i + slidingwindow + predict_time_steps]
        # tmp = MinMaxScaler(feature_range=(0, 1)).fit_transform(tmp.reshape(-1, 1)).ravel()
        tmp = tmp.reshape(-1, 1)
        x = tmp[:slidingwindow]
        y = tmp[slidingwindow:]

        Xs.append(x)
        ys.append(y)
    return np.array(Xs), np.array(ys)


class LSTMModel:
    def __init__(self, slidingwindow=100,
                 predict_time_steps=1,
                 contamination=0.1,
                 epochs=10,
                 patience=10,
                 batch_size=64, verbose=0):
        self.is_deep = True
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.contamination = contamination
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.model_name = 'LSTM2'
        self.batch_size = batch_size
        model = Sequential()
        model.add(LSTM(50,
                       return_sequences=True,
                       input_shape=(slidingwindow, 1)))
        model.add(LSTM(50))
        model.add(Dense(self.predict_time_steps))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self._model = model
    def score(self, X):
        X_train, Y_train = self.create_dataset(X, self.slidingwindow, self.predict_time_steps)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        measure = Fourier()
        measure.set_param()
        measure.detector = self._model

        n_test_ = len(X)
        score = np.zeros(n_test_)
        estimation = self._model.predict(X_train, verbose=self.verbose)
        for i in range(estimation.shape[0]):
            score[i - estimation.shape[0]] = measure.measure(Y_train[i], estimation[i],
                                                             n_test_ - estimation.shape[0] + i)

        score[0: - estimation.shape[0]] = score[- estimation.shape[0]]

        return score

    def fit(self, X_clean, y=None):
        """

        Parameters
        ----------
        X_clean :
        y :  Not used, which is for API consistency by convention.

        Returns
        -------

        """

        X_train, Y_train = self.create_dataset(X_clean, self.slidingwindow, self.predict_time_steps)
        if X_train.shape[0] == 0:
            return
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self._model.fit(X_train, Y_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        verbose=self.verbose)
        # model.fit(X_train, Y_train,
        #           epochs=self.epochs, batch_size=64, verbose=self.verbose, callbacks=[es])

        # prediction = model.predict(X_test)
        #
        # self.Y = Y_test
        # self.estimation = prediction
        # self.estimator = model
        # self.n_initial = X_train.shape[0]

        return self

    def create_dataset(self, X, slidingwindow, predict_time_steps=1):
        return _create_dataset(X, slidingwindow, predict_time_steps)
        # Xs, ys = [], []
        # for i in range(len(X) - slidingwindow - predict_time_steps + 1):
        #     tmp = X[i: i + slidingwindow + predict_time_steps]
        #     # tmp = MinMaxScaler(feature_range=(0, 1)).fit_transform(tmp.reshape(-1, 1)).ravel()
        #     tmp = tmp.reshape(-1, 1)
        #     x = tmp[:slidingwindow]
        #     y = tmp[slidingwindow:]
        #
        #     Xs.append(x)
        #     ys.append(y)
        # return np.array(Xs), np.array(ys)

    def decision_function(self, X=False, measure=None):
        """Derive the decision score based on the given distance measure
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        measure : object
            object for given distance measure with methods to derive the score
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if type(X) != bool:
            self.X_train_ = X
        n_test_ = self.n_test_
        Y_test = self.Y

        score = np.zeros(n_test_)
        estimation = self.estimation

        for i in range(estimation.shape[0]):
            score[i - estimation.shape[0]] = measure.measure(Y_test[i], estimation[i],
                                                             n_test_ - estimation.shape[0] + i)

        score[0: - estimation.shape[0]] = score[- estimation.shape[0]]

        self.decision_scores_ = score
        return self

    def predict_proba(self, X, method='linear', measure=None):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:
        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The model must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        method : str, optional (default='linear')
            probability conversion method. It must be one of
            'linear' or 'unify'.
        Returns
        -------
        outlier_probability : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. Return the outlier probability, ranging
            in [0,1].
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        self.fit(X)
        self.decision_function(measure=measure)
        test_scores = self.decision_scores_

        probs = np.zeros([X.shape[0], int(self._classes)])
        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]
            return probs

        elif method == 'unify':
            # turn output into probability
            pre_erf_score = (test_scores - self._mu) / (
                    self._sigma * np.sqrt(2))
            erf_score = math.erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')

    def save_model(self, filepath):
        if not filepath.endswith(".h5"):
            filepath += ".h5"
        self._model.save_weights(filepath)
        return filepath

    def load_model(self, filepath):
        self._model.built = True
        self._model.load_weights(filepath)
        return self
