from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pylibs.uts_models.benchmark_models.model_utils import create_dataset_for_cnn_and_lstm
from pylibs.distances.euclidean_distance import EuclideanDistance


class cnn:
    def __init__(self, slidingwindow=100, predict_time_steps=1, contamination=0.1, epochs=10, patience=10, verbose=0,
                 batch_size=128):
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.contamination = contamination
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.name = 'cnn'
        self.batch_size = batch_size

    def fit(self, X_clean, y=None):

        slidingwindow = self.slidingwindow
        predict_time_steps = self.predict_time_steps

        X_train, Y_train = create_dataset_for_cnn_and_lstm(X_clean)

        model = Sequential()
        model.add(Conv1D(filters=8,
                         kernel_size=2,
                         strides=1,
                         padding='same',
                         activation='relu',
                         input_shape=(X_train.shape[1], 1)))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Conv1D(filters=16,
                         kernel_size=2,
                         strides=1,
                         padding='valid',
                         activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Conv1D(filters=32,
                         kernel_size=2,
                         strides=1,
                         padding='valid',
                         activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=predict_time_steps))

        model.compile(loss='mse', optimizer='adam')

        # model.fit(X_train, Y_train,
        # epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=[es])
        model.fit(X_train, Y_train,
                  epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # model.fit(X_train,Y_train,validation_split=ratio,
        #           epochs=self.epochs,batch_size=64,verbose=self.verbose, callbacks=[es])

        self.estimator = model
        # self.n_initial = X_train.shape[0]

        return self

    def score(self, X):

        X_train, Y_train = create_dataset_for_cnn_and_lstm(X)
        predict = self.estimator.predict(X_train, batch_size=self.batch_size)
        # Euclidean distance
        ed = EuclideanDistance()
        return ed.score(Y_train.reshape(-1), predict.reshape(-1))

    def create_dataset(self, X, slidingwindow, predict_time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - slidingwindow - predict_time_steps + 1):
            tmp = X[i: i + slidingwindow + predict_time_steps]
            tmp = MinMaxScaler(feature_range=(0, 1)).fit_transform(tmp.reshape(-1, 1)).ravel()
            x = tmp[:slidingwindow]
            y = tmp[slidingwindow:]

            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)

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

        Y_test = self.Y_test

        score = []
        prediction = self.prediction

        for i in range(prediction.shape[0]):
            score.append(measure.measure(Y_test[i], prediction[i], 0))

        score = np.array(score)
        decision_scores_ = np.zeros(self.n_test_)

        decision_scores_[self.slidingwindow: (self.n_test_ - self.predict_time_steps + 1)] = score
        decision_scores_[: self.slidingwindow] = score[0]
        decision_scores_[self.n_test_ - self.predict_time_steps + 1:] = score[-1]

        self.decision_scores_ = decision_scores_
        return self
