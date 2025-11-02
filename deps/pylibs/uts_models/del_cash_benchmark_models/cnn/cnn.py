from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
import numpy as np
from pylibs.uts_models.benchmark_models.utils.distance import Fourier


class CNNModel:
    def __init__(self,
                 slidingwindow=64, predict_time_steps=1,
                 contamination=0.1,
                 epochs=100,
                 patience=10,
                 batch_size=64,
                 verbose=0):
        self._is_trained = None
        self.is_deep = True
        self.batch_size = batch_size
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.contamination = contamination
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.model_name = 'cnn'

        model = Sequential()
        self._model = model
        model.add(Conv1D(filters=8,
                         kernel_size=2,
                         strides=1,
                         padding='same',
                         activation='relu',
                         input_shape=(slidingwindow, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=16,
                         kernel_size=2,
                         strides=1,
                         padding='valid',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32,
                         kernel_size=2,
                         strides=1,
                         padding='valid',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=predict_time_steps))

        model.compile(loss='mse', optimizer='adam')
        self._model = model

    def fit(self, X_clean, y=None):
        """

        Parameters
        ----------
        X_clean :
        y : None
            Not used, which is for API consistency by convention.

        Returns
        -------

        """

        slidingwindow = self.slidingwindow
        predict_time_steps = self.predict_time_steps
        # self.n_test_ = len(X_dirty)

        X_train, Y_train = self.create_dataset(X_clean, slidingwindow, predict_time_steps)
        if X_train.shape[0] == 0:
            self._is_trained = False
            return

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=self.patience)
        # with tf.device('/cpu:0'):
        #     X_train = tf.convert_to_tensor(X_train, tf.float32)
        #     Y_train = tf.convert_to_tensor(Y_train, tf.float32)

        self._model.fit(X_train, Y_train,
                        epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        self._is_trained = True
        # prediction = model.predict(X_test)
        #
        # self.Y_test = Y_test
        # self.prediction = prediction
        # self.estimator = model
        # self.n_initial = X_train.shape[0]

        return self

    def score(self, X):
        n_test = len(X)

        measure = Fourier()
        measure.detector = self
        measure.set_param()
        X_test, Y_test = self.create_dataset(X, self.slidingwindow, self.predict_time_steps)
        prediction = self._model.predict(X_test, verbose=self.verbose)
        score = []
        for i in range(prediction.shape[0]):
            score.append(measure.measure(Y_test[i], prediction[i], 0))

        score = np.array(score)

        decision_scores_ = np.zeros(n_test)

        decision_scores_[self.slidingwindow: (n_test - self.predict_time_steps + 1)] = score
        decision_scores_[: self.slidingwindow] = score[0]
        decision_scores_[n_test - self.predict_time_steps + 1:] = score[-1]
        return decision_scores_

    def create_dataset(self, X, slidingwindow, predict_time_steps=1):
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

    def save_model(self, filepath):
        if not filepath.endswith(".h5"):
            filepath += ".h5"
        self._model.save_weights(filepath)
        return filepath

    def load_model(self, filepath):
        self._model.built = True
        self._model.load_weights(filepath)
        return self
