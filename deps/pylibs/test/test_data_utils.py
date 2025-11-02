import os.path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from unittest import TestCase

from pylibs.utils.util_data import load_cv_data, load_data


class Test(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.file_path = os.path.join(os.path.dirname(__file__), "../utils/data/test_data.csv")

    def test_load_cv_data(self):
        for train_x, train_y, train_missing, test_x, test_y, test_missing, mean, std in \
                load_cv_data(self.file_path):
            assert len(train_x) + len(test_x) == 10

    def test_load_data(self):
        X, Y, missing, mean, std = load_data(self.file_path)
        assert len(X) > 0
        assert missing[-2] == 1

    def test_split_data(self):
        for train_x, train_y, train_missing, test_x, test_y, test_missing, mean, std in \
                load_cv_data(self.file_path):
            X_train, X_test, y_train, y_test, x_missing, y_missing = train_test_split(train_x, train_y, train_missing,
                                                                                      test_size=0.33, shuffle=False)
            print(X_train)

    def test_recall(self):
        m = tf.keras.metrics.Precision()
        m.update_state([0, 0, 0, 0], [0, 0, 0, 0])
        print(m.result().numpy())
