import os.path
import tensorboard as tf
import keras
import numpy as np
from joblib import dump, load
from numpy.testing import assert_almost_equal

from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_directory import make_dirs

from pylibs.utils.util_log import get_logger

log = get_logger()


def get_earning_stop_callback():
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    return callback


class UtilKeras:
    @staticmethod
    def save_model(model, model_path, home=UtilComm.get_runtime_directory()):
        """
         from sklearn import svm
        from sklearn import datasets
        clf = svm.SVC()
        X, y= datasets.load_iris(return_X_y=True)
        clf.fit(X, y)

        UtilSK.save_model(clf,"aaa.keras")
        clf=UtilSK.load_model("aaa.keras")

        Parameters
        ----------
        model : class
            The sklearn model
        model_path :
        home :

        Returns
        -------

        """

        if not model_path.endswith(".keras"):
            model_path = model_path + ".keras"
        model_path = os.path.join(home, model_path)

        make_dirs(home)
        UtilSys.is_debug_mode() and log.info(f"Model is saved to {os.path.abspath(model_path)}")
        model.save(model_path)
        return model_path

    @staticmethod
    def load_model(model_path):
        """

        Parameters
        ----------
        model_path :

        Returns
        -------

        """
        if not model_path.endswith(".keras"):
            model_path = model_path + ".keras"
        # return keras.models.load_model("my_model.keras")
        return keras.models.load_model(model_path)


if __name__ == '__main__':
    def get_model():
        # Create a simple model.
        inputs = keras.Input(shape=(32,))
        outputs = keras.layers.Dense(1)(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
        return model


    model = get_model()

    # Train the model.
    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    model.fit(test_input, test_target)

    # Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
    model.save("my_model.keras")
    path = UtilKeras.save_model(model, "test_model")
    # It can be used to reconstruct the model identically.
    reconstructed_model = keras.models.load_model("my_model.keras")

    # Let's check:
    np.testing.assert_allclose(
        model.predict(test_input), reconstructed_model.predict(test_input)
    )
    my_loaded = UtilKeras.load_model(path)
    np.testing.assert_allclose(
        model.predict(test_input), my_loaded.predict(test_input)
    )
