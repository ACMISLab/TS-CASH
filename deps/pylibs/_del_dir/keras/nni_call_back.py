import nni
from tensorflow import keras

from pylibs.common import ConstKeras


class NNITrainingCallback(keras.callbacks.Callback):
    """
    Report loss to nni by Keras call back.
    """
 
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0 and logs is None:
            loss = logs[ConstKeras.KEY_TRAINING_LOSS]
            valid_loss = logs[ConstKeras.KEY_VAL_LOSS]
            nni.report_intermediate_result({
                "training_epoch": epoch,
                "training_loss": loss,
                'valid_loss': valid_loss
            })
