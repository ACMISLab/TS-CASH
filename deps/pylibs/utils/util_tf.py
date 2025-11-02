import os
import random
import time
import traceback
import warnings

import keras
import numpy as np
import sys
import tensorflow as tf
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_system import UtilSys

log = get_logger()


@DeprecationWarning
def enable_reproduce(seed=1, eager=True):
    """
    让每次训练的结果都一样，让实验可复现
    Parameters
    ----------
    seed

    Returns
    -------

    """
    UtilSys.is_debug_mode() and log.info(f"Set seed={seed} for all environment")
    tf.random.set_seed(seed)
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    if eager:
        disable_eager_execution_tf2()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


def enable_eager_execution_tf2():
    tf.compat.v1.enable_eager_execution()


def disable_eager_execution_tf2():
    tf.compat.v1.disable_eager_execution()


def using_gpu_without_memory_growth_tf2(gpu_index=0):
    """
     Using the specified GPU (only one GPU) to work and allocate all memory of the GPU.

    Returns
    -------

    """
    # todo: use specified gpu to work
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:

            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            UtilSys.is_debug_mode() and log.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            UtilSys.is_debug_mode() and log.info(e)


def set_all_devices_memory_growth(memory_limit=2048):
    """
    TF_FORCE_GPU_ALLOW_GROWTH=true python

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.get_memory_growth(gpus[0])
    Returns
    -------

    """

    """
 
  """
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        if tf.config.experimental.get_memory_growth(physical_device):
            # the gpu has enabled memory_growth
            continue
        else:
            tf.config.experimental.set_memory_growth(physical_device, True)
            tf.config.experimental.set_virtual_device_configuration(physical_device, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        assert tf.config.experimental.get_memory_growth(physical_device) is True


def using_gpu_with_memory_growth_tf2(gpu_index=-1):
    """
    Using the specified GPU (only one GPU) to work and don't allocate all memory of the GPU.

    Returns
    -------
    gpu_index : int, None
        -1 or None means to use CPU
        [0,N] means to use the specified GPU

    """
    UtilSys.is_debug_mode() and log.info(f"Specified GPU INDEX: {gpu_index}")
    if gpu_index == -1 or gpu_index is None:
        # use CPU to work
        UtilSys.is_debug_mode() and log.info("Using CPU to work!")
        return
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpu_index >= len(gpus):
            raise RuntimeError(
                f"Found {len(gpus)} GPUs, "
                f"but received gpu_index = {gpu_index} (expected {[i for i in range(len(gpus))]})")
        else:
            # Restrict TensorFlow to only use the specified GPU
            try:

                if gpu_index < 0:
                    raise RuntimeError("GPU exp_index must be >= 0")
                used_gpu = gpus[gpu_index]
                # fixed: "Attempting to perform BLAS operation using StreamExecutor without BLAS support" error occurs
                tf.config.set_visible_devices(used_gpu, 'GPU')
                tf.config.experimental.set_virtual_device_configuration(
                    used_gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
                # tf.config.experimental.set_virtual_device_configuration(
                #     used_gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
                # tf.config.experimental.set_memory_growth(used_gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                UtilSys.is_debug_mode() and log.info(f"Specified gpu success [{logical_gpus}] !")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                raise e


def select_gpu_and_limit_memory(gpu_index=-1, max_memory=1024):
    """
    Using the specified GPU (only one GPU) to work and don't allocate all memory of the GPU.

    Returns
    -------
    gpu_index : int, None
        -1 or None means to use CPU
        [0,N] means to use the specified GPU

    """
    UtilSys.is_debug_mode() and log.info(f"Specified GPU INDEX: {gpu_index}")
    if gpu_index == -1 or gpu_index is None:
        # use CPU to work
        UtilSys.is_debug_mode() and log.info("Using CPU to work!")
        return
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpu_index >= len(gpus):
            raise RuntimeError(
                f"Found {len(gpus)} GPUs, "
                f"but received gpu_index = {gpu_index} (expected {[i for i in range(len(gpus))]})")
        else:
            # Restrict TensorFlow to only use the specified GPU
            try:

                if gpu_index < 0:
                    raise RuntimeError("GPU exp_index must be >= 0")
                used_gpu = gpus[gpu_index]
                # fixed: "Attempting to perform BLAS operation using StreamExecutor without BLAS support" error occurs
                tf.config.set_visible_devices(used_gpu, 'GPU')
                tf.config.experimental.set_virtual_device_configuration(
                    used_gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                UtilSys.is_debug_mode() and log.info(f"Specified gpu success [{logical_gpus}] !")
            except RuntimeError as e:
                raise e


def select_gpu_and_limit_memory_tf291(gpu_index=0, max_memory=512):
    """
    Using the specified GPU (only one GPU) to work and don't allocate all memory of the GPU.

    Returns
    -------
    gpu_index : int, None
        -1 or None means to use CPU
        [0,N] means to use the specified GPU

    """
    UtilSys.is_debug_mode() and log.info(f"Specified GPU INDEX: {gpu_index}")
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    assert len(gpus) > 0, "Not found any GPU cards in this device."
    used_gpu = gpus[gpu_index]
    tf.config.set_visible_devices(used_gpu, 'GPU')
    tf.config.experimental.set_virtual_device_configuration(
        used_gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory)])
    visiable_devices = tf.config.get_visible_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    UtilSys.is_debug_mode() and log.info(f"Visible devices after set: {visiable_devices},logical gpu: {logical_gpus}")
    assert len(visiable_devices) == 1, "Failed to set visible device"


def allow_gpu_memory_growth():
    if UtilSys.is_macos():
        UtilSys.is_debug_mode() and log.info("Skip set GPU memory auto growth in Macos!")
        return
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus is not None:
            for gpu in gpus:
                UtilSys.is_debug_mode() and log.info(f"Set allowed memory growth for {gpu}")
                tf.config.experimental.set_memory_growth(gpu, True)


def select_gpu_and_allow_growth(gpu_index=-1):
    """
    Using the specified GPU (only one GPU) to work and don't allocate all memory of the GPU.

    Returns
    -------
    gpu_index : int, None
        -1 or None means to use CPU
        [0,N] means to use the specified GPU

    """
    if gpu_index < 0:
        warnings.warn(f"GPU exp_index is out of range, received {gpu_index}")
        return None
    UtilSys.is_debug_mode() and log.info(f"Specified GPU INDEX: {gpu_index}")
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    assert len(gpus) > 0, "Not found any GPU cards in this device."
    used_gpu = gpus[gpu_index]
    tf.config.set_visible_devices(used_gpu, 'GPU')
    visiable_devices = tf.config.get_visible_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    UtilSys.is_debug_mode() and log.info(f"Visible devices after set: {visiable_devices},logical gpu: {logical_gpus}")
    assert len(visiable_devices) == 1, "Failed to set visible device"


def using_all_cpu():
    """
    Specified all CPUs to train
    Returns
    -------
    gpu_index : int, None
        -1 or None means to use CPU
        [0,N] means to use the specified GPU

    """
    cpus = tf.config.list_physical_devices('CPU')
    tf.config.set_visible_devices(cpus, 'CPU')
    tf.config.set_visible_devices([], 'GPU')
    UtilSys.is_debug_mode() and log.info(f"Using CPU to work: {tf.config.list_logical_devices()}")


def allow_memory_growth_tf2():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            UtilSys.is_debug_mode() and log.info(f"{len(gpus)} Physical GPUs,{len(logical_gpus)} Logical GPU")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            UtilSys.is_debug_mode() and log.info(e)


def get_num_gpus():
    """
    Return the number of GPUS. Return 0 if there isn't GPU.

    You can't specify the GPU which you want to use if you call `device_lib.list_local_devices()`.

    Do not use the method `device_lib.list_local_devices()` to get the number of GPUs, using `nvidia-smi -L` instead.

    Returns
    -------
    int
        The number of GPUs.

    """
    sys.path.append("/root/miniconda3/envs/sw_torch_1_10/lib/python3.8/site-packages/tensorrt/")
    from pylibs.utils.util_bash import exec_cmd
    stdout, stderr = exec_cmd("nvidia-smi -L")
    count = 0
    for line in stdout:
        if str(line).find("GPU") > -1:
            count += 1
    UtilSys.is_debug_mode() and log.info(f"Number of GPUs is {count} on this device")
    return count


def enable_reproducible_result_tf2(raise_error=True):
    """
    Enable reproducible result for tensorflow2.

    https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

    Returns
    -------

    """
    import numpy as np
    import tensorflow as tf
    import random as python_random

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(123)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(123)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(1234)

    if os.environ.get('PYTHONHASHSEED') is None:
        errmsg = f"you need to set the environment variable PYTHONHASHSEED=0 " \
                 f"before the program starts. or run by:  \n" \
                 f"PYTHONHASHSEED=0 python3 {sys.argv[0]}"
        if raise_error:
            raise ValueError(errmsg)
        else:
            log.warning(errmsg)
    else:
        UtilSys.is_debug_mode() and log.info("Enable reproducible results")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def log_computed_device_tf2():
    """
    Show device(s) for computation.

    Returns
    -------

    """
    tf.debugging.set_log_device_placement(True)


def enable_growth_memory_tf2():
    """
    Set memory of tensorflow to growth.
    It means that it will not use all memory of GPU.

    Returns
    -------

    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            UtilSys.is_debug_mode() and log.info(f"{len(gpus)} Physical GPUs,{len(logical_gpus)} Logical GPU")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            # UtilSys.is_debug_mode() and
            traceback.print_exc()


class KerasCheckpointHelper:
    """
    A helper of keras Checkpoint.

    Examples
    ---------
    .. code-block::


        enable_reproducible_result_tf2()

        X_train, y_train, X_valid, y_valid = load_csv(DatasetName.ODDS_SHUTTLE, test_size=0.9, dataset_type=0, seed=1)

        ckpt_helper = KerasCheckpointHelper("./tf_ckpts")

        if ckpt_helper.has_checkoint():
            UtilSys.is_debug_mode()  and log.info(f"Restoring from {ckpt_helper.last_check_point} with epoch= {ckpt_helper.last_epoch}")
            vae = keras.models.parse_model(
                ckpt_helper.last_check_point,
                custom_objects={
                    "Decoder": Decoder,
                    "Encoder": Encoder,
                    "Sampling": Sampling,
                    "VariationalAutoEncoder": VariationalAutoEncoder
                }
            )
        else:
            UtilSys.is_debug_mode()  and log.info("Init a new model.")
            vae = VariationalAutoEncoder(input_dim=X_train.shape[1])
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

        vae.fit(
            X_train, X_train, epochs=2, batch_size=10000,
            initial_epoch=ckpt_helper.last_epoch,
            callbacks=[KerasModelCheckPointCallback(ckpt_helper.checkpoint_dir, freq=2, verbose=1)],
        )
    """

    def __init__(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self._checkpoint_dir = checkpoint_dir
        # todo: check  *.pb is exixted
        self._checkpoints = []
        self._epochs_in_checkpoint = []
        self._has_checkpoint = False
        for name in os.listdir(self._checkpoint_dir):
            home = os.path.join(self._checkpoint_dir, name)
            if os.path.exists(os.path.join(home, "keras_metadata.pb")) and \
                    os.path.exists(os.path.join(home, "saved_model.pb")):
                self._checkpoints.append(home)
                self._epochs_in_checkpoint.append(int(name.split('-')[-1]))
                self._has_checkpoint = True

        if self._has_checkpoint:
            _last_epoch_index = tf.argmax(self._epochs_in_checkpoint).numpy()
            self._last_epoch = self._epochs_in_checkpoint[_last_epoch_index]
            self._last_checkpoint = self._checkpoints[_last_epoch_index]
        else:
            self._last_epoch = 0
            self._last_checkpoint = None

    @property
    def last_check_point(self):
        """
        Return the checkpoint.


        Examples
        --------

        .. code-block::

            keras.models.parse_model(
                ckpt_helper.last_check_point
            )

        Returns
        -------

        """
        return self._last_checkpoint

    def has_checkoint(self):
        """
        Check whether it has checkpoint.

        Examples
        -------

        .. code-block::

            if ckpt_helper.has_checkoint():
                UtilSys.is_debug_mode()  and log.info(f"Restoring from {ckpt_helper.last_check_point} with epoch= {ckpt_helper.last_epoch}")
                vae = keras.models.parse_model(
                    ckpt_helper.last_check_point,
                    custom_objects={
                        "Decoder": Decoder,
                        "Encoder": Encoder,
                        "Sampling": Sampling,
                        "VariationalAutoEncoder": VariationalAutoEncoder
                    }
                )
            else:
                UtilSys.is_debug_mode()  and log.info("Init a new model.")
                vae = VariationalAutoEncoder(input_dim=X_train.shape[1])
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())


        Returns
        -------

        """
        return self._has_checkpoint

    @property
    def last_epoch(self):
        """
        The epoch corresponds to the checkpoint.

        Expamples
        ---------
        .. code-block::

            model.fit( ...,
                initial_epoch=ckpt_helper.last_epoch,
            )
        Returns
        -------

        """
        return self._last_epoch

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir


class KerasModelCheckPointCallback(keras.callbacks.Callback):
    """
    Save models by calling model.save() on `freq`, seeing https://github.com/keras-team/keras/blob/v2.10.0/keras/callbacks.py#L1212-L1710

    Examples
    --------
    Save model for every 10 epochs. (Save model in epoch=10,20,30,...)

    .. code-block::

         model.fit(...,
            callbacks=[KerasModelCheckPointCallback(checkpoint_dir, freq=10)],
        )

    """

    def __init__(self, filepath, verbose: int = 0, freq: int = 10):
        """
        Parameters
        ----------
        filepath : str
            The path for checkpoint
        verbose : int

        freq : int
            How frequent to save model.

            If freq=5, it will save model on epoch 5,10,15,20,...
        """
        self.filepath = filepath
        self.verbose = verbose
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        _real_epoch = epoch + 1
        if _real_epoch % self.freq == 0 and epoch > 0:
            _path = os.path.join(self.filepath, f"ckpt-{_real_epoch}")
            if self.verbose > 0:
                UtilSys.is_debug_mode() and log.info(f"\nSave model to {_path} on epoch={_real_epoch}")
            self.model.save(
                _path, overwrite=True
            )


def get_number_of_gpus_tf2():
    """
    Get  number of physical gpus on the machine.

    Returns
    -------
    int
        The number of gpus on the machine.

    """
    avail_gpus = len(tf.config.list_physical_devices('GPU'))
    UtilSys.is_debug_mode() and log.info(f"Found {avail_gpus} physical GPUs device(s).")
    return avail_gpus


def specify_gpus(gpus: str = None):
    """
    Limit the GPUs to work.

    If there have 5 available GPUs, '0,1' means it will not use the 2-4 GPUs, which only work on
    the 0 and 1 GPUs devices.

    Parameters
    ----------
    gpus : str
        The number of GPUs to limit to. e.g., '0,1' or '1,2' or '2,3'

    Parameters
    ----------
    gpus :

    Returns
    -------

    """
    if gpus is None:
        UtilSys.is_debug_mode() and log.info("Using all available gpus to train the model")
    else:
        gpus = gpus.strip().strip(",")
        gpus_index = [int(i) for i in gpus.split(',')]
        UtilSys.is_debug_mode() and log.info(f"Using gpus {gpus} to train the model")
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            avaliable_devices = [physical_devices[i] for i in gpus_index]
            tf.config.set_visible_devices(avaliable_devices, 'GPU')
        except:
            raise RuntimeError(f"Could not config visible devices for args={gpus}. ")

    return tf.config.list_logical_devices('GPU')


def select_one_gpu(exp_sequence, gpus: str):
    """

    Parameters
    ----------
    exp_sequence : int
        the sequence of experiment, exp_sequence is in [0,1,....,N], where N is the number of experiments.

    gpus : str
         The exp_index of the GPU to use for training. It will automate selecting a specified gpu to train the experiments.
         Multi-GPUs can be used by comma separated.
         e.g.,
         "gpus=0,1" means to use the first and second GPU to train the model.
         "gpus=-1" means to use CPUs to train the model


    Returns
    -------
    int
        The specified GPU to train this experiment.

    """

    UtilSys.is_debug_mode() and log.info("Specified gpus: %s" % gpus)
    if gpus == -1 or gpus == '-1':
        using_all_cpu()
    else:
        set_all_devices_memory_growth()
        gpus = get_specify_gpus_from_str(gpus)  # specify the gpus
        if len(gpus) > 0:
            target_gpus_index = exp_sequence % len(gpus)
            UtilSys.is_debug_mode() and log.info(f"Target gpus exp_index: {target_gpus_index}")
            if target_gpus_index < 0:
                using_all_cpu()
            else:

                using_gpu_with_memory_growth_tf2(gpus[target_gpus_index])
        else:
            log.warning(f"Not find GPU. {gpus}")


def get_specify_gpus_from_str(gpus: str = None):
    """
    Limit the GPUs to work.

    If there have 5 available GPUs, '0,1' means it will not use the 2-4 GPUs, which only work on
    the 0 and 1 GPUs devices.

    Parameters
    ----------
    gpus : str
        The number of GPUs to limit to. e.g., '0,1' or '1,2' or '2,3'

    Parameters
    ----------
    gpus :

    Returns
    -------

    """

    if gpus is None or gpus == "None":
        gpus_index = [i for i in range(get_number_of_gpus_tf2())]
    else:
        gpus = gpus.strip().strip(",")
        gpus_index = [int(i) for i in gpus.split(',')]

    return gpus_index


def metrics_auc(y_true, score):
    m = tf.keras.metrics.AUC()
    m.reset_state()
    m.update_state(y_true, score)
    auc = m.result().numpy()
    return auc


def metrics_outlier_factor(x, recon_x):
    """
    Calculate score, which is defined (Cited by 902)  in https://link.springer.com/chapter/10.1007/3-540-46145-0_17

    Parameters
    ----------
    data :

    Returns
    -------

    """

    score = tf.reduce_mean(tf.square(recon_x - x), axis=1)
    return score


def get_model_save_home(prefix=''):
    home = os.getcwd()
    model_save_dir = os.path.abspath(os.path.join(home, f"../../keras_models/{prefix}_{time.time_ns()}"))
    UtilSys.is_debug_mode() and log.info(f"Save model to {model_save_dir}")
    return model_save_dir


from keras.utils import Sequence


class TFDataGenerator(Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set,
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x

# train_gen = DataGenerator(X_train, y_train, 32)
# test_gen = DataGenerator(X_test, y_test, 32)
#
#
# history = model.fit(train_gen,
#                     epochs=6,
#                     validation_data=test_gen)
