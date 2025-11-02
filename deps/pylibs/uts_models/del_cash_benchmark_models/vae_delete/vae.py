import os.path

import keras
import tensorflow as tf
from keras import layers
from keras.engine import data_adapter
from keras.regularizers import l2

from pylibs.uts_models.benchmark_models.vae.vae_conf import VAEConfig
from pylibs.utils.util_log import get_logger

log = get_logger()


class Sampling(keras.layers.Layer):
    """
    Sampling from Gaussian, N(0,I)
    To sample from epsilon = Norm(0,I) instead of from likelihood Q(z|X)
       with latent variables z: z = z_mean + sqrt(var) * epsilon

    Parameters
    ----------

    Returns
    -------
    z : tensor
           Sampled latent variable.
    """

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))  # mean=0, std=1.0
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(keras.layers.Layer):
    """
    """

    def __init__(self, config: VAEConfig):
        super(Encoder, self).__init__()
        self._config = config
        inputs = layers.Input(shape=(self._config.window_size,), name="EncoderInput")
        _layer_origin = inputs

        # for neurons in hpy[HyperParameter.ENCODER_NEURONS]:
        for neurons in self._config.encoder_neurons:
            _layer_origin = layers.Dense(
                neurons,
                activation=self._config.hidden_activation,
                activity_regularizer=l2(self._config.l2_regularizer)
            )(_layer_origin)
            _layer_origin = layers.Dropout(self._config.drop_rate)(_layer_origin)

        # Create mu and sigma of latent variables
        z_mean = layers.Dense(self._config.latent_dim, name="z_mean")(_layer_origin)
        z_log = layers.Dense(self._config.latent_dim, name="z_log_var")(_layer_origin)

        z = Sampling()([z_mean, z_log])
        self.encoder = keras.Model(inputs, [z_mean, z_log, z], name="vae_encoder")
        # self.encoder.summary(print_fn=UtilSys.is_debug_mode()  and log.info)

    def call(self, inputs):
        try:
            z_mean, z_log, z = self.encoder(inputs)
            return z_mean, z_log, z
        except Exception as e:
            log.error(traceback.format_exc())
            raise e

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"hpy": self._hpy})
        return config


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, conf: VAEConfig, **kwargs):
        super(Decoder, self).__init__()
        self._conf = conf
        latent_inputs = layers.Input(shape=(self._conf.latent_dim,), name="latent_inputs")
        _hiden = latent_inputs
        for neurons in self._conf.decoder_neurons:
            _hiden = layers.Dense(neurons, activation=self._conf.hidden_activation)(_hiden)
            _hiden = layers.Dropout(self._conf.drop_rate)(_hiden)

        outputs = layers.Dense(
            self._conf.window_size,
            activation=self._conf.output_activation,
            name="outputs"
        )(_hiden)
        self.decoder = keras.Model(latent_inputs, outputs, name="vae_decoder")
        # self.decoder.summary(print_fn=UtilSys.is_debug_mode()  and log.info)

    def call(self, inputs, **kwargs):
        return self.decoder(inputs)

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"hpy": self._hpy})
        return config


class _VAE(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training.
    Examples seeing examples/vae_*.py
    """

    # def __init__(self, input_dim, hyperparameter=None, loss=tf.keras.losses.mean_squared_error):
    def __init__(self, conf: VAEConfig, loss=tf.keras.losses.mean_squared_error, device=None):
        """
        Init a variational autoencoder

        Parameters
        ----------
        loss :  function
            A loss function of keras.
            Seeing https://keras.io/api/losses/regression_losses/#meansquarederror-function
        """

        super(_VAE, self).__init__()
        self.configs = conf
        self._loss = loss
        self.encoder_ = Encoder(self.configs)
        self.decoder_ = Decoder(self.configs)

    def call(self, inputs, **kwargs):
        z_mean, z_log, z = self.encoder_(inputs)
        recon_x = self.decoder_(z)
        return z_mean, z_log, z, recon_x

    def train_step(self, data):
        train_x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            z_mean, z_log, z, recon_x = self(train_x, training=True)
            # UtilSys.is_debug_mode()  and log.info(f"Training step entered:"
            #          f"\nx.shape{x.shape},"
            #          f"\nz_mean.shape{z_mean.shape}"
            #          f"\nz_log.shape{z_log.shape}"
            #          f"\nrecon_x.shape{recon_x.shape}")
            # loss, recon_loss, kl_loss = self._build_loss(x, recon_x, z_log, z_mean)
            loss, recon_loss, kl_loss = self._build_loss(train_x, recon_x, z_log, z_mean)
        #
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def _build_loss(self, train_x, recon_x, z_log_var, z_mean):
        # 'Auto-Encoding Variational Bayes'  https://arxiv.org/abs/1312.6114 for details.
        # Reference pyod.models.vae
        kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
        kl_loss = -0.5 * tf.math.reduce_sum(kl_loss, axis=-1)
        n_feature = train_x.shape[1]
        recon_loss = n_feature * self._loss(train_x, recon_x)  # correct
        loss = tf.math.reduce_mean(recon_loss + kl_loss)
        return loss, tf.reduce_mean(recon_loss), tf.reduce_mean(kl_loss)

    def test_step(self, data):
        train_x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        z_mean, z_log_var, z = self.encoder_(train_x)
        recon_x = self.decoder_(z)
        loss, recon_loss, kl_loss = self._build_loss(train_x, recon_x, z_log_var, z_mean)
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def predict(self, data):
        """
        The reconstruction x.

        Parameters
        ----------
        data :

        Returns
        -------

        """
        return self(data, training=False)[-1]

    def test_losses(self, x, y=None, eval_times=10):
        """
        Return the loss for each eval_times evaluation. E.g.

        .. code-block::

            return {
             'test_loss_0': 89.717041015625,
             'test_loss_1': 90.60455322265625,
             'test_loss_...': 88.45326232910156,

             'test_loss_kl_0': 15.397006034851074,
             'test_loss_kl_1': 15.397006034851074,
             'test_loss_kl_...': 15.397006034851074,

             'test_loss_recon_0': 74.3200454711914,
             'test_loss_recon_1': 75.2075424194336,
             'test_loss_recon_...': 73.05625915527344,
             }

        Parameters
        ----------
        x :
        y :
        eval_times : int
            The number of times to evaluate

        Returns
        -------
        dict
            a set of loss representation by dict.



        """
        eval_losses = {}
        for i in range(eval_times):
            _losses = self.evaluate(x, return_dict=True, verbose=0)
            eval_losses[f"test_loss_{i}"] = _losses['loss']
            eval_losses[f"test_loss_recon_{i}"] = _losses['recon_loss']
            eval_losses[f"test_loss_kl_{i}"] = _losses['kl_loss']
        return eval_losses

    @property
    def hyperparameters(self):
        return self._hpy

    @staticmethod
    def load_model(model_path):
        return keras.models.parse_model(
            model_path,
            custom_objects={
                "Decoder": Decoder,
                "Encoder": Encoder,
                "Sampling": Sampling,
                "_VAEModel": _VAE
            }
        )


class VAEModel:
    """A modified version of the COCA: https://github.com/ruiking04/COCA"""

    def score(self, data):
        recon_x = self.model.predict(data)
        score = tf.reduce_mean(tf.square(recon_x - data), axis=1)
        return score.numpy()

    def __init__(self, config: VAEConfig = None, verbose=-1):
        self.is_deep = True
        if config is None:
            self.config = VAEConfig()
        else:
            self.config = config
        self.verbose = verbose
        self.model: _VAE = _VAE(self.config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.model.compile(optimizer, run_eagerly=True)

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, train_x, y=None):
        """

        Parameters
        ----------
        train_x :
        y : None
            Not used, which is for API consistency by convention.

        Returns
        -------

        """
        self.model.fit(
            train_x,
            train_x,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            verbose=self.verbose,
        )

    def save_model(self, filepath):

        if not filepath.endswith(".h5"):
            filepath += ".h5"
        self.model.save_weights(filepath)
        return filepath

    def load_model(self, filepath):
        self.model.built = True
        self.model.load_weights(filepath)
        return self
