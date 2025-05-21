import os


from ucr.ucr_dataset_loader import calc_acc_score, load_ucr_by_number

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import mean_absolute_error


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEModel(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(0, 1)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        return self.decoder(z)



class VAE:
    def __init__(
            self, epoch=5, batch_size=128, window_size=128, latent_dim=2,
    ):
        self.epoch = epoch
        self.batch_size = batch_size
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.detect_pos_ = None
        self.anomaly_scores_ = None

    def fit(self, X_train):
        X_train = tf.expand_dims(X_train, axis=-1)
        encoder_inputs = keras.Input(shape=(self.window_size, 1))
        x = layers.Conv1D(8, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv1D(16, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        # encoder.summary()

        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((32, 64))(x)
        x = layers.Conv1DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv1DTranspose(8, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        # decoder.summary()

        print("++++++++++++", X_train.shape)
        self.vae_model = VAEModel(encoder, decoder)
        self.vae_model.compile(optimizer=keras.optimizers.Adam())
        self.vae_model.fit(X_train, epochs=self.epoch, shuffle=False, batch_size=self.batch_size)

    def accuracy_score(self, X_test, baseline_range):
        X_test = np.expand_dims(X_test, axis=-1)
        X_pred = self.vae_model.predict(X_test)
        X_pred = np.nan_to_num(X_pred)
        # print("++++++++++++++result.shape:", X_pred.shape)

        X_score = []
        for test, pred in zip(X_test, X_pred):
            score = mean_absolute_error(test, pred)
            X_score.append(score)

        self.anomaly_scores_ = X_score
        self.anomaly_pos_ = np.argmax(X_score)
        score = calc_acc_score(baseline_range, self.anomaly_pos_, len(X_test))
        return score

    @staticmethod
    def get_hyperparameter_search_space(

    ):
        pass
        # cs = ConfigurationSpace()
        #
        # n_estimators = UniformIntegerHyperparameter(
        #     name="n_estimators", lower=50, upper=500, default_value=50, log=False
        # )
        # learning_rate = UniformFloatHyperparameter(
        #     name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True
        # )
        # algorithm = CategoricalHyperparameter(
        #     name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R"
        # )
        # max_depth = UniformIntegerHyperparameter(
        #     name="max_depth", lower=1, upper=10, default_value=1, log=False
        # )
        #
        # cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth])
        # return cs


if __name__ == "__main__":
    from libs import eval_model
    vae = VAE()
    eval_model(vae)
