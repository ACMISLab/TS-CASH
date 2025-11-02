import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation as ani

from pylibs.utils.util_log import get_logger

log = get_logger()


def plot_3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, zs=z, zdir='z', label='curve in (x, y)')
    ax.scatter(x, y, zs=z, zdir='y', label='points in (x, z)')
    # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0

    return fig


def animation_demo():
    import matplotlib.figure
    import matplotlib.pyplot as plt
    import matplotlib.animation as ani
    import numpy as np

    fig, ax = plt.subplots()  # type: matplotlib.figure.Axes
    x_data = []
    y_data = []

    def update_function(frame):
        """
        When frames=0, call this ten times with frame in {1,2,3,...,10}.

        When frames=[1,3,10], call this three times with frame in {1,3,10}
        """
        print(f"update_function at {frame}")
        x_data.append(frame)
        y_data.append(np.sin(frame ** 2))
        ax.plot(x_data, y_data)

    def init_function():
        """
        Init operation, such as  ax.set_xlim((0,100))
        """
        pass

    animations = ani.FuncAnimation(
        fig=fig, func=update_function, frames=np.linspace(1, 100, 100), init_func=init_function,
        interval=200
    )

    # plt.show()
    # save to .gif or save to .mp4
    animations.save("ani.gif", writer="pillow")
    # animations.save("ani.mp4",writer="pillow")


def animation_latent_space_2dim(epoch_latent_array, label):
    """
    Save the epoch and latent space to gif.

    Examples
    --------

    .. code-block::

        # define the data and callback
        latent_space_data = []

        class ModelAnimationCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                global latent_space_data
                z = self.model.encoder_(x_test)
                latent_space_data.append((epoch, z.numpy()))


        # define the model
        vae = AutoEncoder(input_dim=x_train.shape[1], hyperparameter=_hpy)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        vae.compile(optimizer, run_eagerly=True)

        # fit
        vae.fit(
            x_train, x_train,
            epochs=_hpy[HyperParameter.EPOCHS],
            batch_size=_hpy[HyperParameter.BATCH_SIZE],
            callbacks=[ModelAnimationCallback()]
        )

        # call after fit
        animation_latent_space_2dim(latent_space_data, y_test)

    Parameters
    ----------
    epoch_latent_array : list
         [
            [epoch,[z_1, z_2],
            ....
         ]
    label : list
        the label of the epoch data

    Returns
    -------

    """
    fig, ax = plt.subplots()  # type: matplotlib.figure.Axes
    assert len(label.shape) == 1

    def update_function(data):
        epoch, frame = data
        ax.scatter(frame[:, 0], frame[:, 1], c=label)
        ax.set_title(f"epoch={epoch}")

    animations = ani.FuncAnimation(
        fig=fig,
        func=update_function,
        frames=epoch_latent_array,
        repeat=False,
    )
    img_file = f"{os.path.splitext(sys.argv[0])[0]}.gif"
    UtilSys.is_debug_mode() and log.info(f"Saving gif to {img_file}")
    animations.save(img_file, writer="pillow")


def save_latent_2dim_to_image(z, label, postfix=""):
    """

    """
    fig: matplotlib.pyplot.Figure = plt.figure(figsize=(15, 15))
    fig.suptitle(f"{postfix}")
    ax1 = fig.add_subplot(111)
    assert z.shape[1] == 2
    assert len(label.shape) == 1
    ax1.scatter(z[:, 1], z[:, 0], c=label)
    img_file = f"{os.path.splitext(sys.argv[0])[0]}{postfix}.png"
    UtilSys.is_debug_mode() and log.info(f"Saving gif to {img_file}")
    fig.savefig(img_file)


def save_latent_3dim_to_image(z, label, postfix=""):
    assert z.shape[1] == 3
    assert len(label.shape) == 1
    x, y, z = z[:, 0], z[:, 1], z[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, zs=z, zdir='y', c=label, label='points in (x, z)')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    img_file = f"{os.path.splitext(sys.argv[0])[0]}{postfix}_3d.png"
    fig.savefig(img_file)


def save_fig(fig: matplotlib.pyplot.Figure, path=None):
    home = f"{os.environ.get('HOME')}/rumtime/"
    if not os.path.exists(home):
        os.makedirs(home)
    if path is None:
        img_file = f"{home}/img_{time.time()}.png"
    else:
        img_file = path

    fig.savefig(img_file)


class AxHelper:
    @staticmethod
    def disable_grid(ax:matplotlib.pyplot.Axes):
        ax.grid(False)

    @classmethod
    def enable_grid(cls, ax):
        ax.grid(True)

    @classmethod
    def disable_xlable(cls,ax):
        ax.set_xlabel("")

    @classmethod
    def disable_ylable(cls, ax):
        ax.set_ylabel("")
    @classmethod
    def disabel_ytick_labels(cls, ax):
        ax.set_yticklabels([])

    @classmethod
    def disable_xytics(cls,ax):
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())


    @classmethod
    def set_xticks(cls):
        # _x_labels=np.arange(0, 8, 1)
        # ax.set_xticks(_x_labels,labels=_x_labels)
        pass
    @classmethod
    def adjust_subplot_padding(cls):
        # plt.subplots_adjust(wspace=0.1, hspace=0.52)
        pass
    @classmethod
    def adjust_subplot(cls):
        # plt.subplots_adjust(bottom=0.03, top=0.992, left=0.01, right=0.99)
        pass