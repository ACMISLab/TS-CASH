# check search space
import tensorflow as tf

from pylibs.common import HyperParameter, ConstActivationFunction


def valid_donut_search_space(search_space):
    search_space[HyperParameter.LATENT_DIM] = int(search_space[HyperParameter.LATENT_DIM])
    search_space[HyperParameter.WINDOW_SIZE] = int(search_space[HyperParameter.WINDOW_SIZE])
    search_space[HyperParameter.HIDDEN_NEURONS] = int(search_space[HyperParameter.HIDDEN_NEURONS])
    search_space[HyperParameter.NUM_L_SAMPLES] = int(search_space[HyperParameter.NUM_L_SAMPLES])
    search_space[HyperParameter.BATCH_SIZE] = int(search_space[HyperParameter.BATCH_SIZE])
    search_space[HyperParameter.LR_ANNEAL_EPOCHS] = int(search_space[HyperParameter.LR_ANNEAL_EPOCHS])
    search_space[HyperParameter.EPOCHS] = int(search_space[HyperParameter.EPOCHS])
    search_space[HyperParameter.GRAD_CLIP_NORM] = int(search_space[HyperParameter.GRAD_CLIP_NORM])
    # "relu", 1
    # HyperParameter.ACTIVATION_FUNCTION,1
    # "softmax", 1
    # "softplus",1
    # "tanh", 1
    # "selu" 1

    if type(search_space[HyperParameter.ACTIVATION_FUNCTION]) == str:
        if str(search_space[HyperParameter.ACTIVATION_FUNCTION]).lower() == ConstActivationFunction.RELU:
            search_space[HyperParameter.ACTIVATION_FUNCTION] = tf.nn.relu
        if str(search_space[HyperParameter.ACTIVATION_FUNCTION]).lower() == ConstActivationFunction.SELU:
            search_space[HyperParameter.ACTIVATION_FUNCTION] = tf.nn.selu
        if str(search_space[HyperParameter.ACTIVATION_FUNCTION]).lower() == ConstActivationFunction.TANH:
            search_space[HyperParameter.ACTIVATION_FUNCTION] = tf.nn.tanh
        if str(search_space[HyperParameter.ACTIVATION_FUNCTION]).lower() == ConstActivationFunction.SOFTMAX:
            search_space[HyperParameter.ACTIVATION_FUNCTION] = tf.nn.softmax
        if str(search_space[HyperParameter.ACTIVATION_FUNCTION]).lower() == ConstActivationFunction.SIGMOID:
            search_space[HyperParameter.ACTIVATION_FUNCTION] = tf.nn.sigmoid
        if str(search_space[HyperParameter.ACTIVATION_FUNCTION]).lower() == ConstActivationFunction.SOFTPLUS:
            search_space[HyperParameter.ACTIVATION_FUNCTION] = tf.nn.softplus

    assert type(search_space[HyperParameter.ACTIVATION_FUNCTION]) != str
    assert type(search_space[HyperParameter.WINDOW_SIZE]) == int
    assert type(search_space[HyperParameter.BATCH_SIZE]) == int
    assert type(search_space[HyperParameter.MISSING_DATA_INJECTION_RATE]) == float
    assert type(search_space[HyperParameter.LEARNING_RETE]) == float
    assert type(search_space[HyperParameter.GRAD_CLIP_NORM]) == int

    return search_space
