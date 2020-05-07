"""RMSprop optimizer function."""

import tensorflow as tf


def rmsprop(learning_rate):
    """Return the rmsprop optimizer with a specified learning rate."""
    return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
