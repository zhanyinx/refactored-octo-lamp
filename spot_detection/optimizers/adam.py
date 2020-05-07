"""Adam optimizer function."""

import tensorflow as tf


def adam(learning_rate):
    """Return the adam optimizer with a specified learning rate."""
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)
