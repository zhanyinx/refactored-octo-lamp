"""Binary crossentropy function."""

import tensorflow as tf


def binary_crossentropy(y_true, y_pred):
    """Return the binary crossentropy loss."""
    return tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
