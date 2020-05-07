"""Categorical crossentropy function."""

import tensorflow as tf


def categorical_crossentropy(y_true, y_pred):
    """Categorical cross-entropy."""
    return tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)
