import tensorflow as tf
from typing import Tuple


OPTIONS_CONV = {
    "kernel_size": 3,
    "padding": "same",
    "kernel_initializer": "he_normal"
}


def conv_block(inputs: tf.keras.layers.Layer,
               filters: int,
               n_convs: int = 2) -> tf.keras.layers.Layer:
    """ Convolutional block. """

    x = inputs
    for _ in range(n_convs):
        x = tf.keras.layers.Conv2D(filters, **OPTIONS_CONV)(inputs)
        x = tf.keras.layers.Activation("relu")(x)

    return x


def convpool_block(inputs: tf.keras.layers.Layer,
                   filters: int,
                   n_convs: int = 2) -> tf.keras.layers.Layer:
    """ n_convs * (Convolution -> ReLU) -> MaxPooling. """

    x = inputs
    for _ in range(n_convs):
        x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)
        x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    return x


def convpool_skip_block(inputs: tf.keras.layers.Layer,
                   filters: int,
                   n_convs: int = 2) -> Tuple[tf.keras.layers.Layer]:
    """ n_convs * (Convolution -> ReLU) -> MaxPooling. """

    x = inputs
    for _ in range(n_convs):
        x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)
        x = tf.keras.layers.Activation("relu")(x)
    skip = x
    x = tf.keras.layers.MaxPooling2D()(x)

    return skip, x


def upconv_block(inputs: tf.keras.layers.Layer,
                 skip: tf.keras.layers.Layer,
                 filters: int,
                 n_convs: int = 2) -> tf.keras.layers.Layer:
    """ Upsampling -> Conv -> Concat -> Conv. """
    x = inputs
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    for _ in range(n_convs):
        x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)
        x = tf.keras.layers.Activation("relu")(x)

    return x

