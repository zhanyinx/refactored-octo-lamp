import tensorflow as tf
import numpy as np

from .util import convpool_skip_block, upconv_block, OPTIONS_CONV


def unet(n_channels: int, depth: int = 4, width: int = 64) -> tf.keras.models.Model:
    """
    Flexible UNet architecture with variable depth and width.
    Args:
        - n_channels: Number of output channels determining over final activation.
        - depth: Number of layers of down / up convolutions.
        - width: Initial filter count. Doubles every layer depth.
    Returns
    """
    w = int(np.log2(width))

    inputs = tf.keras.layers.Input((None, None, 1))

    # Down
    skips = []
    x = inputs
    for d in range(depth):
        skip, x = convpool_skip_block(inputs=x, filters=2**(w+d))
        skips.append(skip)

    # Bottom
    x = tf.keras.layers.Conv2D(filters=2**(w+1+d), **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=2**(w+1+d), **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Up
    for d, skip in enumerate(skips[::-1]):
        x = upconv_block(inputs=x, skip=skip, filters=2**(w+depth-1-d))

    # Connected
    x = tf.keras.layers.Conv2D(filters=2, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=1)(x)
    if n_channels == 1:
        x = tf.keras.layers.Activation("sigmoid")(x)
    else:
        x = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.models.Model(inputs=inputs, outputs=x)
