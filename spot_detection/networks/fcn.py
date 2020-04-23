import tensorflow as tf

from .util import convpool_block, OPTIONS_CONV


def fcn32(n_channels: int) -> tf.keras.models.Model:
    """ Simplest FCN architecture without skips. """

    inputs = tf.keras.layers.Input(shape=(None, None, 1))

    # Convs
    x = tf.keras.layers.Conv2D(filters=32, strides=1, **OPTIONS_CONV)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=64, strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Down: 256 -> 32
    x = tf.keras.layers.Conv2D(filters=128, strides=2, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=256, strides=2, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=512, strides=2, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Connected
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.UpSampling2D((8, 8))(x)
    if n_channels == 1:
        x = tf.keras.layers.Activation("sigmoid")(x)
    else:
        x = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def fcn16(n_channels: int) -> tf.keras.models.Model:
    """ FCN architecture with one skip connection. """

    inputs = tf.keras.layers.Input((None, None, 1))

    # Down: 256 -> 16
    x = convpool_block(inputs=inputs, filters=64, n_convs=2)
    x = convpool_block(inputs=x, filters=128, n_convs=2)
    x = convpool_block(inputs=x, filters=256, n_convs=3)
    skip = x
    x = convpool_block(inputs=x, filters=512, n_convs=3)

    # Up: 16 -> 128
    skip = tf.keras.layers.Conv2D(filters=32, **OPTIONS_CONV)(skip)
    x = tf.keras.layers.Conv2DTranspose(
        filters=32, strides=(2, 2), **OPTIONS_CONV)(x)
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Conv2DTranspose(
        filters=32, strides=(4, 4), **OPTIONS_CONV)(x)

    # Connected
    x = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.UpSampling2D()(x)
    if n_channels == 1:
        x = tf.keras.layers.Activation("sigmoid")(x)
    else:
        x = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def fcn8(n_channels: int) -> tf.keras.models.Model:
    """ FCN architecture with two skip connections. """

    inputs = tf.keras.layers.Input((None, None, 1))

    # Down: 256 -> 16
    x = convpool_block(inputs=inputs, filters=64, n_convs=2)
    x = convpool_block(inputs=x, filters=128, n_convs=2)
    skip2 = x
    x = convpool_block(inputs=x, filters=256, n_convs=3)
    skip1 = x
    x = convpool_block(inputs=x, filters=512, n_convs=3)

    # Up 1: 16 -> 32
    skip1 = tf.keras.layers.Conv2D(filters=32, **OPTIONS_CONV)(skip1)
    x = tf.keras.layers.Conv2DTranspose(
        filters=32, strides=(2, 2), **OPTIONS_CONV)(x)
    x = tf.keras.layers.Add()([x, skip1])

    # Up 2: 32 -> 64
    skip2 = tf.keras.layers.Conv2D(filters=32, **OPTIONS_CONV)(skip2)
    x = tf.keras.layers.Conv2DTranspose(
        filters=32, strides=(2, 2), **OPTIONS_CONV)(x)
    x = tf.keras.layers.Add()([x, skip2])

    # Up 3: 64 -> 128
    x = tf.keras.layers.Conv2DTranspose(
        filters=32, strides=(2, 2), **OPTIONS_CONV)(x)

    # Connected
    x = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.UpSampling2D()(x)
    if n_channels == 1:
        x = tf.keras.layers.Activation("sigmoid")(x)
    else:
        x = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
