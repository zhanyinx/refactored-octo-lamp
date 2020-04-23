import tensorflow as tf

from .util import conv_block


def resnet50(n_channels: int) -> tf.keras.models.Model:
    """ ResNet50 model with pretrained weights and skip connections to upconvolve. """

    inputs = tf.keras.layers.Input((None, None, 1))
    inputs_concat = tf.keras.layers.Concatenate()([inputs, inputs, inputs])

    model = tf.keras.applications.ResNet50(
        include_top=False, input_tensor=inputs_concat, weights='imagenet')

    # Freeze layers
    for layer in model.layers:
        layer.trainable = False

    # Get skip connections (256)
    skip_names = ["conv1_relu",  # 128
                  "conv2_block3_out",  # 64
                  "conv3_block4_out",  # 32
                  "conv4_block6_out",  # 16
                  "conv5_block3_out"]  # 8 -> Final
    layer_names = [layer.name for layer in model.layers]
    skip_indices = [layer_names.index(name) for name in skip_names]
    skip_layers = [model.layers[i] for i in skip_indices][::-1]
    filters = [layer.input.shape[-1] for layer in skip_layers]

    # Upconvolution
    x = model.layers[-1].output
    for f, skip in zip(filters, skip_layers):
        x = conv_block(inputs=x, filters=f)
        x = tf.keras.layers.Concatenate()([x, skip.output])
        x = tf.keras.layers.UpSampling2D()(x)

    # Final layers
    x = conv_block(inputs=x, filters=64)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=1)(x)
    x = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=1)(x)
    if n_channels == 1:
        x = tf.keras.layers.Activation("sigmoid")(x)
    else:
        x = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
