import tensorflow as tf

def binary_crossentropy(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)