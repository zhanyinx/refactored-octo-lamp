import tensorflow as tf

def adam(learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)
