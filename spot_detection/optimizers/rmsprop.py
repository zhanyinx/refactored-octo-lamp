import tensorflow as tf

def rmsprop(learning_rate):
    return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
