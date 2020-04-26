import tensorflow.keras.backend as K
import tensorflow as tf
from .f1_score import f1_score_loss, f1_score


def l2_norm(y_true, y_pred):
    """
    Calculate L2 norm between true and predicted coordinates 
    """
    if not K.ndim(y_true) == K.ndim(y_pred):
        raise ValueError(
            f"true/pred shapes must match: {y_true.shape} != {y_pred.shape}"
        )

    coord_true = y_true[..., 1:]
    coord_pred = y_pred[..., 1:]

    comparison = tf.equal(coord_true, tf.constant(0, dtype=tf.float32))

    coord_true_new = tf.where(comparison, tf.zeros_like(coord_true), coord_true)
    coord_pred_new = tf.where(comparison, tf.zeros_like(coord_pred), coord_pred)

    l2_norm = K.mean(K.sum(K.square(coord_true_new - coord_pred_new), axis=-1))

    return l2_norm


def f1_l2_combined_loss(y_true, y_pred):
    return l2_norm(y_true, y_pred) + f1_score_loss(y_true, y_pred)
