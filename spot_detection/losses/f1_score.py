"""F1 score metrics and loss function."""

import tensorflow.keras.backend as K


def recall_score(y_true, y_pred):
    """Recall score metrics."""
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_true, y_pred):
    """Precision score metrics."""
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    """F1 = 2 * (precision*recall) / (precision+recall)."""
    if not K.ndim(y_true) == K.ndim(y_pred):
        raise ValueError(f"true/pred shapes must match: {y_true.shape} != {y_pred.shape}")

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def f1_score_loss(y_true, y_pred):
    """F1 score loss."""
    return 1 - f1_score(y_true, y_pred)
