import numpy as np
import pytest
import sys
sys.path.append("../../")

from training.util_metrics import dice_coef


def test_dice_coef():

    # Empty score
    assert dice_coef(np.zeros((100, 100)), np.zeros((100, 100)), 20.0) == 20.0

    # Values
    assert dice_coef(np.ones((100, 100)), np.ones((100, 100))) == 1.0
    assert dice_coef(np.ones((100, 100)), np.zeros((100, 100))) == 0.0
    assert dice_coef(np.zeros((100, 100)), np.ones((100, 100))) == 0.0
    a = np.zeros((100, 100))
    a[0:20] = 1
    assert dice_coef(np.ones((100, 100)), a) == 0.3333333333333333
    a[0:50] = 1
    assert dice_coef(np.ones((100, 100)), a) == 0.6666666666666666
    a[0:60] = 1
    assert dice_coef(np.ones((100, 100)), a) == 0.75

    # Errors
    with pytest.raises(ValueError):
        dice_coef(np.ones((100, 100)), np.ones((90, 90)))
    with pytest.raises(TypeError):
        dice_coef([1, 1, 1], [0, 0, 0])