import pytest
import sys
sys.path.append("../../")

from training.util_prepare import extract_basename, train_valid_split


def test_extract_basename():
    assert extract_basename("././././///test.ext") == "test"
    assert extract_basename("../test.ext") == "test"
    assert extract_basename("../test/") == ""


def test_train_valid_split():
    x = [i for i in range(10)]
    y = [i for i in range(10)] 

    # Functionality
    try:
        x1, x2, y1, y2 = train_valid_split(x, y, 0.2)
        assert len(x1) == len(y1)
        assert len(x2) == len(y2)
        assert len(x1) == 8
    except Exception:
        assert 0
    
    # Errors
    with pytest.raises(TypeError):
        train_valid_split(1, 2, 0.2)
    with pytest.raises(ValueError):
        train_valid_split([], [], 0.2)
    with pytest.raises(ValueError):
        train_valid_split(x, [1, 2, 3], 0.2)
