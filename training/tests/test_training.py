import pytest
import sys
sys.path.append("./")
sys.path.append("../../")

from training.util_training import get_from_module


def test_get_from_module():
    assert get_from_module("support", "SomeFunnyClass")().some_funny_property == 1
    assert get_from_module("support", "some_funny_func")() == 1
