# tests/test_constants.py
import pytest
from shard.constants import INPUT_LAYER, OUTPUT_LAYER


class TestConstants:
    """Test suite for shard/constants.py"""

    def test_input_layer_value(self):
        """Test INPUT_LAYER constant value"""
        assert INPUT_LAYER == -1

    def test_output_layer_value(self):
        """Test OUTPUT_LAYER constant value"""
        assert OUTPUT_LAYER == -2

    def test_constants_are_integers(self):
        """Test that constants are integers"""
        assert isinstance(INPUT_LAYER, int)
        assert isinstance(OUTPUT_LAYER, int)

    def test_constants_are_negative(self):
        """Test that constants are negative (distinguishing from regular layer indices)"""
        assert INPUT_LAYER < 0
        assert OUTPUT_LAYER < 0

    def test_constants_are_unique(self):
        """Test that constants have unique values"""
        assert INPUT_LAYER != OUTPUT_LAYER
