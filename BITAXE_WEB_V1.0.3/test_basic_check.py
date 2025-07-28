"""Basic test to verify testing framework is working"""

import pytest
from datetime import datetime


def test_basic_functionality():
    """Test basic Python functionality"""
    assert 1 + 1 == 2
    assert "hello" == "hello"
    assert len([1, 2, 3]) == 3
    print("Basic functionality test passed")


if __name__ == "__main__":
    test_basic_functionality()
    print("All basic tests passed - testing framework is working!")