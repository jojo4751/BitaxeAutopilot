"""
Basic test to verify testing framework is working
"""

import pytest
from datetime import datetime


def test_basic_functionality():
    """Test basic Python functionality"""
    assert 1 + 1 == 2
    assert "hello" == "hello"
    assert len([1, 2, 3]) == 3


def test_datetime_functionality():
    """Test datetime functionality"""
    now = datetime.now()
    assert isinstance(now, datetime)
    assert now.year >= 2023


@pytest.mark.unit
def test_marked_test():
    """Test pytest markers work"""
    assert True


def test_mock_basic():
    """Test basic mocking works"""
    from unittest.mock import Mock
    
    mock_obj = Mock()
    mock_obj.method.return_value = "test_value"
    
    result = mock_obj.method()
    assert result == "test_value"
    mock_obj.method.assert_called_once()


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality"""
    import asyncio
    
    async def async_function():
        await asyncio.sleep(0.01)
        return "async_result"
    
    result = await async_function()
    assert result == "async_result"