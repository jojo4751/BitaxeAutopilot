"""
Simplified Pytest configuration and fixtures for BitAxe Web Management tests
"""

import os
import pytest
import tempfile
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import json

# Flask imports
from flask import Flask


@pytest.fixture(scope='session')
def app():
    """Create simple Flask application for testing"""
    app = Flask(__name__)
    app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test-secret-key'
    })
    
    @app.route('/health')
    def health():
        return {'status': 'healthy'}
    
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def sample_miner_data():
    """Generate sample miner data for testing"""
    return {
        'ip': '192.168.1.100',
        'hostname': 'test-miner',
        'temp': 65.5,
        'hashRate': 485.2,
        'power': 12.8,
        'voltage': 12.0,
        'frequency': 800,
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def mock_database():
    """Create mock database for testing"""
    db = Mock()
    db.get_session = Mock()
    db.init_database = Mock()
    return db


@pytest.fixture
def performance_metrics():
    """Performance metrics collection for benchmarking"""
    class PerformanceMetrics:
        def __init__(self):
            self.metrics = {}
            
        def start_timer(self, name):
            self.metrics[name] = {'start': datetime.now()}
            
        def end_timer(self, name):
            if name in self.metrics:
                self.metrics[name]['end'] = datetime.now()
                self.metrics[name]['duration'] = (
                    self.metrics[name]['end'] - self.metrics[name]['start']
                ).total_seconds()
                
        def get_duration(self, name):
            return self.metrics.get(name, {}).get('duration', 0)
            
        def assert_performance(self, name, max_duration):
            duration = self.get_duration(name)
            assert duration <= max_duration, f"{name} took {duration}s, expected <= {max_duration}s"
    
    return PerformanceMetrics()


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test"""
    yield
    
    # Clean up any test files or temporary directories
    test_dirs = ['test_models', 'test_data', 'test_logs']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


# Helper functions
def assert_success_response(response_json):
    """Assert response is successful"""
    assert 'success' in response_json
    assert response_json['success'] is True


def assert_error_response(response_json, expected_code=None):
    """Assert response contains error"""
    assert 'success' in response_json
    assert response_json['success'] is False
    assert 'error' in response_json
    
    error = response_json['error']
    assert 'code' in error
    assert 'message' in error
    if expected_code:
        assert error['code'] == expected_code